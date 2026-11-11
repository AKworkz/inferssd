#include "explicit-fixture-inputs.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gpu-services.h"
#include "vram-tier.cpp"
#include "test-ssd-vram-format-fixture.h"

#include <functional>

namespace {
int assertions = 0;
void require(bool ok, const char * message) {
    ++assertions;
    if (!ok) throw std::runtime_error(message);
}
void rejected(const std::function<void()> & fn) {
    bool failed = false;
    try { fn(); } catch (const std::exception &) { failed = true; }
    require(failed, "unsafe elastic operation was accepted");
}
kimi_vram_request request(size_t role) {
    kimi_vram_request r = {};
    r.layer = 1; r.expert = 7; r.weight = 1;
    for (int i = 0; i < 3; ++i) r.role[i] = {1,2,1000000,uint64_t(i)*role,role,3,4,5,6};
    return r;
}
void common_environment(fixture_inputs & inputs) {
    inputs.set("vram_async", "0");
    inputs.set("vram_deferred", "0");
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");
    inputs.set("tier.keep_across_requests", "1");
    inputs.set("tier.elastic", "1");
    inputs.set("tier.demand_reclaim", "0");
    inputs.set("tier.safety_blocks", "4096");
    inputs.set("tier.margin_mib", "0");
    inputs.set("tier.max_slots", "53");
    inputs.unset("tier.fail_malloc_at");
    inputs.unset("tier.fail_free_at");
}
}

int main() {
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services()); try {
    // A discrete GPU grows at allocation and at the decode phase boundary; unified memory grows at the token
    // boundary after the first decode step (test-ssd-vram-reclaim has its own cases). Everything else is the same.
    const bool unified = ggml_backend_cuda_device_is_unified(0);
    inputs.unset("tier.safety_blocks");
    inputs.unset("tier.margin_mib");
    { tier defaults(1, inputs.tier_policy()); require(defaults.elastic_safety_blocks==0 && defaults.elastic_safety_mib==32,
            "elastic runtime safety defaults changed"); }
    common_environment(inputs);
    const size_t role = lifecycle_role_bytes(), stride = 3*role;
    const auto key = request(role);
    std::vector<float> output(256);

    // Base and elastic slots share one block map. Tail IDs cross three CUDA
    // allocations, while the cache table remains a single logical global pool.
    inputs.set("tier.global", "1");
    {
        tier t(20*stride, inputs.tier_policy());
        int slot = -1;
        t.begin(17,1,true);lifecycle_prepare(t,&key,1,&slot);t.finish(output.data(),256);
        const size_t first_decode_slots=t.table->slots.size();
        t.phase_complete(true);
        require(t.table->slots.size()==first_decode_slots && t.grow_events==(unified?0:1),
                "duplicate decode phase grew the arena twice");
        // A tier created during prefill is measured against the large prompt graph. The
        // prefill_done callback restores the decode graph before the first
        // decode begin(), so it must re-arm and complete one grow here. Once
        // begin() adopts the arena, request_end must not attempt a second grow
        // around potentially live deferred work.
        t.adopt_next_decode=true;t.have_token=false;t.active=false;t.decode_growth_done=true;
        const uint64_t before_prefill_transition=t.grow_events;
        t.phase_complete(true);
        if(!unified){
            require(t.decode_growth_done && t.grow_events==before_prefill_transition+1,
                    "prefill-created tier decode graph transition did not re-arm growth");
            t.begin(17,2,true);
        }else{
            // Re-armed there, grown at the boundary after the first decode step (token 2 here).
            require(!t.decode_growth_done && t.grow_events==before_prefill_transition,
                    "unified memory grew at the decode phase boundary");
            t.begin(17,2,true);
            require(!t.decode_growth_done,"unified memory grew before the first decode step ended");
            t.begin(17,3,true);
            require(t.decode_growth_done && t.grow_events==before_prefill_transition+1 && t.table->slots.size()==53,
                    "unified memory did not grow after the first decode step");
        }
        const uint64_t after_prefill_adopt=t.grow_events;
        t.phase_complete(true);
        require(!t.adopt_next_decode && t.grow_events==after_prefill_adopt,
                "request end repeated prefill-created tier decode growth");
        t.prepare_prefill_reclaim();
        require(t.elastic_active && !t.arena && t.blocks.size()==2 && t.table->slots.size()==20,
                "elastic base was not split into 16-slot blocks");
        require(t.blocks[0].slot_count==16 && t.blocks[1].slot_count==4 &&
                t.slot_base(15)==t.blocks[0].data+15*stride && t.slot_base(16)==t.blocks[1].data,
                "cross-block slot address is wrong");
        const size_t stable_slots=t.table->slots.size(),stable_blocks=t.blocks.size(),stable_bytes=t.capacity_bytes;
        t.fail_malloc_at=t.malloc_calls+1;
        require(!t.allocate_block(16,"injected") && t.table->slots.size()==stable_slots &&
                t.blocks.size()==stable_blocks && t.capacity_bytes==stable_bytes,
                "failed elastic grow changed visible state");
        t.fail_malloc_at=0;
        require(t.allocate_block(16,"test") && t.allocate_block(1,"test") && t.table->slots.size()==37,
                "elastic grow boundary failed");
        const unsigned char values[3] = {15,16,36};
        for (int i = 0; i < 3; ++i) cuda_check(cudaMemset(t.slot_base(values[i]),values[i],1));
        unsigned char copied[3] = {};
        for (int i = 0; i < 3; ++i) cuda_check(cudaMemcpy(copied+i,t.slot_base(values[i]),1,cudaMemcpyDeviceToHost));
        require(!std::memcmp(values,copied,sizeof(values)),"cross-block device payload mapping failed");
        // A block given back keeps the tier's most valuable entry: a selected expert in the last block's slot moves,
        // bytes and all, into an empty slot below it before the block goes (compact_into).
        {
            auto & table=*t.table;const size_t last=table.slots.size()-1;
            table.change(table.slots[last],[&](vram_slot_table::entry & x){x={};x.key=key;x.keyed=true;x.valid=true;});
            table.layers[size_t(key.layer)].counts[size_t(key.expert)]=5;
            cuda_check(cudaMemset(t.slot_base(last),0xab,stride));
            require(t.free_last_block("test",false) && t.compaction_moves==1 && table.holds(key.layer,key.expert),
                    "a block given back took its selected expert with it");
            const int moved=table.find(key,[](const vram_slot_table::entry & e){return e.valid;});
            require(moved>=0 && size_t(moved)<last,"the moved expert is not held below the freed block");
            std::vector<unsigned char> bytes(stride);
            cuda_check(cudaMemcpy(bytes.data(),t.slot_base(size_t(moved)),stride,cudaMemcpyDefault));
            require(std::all_of(bytes.begin(),bytes.end(),[](unsigned char b){return b==0xab;}),
                    "the moved expert's bytes did not move with it");
            table.change(table.slots[size_t(moved)],[](vram_slot_table::entry & x){x={};});
            table.layers[size_t(key.layer)].counts[size_t(key.expert)]=0;
            require(t.allocate_block(1,"test") && table.slots.size()==37,"the freed block did not come back");
        }

        kimi_vram_ready_lease lease = {}; int ready = 0;
        t.acquire_ready(17,&key,1,&lease,&ready);
        rejected([&]{t.prepare_prefill_reclaim();});
        require(t.table->slots.size()==37,"exclusive lease changed capacity");
        t.release_ready(lease);
        t.prepare_prefill_reclaim();
        require(t.table->slots.size()==20 && t.blocks.size()==2 && t.reclaimer_armed,
                "proactive shrink did not return to the base boundary");

        // The allocator sees an injected first OOM, asks the armed tier for
        // one block, and retries before returning a buffer to the caller. With
        // no elastic block left this deliberately consumes the base tail.
        require(ggml_cuda_oom_reclaimer_inject_failures(t.device,1),"OOM injection was not armed");
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(t.device),4096);
        require(buffer!=nullptr,"allocator did not retry after reclaim");
        ggml_backend_buffer_free(buffer);
        require(t.table->slots.size()==16 && t.base_blocks_reclaimed==1 && t.reactive_blocks==1,
                "reactive reclaim did not continue into the base");
        t.phase_complete(false);
    }

    // A base block that does not fit (device memory moved after the base's measure) ends the base at the blocks
    // already allocated, and the table follows them; the first request is served from them, a discrete GPU's
    // decode growth adds elastic blocks after them, and a prefill gives back to that base.
    {
        inputs.set("tier.fail_malloc_at", "2");
        tier t(20*stride, inputs.tier_policy());
        inputs.unset("tier.fail_malloc_at");
        int slot = -1;
        t.begin(19,1,true);lifecycle_prepare(t,&key,1,&slot);t.finish(output.data(),256);
        require(t.elastic_active && t.base_slots==16 && t.blocks[0].slot_count==16 &&
                t.table->slots.size()==t.slot_locations.size() && t.capacity_bytes==t.slot_locations.size()*stride,
                "a base block that did not fit did not end the base at the blocks allocated");
        t.phase_complete(true);
        t.prepare_prefill_reclaim();
        require(t.blocks.size()==1 && t.slot_locations.size()==16 && t.table->slots.size()==16,
                "a prefill did not give back to the base the allocation ended at");
        t.phase_complete(false);
    }

    // A failed free retains the block/table mapping and poisons the tier. The
    // test clears only its injected fault so destruction can release the GPU.
    {
        tier t(20*stride, inputs.tier_policy());int slot=-1;
        t.begin(18,1,true);lifecycle_prepare(t,&key,1,&slot);t.finish(output.data(),256);
        t.prepare_prefill_reclaim();
        const size_t old_slots=t.table->slots.size(),old_blocks=t.blocks.size();
        t.fail_free_at=t.free_calls+1;size_t freed=0;
        rejected([&]{t.reclaim_one(4096,&freed);});
        require(t.poisoned && !freed && t.table->slots.size()==old_slots && t.blocks.size()==old_blocks,
                "failed free did not preserve metadata and fail closed");
        t.fail_free_at=0;t.reclaimer_armed=false;
    }

    // Demand mode retains the decode-grown tail until a real backend-buffer
    // allocation fails. One callback frees exactly one complete tail block,
    // and the backend retries the same allocation.
    inputs.set("tier.demand_reclaim", "1");
    inputs.set("tier.max_slots", "37");
    {
        tier t(20*stride, inputs.tier_policy());int slot=-1;
        t.begin(20,1,true);lifecycle_prepare(t,&key,1,&slot);t.finish(output.data(),256);
        if(unified)t.begin(20,2,true); // unified memory grows at the boundary after the first decode step
        require(t.table->slots.size()==37,"demand fixture did not create a decode-grown tail");
        const size_t before=t.table->slots.size();
        t.prepare_prefill_reclaim();
        require(t.elastic_demand_reclaim && t.reclaimer_armed && t.table->slots.size()==before &&
                t.proactive_blocks==0,"demand arm discarded capacity proactively");
        require(ggml_cuda_oom_reclaimer_inject_failures(t.device,1),"demand OOM injection was not armed");
        ggml_backend_buffer_t buffer=ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(t.device),4096);
        require(buffer!=nullptr,"demand allocator did not retry after reclaim");
        ggml_backend_buffer_free(buffer);
        require(t.table->slots.size()==36 && t.reactive_blocks==1 && t.base_blocks_reclaimed==0 &&
                t.phase_reclaim_requests==1,"demand reclaim did not free exactly the physical tail block");
        t.phase_complete(false);
    }

    // The grouped prefill engine reserves after the graph's reserve disarmed the reclaimer, while the tier is
    // quiescent: the tier arms it for the reserve, so an engine allocation that runs out of memory gets the tail block
    // back and succeeds, and the reclaimer is disarmed again after it. (Unified memory reclaims in any phase.)
    {
        tier t(20*stride, inputs.tier_policy());int slot=-1;
        t.begin(20,1,true);lifecycle_prepare(t,&key,1,&slot);t.finish(output.data(),256);
        if(unified)t.begin(20,2,true);
        if(!t.group){
            std::printf("case=engine_reserve_reclaim status=SKIP reason=no_grouped_engine_for_the_fixture_format\n");
        }else{
            t.prepare_prefill_reclaim();t.phase_complete(false);
            require(!t.reclaimer_armed,"the graph's phase completion did not disarm the reclaimer");
            const size_t before=t.table->slots.size();const uint64_t reactive=t.reactive_blocks;
            require(ggml_cuda_oom_reclaimer_inject_failures(t.device,1),"engine OOM injection was not armed");
            t.reserve_prefill_engine(64);
            require(t.group_rows==64 && t.table->slots.size()<before && t.reactive_blocks==reactive+1 && !t.reclaimer_armed,
                    "an engine reserve out of memory did not get a block back from the tier");
            // the engine's fixed scratch, as this reservation measured it, prices the next one after a release
            vram_group_geometry g={};t.group_api->geometry(t.group,&g);
            require(g.fixed_bytes>0 && t.group_fixed_bytes==g.fixed_bytes,"the tier did not keep the engine's measured fixed bytes");
            t.release_group();
            require(t.group_rows==0 && t.group_fixed_bytes==g.fixed_bytes,"a release forgot the engine's measured fixed bytes");
            t.reserve_prefill_engine(64);
            require(t.group_rows==64 && t.group_fixed_bytes==g.fixed_bytes,"the engine reserved again a different fixed size");
            std::printf("case=engine_reserve_reclaim status=PASS slots=%zu->%zu\n",before,t.table->slots.size());
        }
    }
    inputs.set("tier.demand_reclaim", "0");
    inputs.set("tier.max_slots", "53");

    // Non-global uniform policy keeps the same single allocation
    // even when the process-wide elastic request is present.
    inputs.set("tier.global", "0");
    {
        tier t(2*stride, inputs.tier_policy());int slot=-1;
        t.begin(19,1,true);lifecycle_prepare(t,&key,1,&slot);t.finish(output.data(),256);
        require(!t.elastic_active && t.arena && t.blocks.empty() &&
                t.slot_base(1)==t.arena+stride,"non-global path no longer uses its single arena");
    }

    // Replacements are priced from the second decode token, by the costs the sink reports for each layer of the first
    // (a use from the RAM cache, one from the drive) against the resident cost. Until then the fixed minimum count
    // (2.5 here) decides. From the second token a free slot takes a count-1 miss where a RAM hit costs more than a
    // resident expert, and none where it costs less; one layer whose drive reads cost 5 times the others' weighs little.
    inputs.set("tier.global", "1");
    inputs.set("tier.policy", "count_decay2");
    inputs.set("tier.min_count", "2.5");
    for (const double ram_over_hit : {4.0, 0.5}) {
        tier t(20*stride, inputs.tier_policy());int slot=-1;
        auto r=key;
        t.begin(21,1,true);
        const double hit=100,drive=2000;
        for (int layer = 1; layer <= 40; ++layer) {
            r.layer=layer;lifecycle_prepare(t,&r,1,&slot);
            require(!t.adaptive_admission && t.table->admissions[0]<0,"the first decode token was priced");
            t.finish(output.data(),256);
            policy_set_miss_cost(&t,ram_over_hit*hit,layer==1?5*drive:drive);
        }
        {std::lock_guard<std::mutex> lock(t.mutex);t.resident_us_per_expert=hit;}
        t.begin(21,2,true);
        require(t.adaptive_admission && t.admission_updates==1 && t.table->candidate_saving==t.table->ram_saving &&
                std::fabs(t.table->ram_saving-std::max(0.0,(ram_over_hit-1)*hit))<1e-6 &&
                std::fabs(t.table->drive_saving/(drive-hit)-1)<0.1,
                "the second decode token was not priced from the first token's costs");
        r.layer=1;r.expert=8;lifecycle_prepare(t,&r,1,&slot);
        require(ram_over_hit>1?t.table->admissions[0]>=0:t.table->admissions[0]<0,
                "a free slot did not follow the priced rule in the second decode token");
        t.finish(output.data(),256);
    }
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");

    // At load (kimi_vram_elastic_api reserve) a unified-memory tier takes its storage and grows into what is free;
    // a prefill keeps it and the first decode adopts it. A discrete GPU's automatic tier takes one block at load and
    // grows into the card at the first phase boundary after a prefill pass's memory settled, or at the first decode
    // token (phase_complete); a byte budget leaves the arena to its first decode.
    if (!unified) {
        const size_t mx = 36864;
        kimi_vram_request r = {};r.layer=1;r.weight=1;
        for (int i = 0; i < 3; ++i) r.role[i].bytes = mx;
        const int32_t layers[] = {1,2};
        const kimi_vram_expert_format f = {256,256,int32_t(GGML_TYPE_MXFP4),int32_t(GGML_TYPE_MXFP4),64,2,layers};
        for (bool settle : {true, false}) {
            tier t(tier::automatic_budget, inputs.tier_policy());
            t.reserve(r,f);
            require(t.has_storage() && t.table && t.growth_deferred && t.grow_events==0 && t.slot_locations.size()==16 &&
                    t.adopt_next_decode, "a discrete GPU's automatic tier did not take one block at load");
            t.begin(22,0,false);
            require(t.growth_deferred && t.grow_events==0 && t.slot_locations.size()==16,
                    "a discrete tier grew before a prefill pass's memory settled");
            if (settle) {
                // two layer boundaries of a pass that find the same free memory
                t.observe_settle();t.observe_settle();
                require(t.settled,"two equal layer boundaries did not settle the tier's memory");
                t.begin(22,0,false);
            } else {
                t.phase_complete(true);
            }
            require(!t.growth_deferred && t.grow_events==1 && t.max_slots>16 && t.slot_locations.size()==t.max_slots,
                    settle ? "a discrete tier made at load did not grow at the phase boundary after its memory settled" :
                             "a discrete tier made at load did not grow at the first decode token");
            t.begin(22,1,true);
            require(t.has_storage() && t.table && !t.adopt_next_decode && t.arena_epoch==22,
                    "the first decode did not adopt the discrete tier made at load");
        }
        // At load after the load's passes (preload_slots): a tier whose memory settled grows into the card, then takes
        // experts no request has named into its empty slots, preloaded (vram-slot-table.h), one slot an expert.
        {
            tier t(tier::automatic_budget, inputs.tier_policy());
            t.reserve(r,f);
            t.observe_settle();t.observe_settle();
            const size_t empty=t.preload_slots();
            require(!t.growth_deferred && t.grow_events==1 && empty==t.slot_locations.size() && empty>16,
                    "the tier did not grow into the card before its preload");
            kimi_vram_request key=r;key.expert=5;
            std::vector<char> roles(3*mx);
            for (size_t i = 0; i < roles.size(); ++i) roles[i]=char(uint8_t(i*13+7));
            const void * data[3]={roles.data(),roles.data()+mx,roles.data()+2*mx};
            require(t.preload(key,data) && !t.preload(key,data) && t.preload_slots()==empty-1,
                    "a preload did not take one empty slot, or took a second for the same expert");
            const int slot=t.table->find(key,[](const vram_slot_table::entry & e){return e.valid;});
            require(slot>=0 && t.table->slots[size_t(slot)].preloaded && t.table->holds(1,5),
                    "the preloaded expert is not a held, preloaded entry");
            std::vector<char> got(3*mx);
            cuda_check(cudaMemcpy(got.data(),t.slot_base(size_t(slot)),3*mx,cudaMemcpyDeviceToHost));
            require(!std::memcmp(got.data(),roles.data(),3*mx),"a preloaded role holds other bytes");
        }
    }
    {
        const size_t mx = 36864; // one 256 x 256 MXFP4 role, padded to 4 KiB
        kimi_vram_request r = {};r.layer=1;r.weight=1;
        for (int i = 0; i < 3; ++i) r.role[i].bytes = mx;
        const int32_t layers[] = {1,2};
        const kimi_vram_expert_format f = {256,256,int32_t(GGML_TYPE_MXFP4),int32_t(GGML_TYPE_MXFP4),8,2,layers};
        {
            tier t(10*3*mx, inputs.tier_policy());
            t.reserve(r,f);
            if (!unified) {
                require(!t.has_storage() && !t.table,"a discrete GPU took its tier's storage at load");
            } else {
                require(t.has_storage() && t.adopt_next_decode && t.decode_growth_done && t.grow_events==1 &&
                        t.table->slots.size()==16 && t.upload_us_per_expert>0,
                        "unified memory did not reserve and grow its tier at load");
                t.begin(22,0,false);
                // A prefill's seed: an expert the pass routes, planned into an empty slot before its layer's prepare,
                // its three roles copied from the payload the CPU computed it from, published when the decode phase
                // completes the prefill.
                {
                    kimi_vram_request key=r;key.expert=5;
                    t.plan_sink_seeds(&key,1,{3});
                    require(t.seeds.size()==1 && t.sink_seed_slot.size()==1 && t.sink_seed_slot[0]>=0,
                            "no empty slot for a seed");
                    const int slot=t.sink_seed_slot[0];
                    char * payload=nullptr;cuda_check(cudaMallocHost(reinterpret_cast<void **>(&payload),3*mx));
                    for (size_t i = 0; i < 3*mx; ++i) payload[i]=char(uint8_t(i*7+3));
                    const void * roles[3]={payload,payload+mx,payload+2*mx};
                    t.seed_from_host(0,roles);
                    t.wait_host_seed(0);
                    t.phase_complete(true);
                    require(t.table->slots[size_t(slot)].valid && t.table->slots[size_t(slot)].seed && t.seeds.empty(),
                            "the complete seed was not published");
                    std::vector<char> got(3*mx);
                    cuda_check(cudaMemcpy(got.data(),t.slot_base(size_t(slot)),3*mx,cudaMemcpyDeviceToHost));
                    require(!std::memcmp(got.data(),payload,3*mx),"a seeded role holds other bytes");
                    cuda_check(cudaFreeHost(payload));
                }
                require(t.has_storage() && t.adopt_next_decode,"a prefill before the first decode dropped the reserved tier");
                t.begin(22,1,true);
                require(t.has_storage() && t.table && !t.adopt_next_decode && t.arena_epoch==22,
                        "the first decode did not adopt the tier reserved at load");
            }
        }
        // A layer's seeds: its share of the empty slots, those free over the layers not planned yet in this prefill
        // (what a layer leaves rolls forward), the prompt's most-routed experts first, an expert already resident
        // skipped.
        if (unified) {
            const int32_t three[] = {1,2,3};
            const kimi_vram_expert_format f3 = {256,256,int32_t(GGML_TYPE_MXFP4),int32_t(GGML_TYPE_MXFP4),8,3,three};
            inputs.set("tier.max_slots", "16");
            tier q(10*3*mx, inputs.tier_policy());
            inputs.set("tier.max_slots", "53");
            q.reserve(r,f3);
            require(q.has_storage() && q.table->slots.size()==16,"the quota test's tier has another size");
            kimi_vram_request resident=r;resident.layer=3;resident.expert=0;
            const int held=q.table->reserve_cold_prefill(resident);
            require(held>=0,"no slot for the resident expert");
            q.table->publish_cold_prefill(size_t(held),resident);
            const std::vector<int32_t> counts[3]={{0,5,0,3,0,0,1,0},{1,2,3,4,5,6,7,8},{9,1,1,1,1,1,1,1}};
            for (int layer = 1; layer <= 3; ++layer) {
                std::vector<kimi_vram_request> routed(8,r);
                for (int e = 0; e < 8; ++e) { routed[size_t(e)].layer=layer;routed[size_t(e)].expert=e; }
                q.plan_sink_seeds(routed.data(),8,counts[layer-1]);
            }
            std::vector<int> planned[4];
            for (const auto & seed : q.seeds) planned[seed.key.layer].push_back(seed.key.expert);
            // 15 free: layer 1 a third (5) but routed 3; layer 2 half of the 12 left; layer 3 the 6 left, not expert 0
            require(planned[1]==std::vector<int>({1,3,6}) && planned[2]==std::vector<int>({7,6,5,4,3,2}) &&
                    planned[3]==std::vector<int>({1,2,3,4,5,6}) && q.table->free_slots(1)==0 && q.seeds_planned==15,
                    "the seed quota did not follow the free slots, the routing counts or residency");
            q.cancel_seeds();
            require(q.table->free_slots(1)==15,"cancelled seeds kept their slots");
        }
    }

    std::printf("REAL_CUDA_ELASTIC_PASS assertions=%d unified=%d\n",assertions,int(unified));
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr,"REAL_CUDA_ELASTIC_FAIL: %s\n",e.what());
    return 1;
}
}
