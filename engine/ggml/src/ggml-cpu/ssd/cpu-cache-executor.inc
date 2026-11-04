static void sink_location_provider_initialize(kimi_cpu_sink * sink) {
    if(!sink->vram.location_api)return;
    const auto & api=*sink->vram.location_api->view;
    if(!api.acquire_native || !api.begin_layer || !api.launch || !api.admit || !api.finish || !api.cancel || !api.release)
        throw std::runtime_error("incomplete VRAM cache location provider");
    sink->locations=api;sink->locations_context=sink->vram.context;
    sink->locations.finished=[](void * context,kimi_cpu_sink * owner,char * error,size_t size)->int {
        try{
            if(context!=owner->vram.context || !owner->vram.ram_leases.empty())throw std::runtime_error("VRAM finished owner/lease mismatch");
            owner->vram.drop_ready(owner->cache,owner->vram.location_sources);return 0;
        }catch(const std::exception & e){sink_error(error,size,e.what());return 1;}
    };
}
extern "C" bool kimi_cpu_sink_set_cache_locations(kimi_cpu_sink * sink,
        const kimi_cache_location_api * api, void * context, char * error, size_t size) {
    sink_error(error,size,""); if (!sink) { sink_error(error,size,"null sink"); return false; }
    std::lock_guard<sink_run_mutex> lock(sink->run_mutex);
    if (api && (api->version!=1 || api->size!=sizeof(*api) || !api->acquire_native || !api->begin_layer ||
            !api->launch || !api->finish || !api->cancel || !api->release)) {
        sink_error(error,size,"invalid cache location API v1"); return false;
    }
    sink->locations=api?*api:kimi_cache_location_api{}; sink->locations_context=api?context:nullptr; return true;
}
// Included after the ordinary executor. All mutable cache/ring state is owned
// by run_mutex; payload leases last through compute barriers and CQE drain.
static uint32_t cache_compute_resident(kimi_cpu_sink * sink, const ggml_tensor * up,
        const ggml_tensor * gate, const ggml_tensor * down, const int32_t * ids, int count) {
    auto & cache = sink->cache;
    if (!cache.active || cache.paused || !cache.arena) { return 0; }
    const ggml_tensor * tensors[] = {up,gate,down};
    for (int role=0; role<3; ++role) {
        if (!tensors[role] || ((tensors[role]->nb[2]+4095) & ~size_t(4095)) != cache.role_bytes[role]) { return 0; }
    }
    int layer = -1;
    if (std::sscanf(up->name,"blk.%d.ffn_up_exps.weight",&layer) != 1) { return 0; }
    auto found = std::find_if(cache.layers.begin(),cache.layers.end(),[&](const sink_cache::layer & l){return l.id==layer;});
    if (found==cache.layers.end()) { return 0; }
    const size_t li = size_t(found-cache.layers.begin()); uint32_t mask=0;
    for (int n=0;n<count;++n) {
        for (size_t i=found->begin;i<found->begin+found->count;++i) {
            const auto & e=cache.entries[i];
            if (e.valid && e.layer_index==li && e.id==ids[n]) { mask |= 1u<<n; break; }
        }
    }
    return mask;
}
static void cache_compute_print(const kimi_cache_compute_stats & a, uint64_t epoch, uint64_t token, int layer,
        const kimi_cpu_sink_stats & s, bool success, uint32_t vram_mask=0) {
    if (!a.plan.config.trace) { return; }
    const auto & p=a.plan;
    std::fprintf(stderr,"[kimi_cache_compute] {\"schema\":1,\"epoch\":%llu,\"token\":%llu,\"layer\":%d,"
        "\"vram_mask\":%u,\"success\":%s,\"mode\":%d,\"guarded\":%d,\"route_us\":%llu,\"prepared_us\":%llu,\"end_us\":%llu,"
        "\"original\":%u,\"cached\":%u,\"base\":%u,\"protected\":%u,\"pin\":%u,\"bonus\":%u,\"replacement\":%u,"
        "\"dropped\":%u,\"final\":%u,\"computed\":%u,\"deadline_applied\":%d,\"deadline_us\":%llu,"
        "\"baseline_miss_bytes\":%llu,\"submitted_bytes\":%llu,\"completed_bytes\":%llu,\"unused_read_bytes\":%llu,"
        "\"wait_io_us\":%.3f,\"compute_us\":%.3f,\"native\":[",
        (unsigned long long)epoch,(unsigned long long)token,layer,vram_mask,success?"true":"false",p.config.mode,p.guarded,
        (unsigned long long)a.route_visible_us,(unsigned long long)a.prepared_us,(unsigned long long)a.finished_us,
        p.original_mask,p.cached_mask,p.base_mask,p.protected_mask,p.pin_mask,p.bonus_mask,p.replacement_mask,
        p.dropped_mask,p.final_mask,a.computed_mask,p.deadline_applied,(unsigned long long)p.config.deadline_us,
        (unsigned long long)a.baseline_miss_bytes,(unsigned long long)a.submitted_bytes,(unsigned long long)s.read_bytes,
        (unsigned long long)a.unused_read_bytes,s.wait_io_us,s.compute_us);
    for (int i=0;i<p.native_count;++i) {
        std::fprintf(stderr,"%s{\"slot\":%d,\"id\":%d,\"native_w\":%.9g,\"final_w\":%.9g,\"bytes\":%llu,\"t\":[",
            i?",":"",i,p.native_ids[i],p.native_weights[i],a.final_weights[i],(unsigned long long)a.completed_bytes[i]);
        for (int j=0;j<8;++j) { std::fprintf(stderr,"%s%llu",j?",":"",(unsigned long long)a.events[i][j]); }
        std::fprintf(stderr,"]}");
    }
    std::fprintf(stderr,"]}\n");
}
extern "C" bool kimi_cpu_sink_run_cache_compute(kimi_cpu_sink * sink,
        const ggml_tensor * up,const ggml_tensor * gate,const ggml_tensor * down,const float * input,
        const int32_t * ids,const float * weights,int nslots,
        const int32_t * native_ids,const float * native_weights,int native_count,
        float routed_scale,bool guarded,const kimi_cache_compute_config * config,
        float situ_beta,float situ_linear_beta,float * output,kimi_cpu_sink_stats * stats,
        kimi_cache_compute_stats * decision,char * error,size_t error_size) {
    sink_error(error,error_size,""); if (stats) {*stats={};} if (decision) {*decision={};}
    kimi_cache_compute_stats a{}; a.route_visible_us=ggml_time_us();
    if (!sink || !config || !up || !gate || !down || !input || !output ||
            !std::isfinite(routed_scale) || routed_scale<=0 ||
            !kimi_cache_compute_plan_init(config,native_ids,native_weights,native_count,ids,weights,nslots,0,guarded,
                &a.plan,error,error_size)) {
        if (!error || !error_size || !*error) {sink_error(error,error_size,"invalid cache-compute input");} return false;
    }
    int layer=-1; std::sscanf(up->name,"blk.%d.ffn_up_exps.weight",&layer);
    if (a.plan.config.mode==KIMI_CACHE_COMPUTE_OFF && !sink->locations.version) {
        kimi_cpu_sink_stats local{};
        const bool ok=kimi_cpu_sink_run(sink,up,gate,down,input,ids,weights,nslots,situ_beta,situ_linear_beta,output,&local,error,error_size);
        a.computed_mask=ok?a.plan.original_mask:0;a.finished_us=ggml_time_us();
        for(int i=0;i<native_count;++i){a.final_weights[i]=a.plan.original_weights[i];}
        if(stats){*stats=local;}if(decision){*decision=a;}
        cache_compute_print(a,sink->cache.epoch,sink->cache.token,layer,local,ok);return ok;
    }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex);
    const auto start=sink_clock::now();const uint64_t completed_before=sink->ring.completed();
    sink->current={};sink->requests.clear();sink->begin_observation(up,true);
    // The dedicated trace represents uncomputed/dropped slots too; legacy
    // TIMELINE assumes every request was computed and is not emitted here.
    sink->timeline_active=false;sink->cache_compute_active=true;
    std::array<int,16> request_for_native{};request_for_native.fill(-1);
    const auto & locations=sink->locations; void * location_context=sink->locations_context;
    std::array<kimi_cache_location_candidate,16> candidates{};
    std::array<kimi_cache_location_lease,16> leases{};
    std::vector<float> gpu_output;
    uint32_t gpu_mask=0; bool provider_acquired=false,provider_begun=false,gpu_launched=false;
    uint64_t gpu_begin_us=0;
    auto checked=[&](int code) { if(code){throw std::runtime_error(error&&*error?error:"cache location callback failed");} };
    auto release_leases=[&](uint32_t mask) {
        std::array<kimi_cache_location_lease,16> selected{};int count=0;
        for(int n=0;n<native_count;++n){if((mask&(1u<<n))&&leases[n].handle){selected[count++]=leases[n];}}
        if(count){checked(locations.release(location_context,selected.data(),count,error,error_size));}
        for(int n=0;n<native_count;++n){if(mask&(1u<<n)){leases[n]={};}}
    };
    auto collect=[&] {
        a.finished_us=ggml_time_us();
        for(size_t i=0;i<sink->requests.size();++i) {
            const auto & r=sink->requests[i];
            const int n=int(std::find(native_ids,native_ids+native_count,r.id)-native_ids);
            if(n>=native_count){continue;}
            std::copy(r.events,r.events+8,a.events[n]);a.completed_bytes[n]=r.read_bytes;
            if(r.events[2]){for(const auto & role:r.role){a.submitted_bytes+=role.nbytes;}}
            if(a.plan.dropped_mask&(1u<<n)){a.unused_read_bytes+=r.read_bytes;}
        }
    };
    try {
        sink->begin_participants(true);
        if(!sink->ready_first || sink->early_up_gate || !std::isfinite(situ_beta) || situ_beta<=0 ||
                !std::isfinite(situ_linear_beta) || sink->poisoned) {
            throw std::runtime_error("cache-compute requires READY_FIRST=1, EARLY_UP_GATE=0 and a healthy sink");
        }
        uint32_t before=cache_compute_resident(sink,up,gate,down,native_ids,native_count);
        if(locations.version) {
            const ggml_tensor * tensors[]={up,gate,down};
            // Bound metadata before provider lookup or its output allocation.
            // The CPU-only path performs these checks in prepare().
            for(const auto * t:tensors) {
                if(t->type!=GGML_TYPE_Q3_K||!ggml_is_contiguous(t)||t->ne[0]<=0||t->ne[1]<=0||t->ne[2]<=0||
                    t->ne[3]!=1||t->ne[0]>16384||t->ne[1]>16384||t->ne[0]%256) {
                    throw std::runtime_error("cache locations require valid contiguous Q3_K metadata");
                }
            }
            if(up->ne[0]!=gate->ne[0]||up->ne[1]!=gate->ne[1]||up->ne[2]!=gate->ne[2]||
                down->ne[0]!=up->ne[1]||down->ne[1]!=up->ne[0]||down->ne[2]!=up->ne[2]) {
                throw std::runtime_error("cache location tensor shape mismatch");
            }
            sink_direct_check direct;
            for(int n=0;n<native_count;++n) {
                auto & c=candidates[n];c.epoch=sink->cache.epoch;c.token=sink->cache.token;c.layer=layer;c.expert=native_ids[n];
                if(native_ids[n]>=up->ne[2]){throw std::runtime_error("native ID exceeds tensor metadata");}
                for(int role=0;role<3;++role) {
                    auto & source=c.role[role];
                    if(!sink->resolve(tensors[role]->name,native_ids[n],&source)) {
                        throw std::runtime_error("cache location source resolve failed");
                    }
                    const size_t bytes=tensors[role]->nb[2];
                    if(!direct(source.fd_direct)||source.offset%4096||source.nbytes%4096||source.nbytes<bytes||source.nbytes>bytes+4095) {
                        throw std::runtime_error("invalid cache location direct source");
                    }
                }
            }
            sink->vram.capture_locations(candidates.data(),native_count,int(up->ne[0]),int(up->ne[1]));
            provider_acquired=true;
            checked(locations.acquire_native(location_context,candidates.data(),native_count,leases.data(),error,error_size));
            for(int n=0;n<native_count;++n){if(leases[n].handle){gpu_mask|=1u<<n;}}
            before|=gpu_mask;
        }
        if(!kimi_cache_compute_plan_init(config,native_ids,native_weights,native_count,ids,weights,nslots,before,guarded,&a.plan,error,error_size)) {
            throw std::runtime_error(error&&*error?error:"cache-compute planning failed");
        }
        if(locations.version) {
            release_leases(gpu_mask&~a.plan.pin_mask);gpu_mask&=a.plan.pin_mask;
            provider_begun=true;
            checked(locations.begin_layer(location_context,candidates.data(),native_count,a.plan.base_mask,a.plan.pin_mask,error,error_size));
            gpu_output.resize(size_t(native_count)*up->ne[0]);
        }
        std::vector<int32_t> work_ids;std::vector<float> work_weights;std::vector<uint8_t> cache_only;
        for(int i=0;i<a.plan.original_count;++i) {
            const int n=a.plan.original_order[i];if(!(a.plan.base_mask&(1u<<n))||(gpu_mask&(1u<<n))){continue;}
            work_ids.push_back(native_ids[n]);work_weights.push_back(1.0f);cache_only.push_back(0);
        }
        for(int n=0;n<native_count;++n) {
            if((a.plan.pin_mask&~a.plan.base_mask&~gpu_mask)&(1u<<n)) {
                work_ids.push_back(native_ids[n]);work_weights.push_back(1.0f);cache_only.push_back(1);
            }
        }
        sink->prepare(up,gate,down,work_ids.data(),work_weights.data(),int(work_ids.size()),true,cache_only.data());
        uint32_t actual_cached=gpu_mask;uint64_t stride=0;
        for(size_t i=0;i<sink->requests.size();++i) {
            const auto & r=sink->requests[i];const int n=int(std::find(native_ids,native_ids+native_count,r.id)-native_ids);
            if(n>=native_count){throw std::runtime_error("prepared expert outside native top16");}
            request_for_native[n]=int(i);if(r.cache_hit>=0){actual_cached|=1u<<n;}
            if(!stride){for(const auto & role:r.role){stride+=role.nbytes;}}
        }
        // A headroom pause or source invalidation can withdraw a cache hit
        // before any worker/IO starts. Replan from actual pinned hits; extras
        // still cannot create SSD reads. Resource validation reports the pause.
        if(!kimi_cache_compute_plan_init(config,native_ids,native_weights,native_count,ids,weights,nslots,actual_cached,guarded,&a.plan,error,error_size)) {
            throw std::runtime_error(error&&*error?error:"cache-compute pin validation failed");
        }
        if(!stride){for(const auto * tensor:{up,gate,down}){stride+=(tensor->nb[2]+4095)&~size_t(4095);}}
        // prepare() sees only the CPU subset. Keep ordinary demand accounting
        // independent of location; cache-only GPU jobs are not SSD demand.
        sink->current.demand_bytes+=uint64_t(std::bitset<32>(a.plan.base_mask&gpu_mask).count())*stride;
        for(int n=0;n<native_count;++n){if((a.plan.original_mask&~before)&(1u<<n)){a.baseline_miss_bytes+=stride;}}
        a.prepared_us=ggml_time_us();
        sink->activation_kind=KIMI_CPU_SINK_ACTIVATION_SITU;sink->swiglu_limit=0.0f;
        sink->beta=situ_beta;sink->linear_beta=situ_linear_beta;sink->weight=1.0f;
        auto phase=sink_clock::now();sink->quantize_input(input,sink->quant_input.data(),sink->latent);
        sink->current.input_quantize_us=elapsed_us(phase);sink->current.compute_us+=sink->current.input_quantize_us;
        std::fill(output,output+sink->latent,0.0f);
        {
            std::lock_guard<std::mutex> lock(sink->io_mutex);sink->io_error.clear();sink->io_cancel=false;
            if(!sink->io_requests.empty()){sink->io_done=false;sink->io_job=true;}
        }
        sink->io_cv.notify_one();
        const uint64_t deadline=a.route_visible_us+a.plan.config.deadline_us;
        for(;;) {
            int selected=-1;bool launch_gpu=false;
            {
                std::unique_lock<std::mutex> lock(sink->io_mutex);
                if(!sink->io_error.empty()){throw std::runtime_error(sink->io_error);}
                uint32_t ready_now=actual_cached,ready_before=actual_cached;
                for(int n=0;n<native_count;++n) {
                    const int j=request_for_native[n];if(j<0){continue;}const auto & r=sink->requests[j];
                    if(r.ready_us){ready_now|=1u<<n;if(r.ready_us<=deadline){ready_before|=1u<<n;}}
                }
                const uint64_t now=ggml_time_us();kimi_cache_compute_plan_update(&a.plan,ready_before,now-a.route_visible_us);
                // Release only fully completed unused reads. This also returns
                // ring credits when depth is smaller than the planned S.
                for(int n=0;n<native_count;++n) {
                    const uint32_t bit=1u<<n;const int j=request_for_native[n];if(j<0){continue;}
                    const auto & r=sink->requests[j];
                    if(r.cache_only||r.cache_hit>=0||!(a.plan.dropped_mask&bit)||(a.released_mask&bit)||!r.ready_us){continue;}
                    auto & slot=sink->slots[r.io_slot];
                    if(slot.request!=size_t(j)||slot.state!=sink_slot::ready){throw std::runtime_error("late read lost its ring lease");}
                    sink->cache.admit(r,sink->current);slot.state=sink_slot::empty;a.released_mask|=bit;sink->io_cv.notify_all();
                }
                const uint32_t todo=a.plan.final_mask&~a.computed_mask&~gpu_mask;
                const uint32_t undecided=a.plan.base_mask&~a.plan.protected_mask&~ready_before;
                launch_gpu=locations.version&&!gpu_launched&&(a.plan.final_mask&gpu_mask)&&
                    (a.plan.config.mode!=KIMI_CACHE_COMPUTE_HYBRID||a.plan.deadline_applied||!undecided);
                for(int pass=0;pass<2&&selected<0;++pass) {
                    for(size_t j=0;j<sink->requests.size();++j) {
                        const auto & r=sink->requests[j];const int n=int(std::find(native_ids,native_ids+native_count,r.id)-native_ids);
                        if((todo&ready_now&(1u<<n)) && bool(a.plan.base_mask&(1u<<n))==(pass==0)){selected=int(j);break;}
                    }
                }
                if(selected<0&&!launch_gpu) {
                    if(!todo&&sink->io_done&&(!(a.plan.final_mask&gpu_mask)||gpu_launched)){break;}
                    const auto wait=sink_clock::now();
                    if(a.plan.config.mode==KIMI_CACHE_COMPUTE_HYBRID&&!a.plan.deadline_applied&&now<deadline) {
                        sink->io_cv.wait_for(lock,std::chrono::microseconds(deadline-now));
                    } else {sink->io_cv.wait(lock);}
                    sink->current.wait_io_us+=elapsed_us(wait);continue;
                }
            }
            if(launch_gpu) {
                std::array<kimi_cache_location_job,16> jobs{};int count=0;
                for(int n=0;n<native_count;++n){if(a.plan.final_mask&gpu_mask&(1u<<n)) {
                    jobs[count++]={leases[n],n,gpu_output.data()+size_t(n)*sink->latent};
                }}
                gpu_begin_us=ggml_time_us();gpu_launched=true;
                checked(locations.launch(location_context,jobs.data(),count,sink->quant_input.data(),sink->quant_input.size(),
                    sink->latent,sink->hidden,situ_beta,situ_linear_beta,error,error_size));
                continue;
            }
            auto & r=sink->requests[selected];const int n=int(std::find(native_ids,native_ids+native_count,r.id)-native_ids);
            const auto compute_start=sink_clock::now();r.events[6]=ggml_time_us();
            const void * data[3];for(int role=0;role<3;++role){data[role]=r.cache_hit>=0?sink->cache.data(r.cache_hit,role):r.read_target[role];}
            if(r.cache_only&&r.cache_hit<0){throw std::runtime_error("extra expert lacks a cache lease");}
            const bool upload_early=locations.version&&locations.admit&&!r.cache_only&&sink->vram.async_upload&&
                (r.cache_hit>=0||r.cache_admit>=0);
            if(upload_early){checked(locations.admit(location_context,&candidates[n],data[0],data[1],data[2],1,error,error_size));}
            sink->up_data=static_cast<const char *>(data[0]);sink->gate_data=static_cast<const char *>(data[1]);sink->down_data=static_cast<const char *>(data[2]);
            sink->output=sink->expert_output.data()+size_t(selected)*sink->latent;sink->weight=1.0f;
            phase=sink_clock::now();sink->compute_phase(false);sink->current.up_gate_us+=elapsed_us(phase);
            phase=sink_clock::now();sink->quantize_hidden(sink->activation.data(),sink->quant_hidden.data(),sink->hidden);sink->current.hidden_quantize_us+=elapsed_us(phase);
            phase=sink_clock::now();sink->compute_phase(true);sink->current.down_reduce_us+=elapsed_us(phase);
            r.events[7]=ggml_time_us();sink->current.compute_us+=elapsed_us(compute_start);a.computed_mask|=1u<<n;
            if(locations.version&&locations.admit&&!r.cache_only&&!upload_early) {
                checked(locations.admit(location_context,&candidates[n],data[0],data[1],data[2],
                    r.cache_hit>=0||r.cache_admit>=0,error,error_size));
            }
            if(!r.cache_only&&r.cache_hit<0) {
                std::lock_guard<std::mutex> lock(sink->io_mutex);auto & slot=sink->slots[r.io_slot];
                if(slot.request!=size_t(selected)||slot.state!=sink_slot::ready){throw std::runtime_error("computed expert lost its ring lease");}
                sink->cache.admit(r,sink->current);slot.state=sink_slot::empty;a.released_mask|=1u<<n;sink->io_cv.notify_all();
            }
        }
        if(provider_begun) {
            checked(locations.finish(location_context,error,error_size));
            const uint64_t gpu_end=ggml_time_us();
            for(int n=0;n<native_count;++n){if(a.plan.final_mask&gpu_mask&(1u<<n)) {
                a.events[n][6]=gpu_begin_us;a.events[n][7]=gpu_end;a.computed_mask|=1u<<n;
            }}
        }
        if((a.computed_mask&a.plan.final_mask)!=a.plan.final_mask){throw std::runtime_error("final cache-compute set incomplete");}
        // S order first; new native slots afterwards in their original order.
        std::vector<int> order;
        for(int i=0;i<a.plan.original_count;++i){const int n=a.plan.original_order[i];if(a.plan.final_mask&(1u<<n)){order.push_back(n);}}
        for(int n=0;n<native_count;++n){if((a.plan.final_mask&~a.plan.original_mask)&(1u<<n)){order.push_back(n);}}
        float sum=0.0f;for(int n:order){sum+=native_weights[n];}
        if(!std::isfinite(sum)||sum<=0){throw std::runtime_error("invalid final native weight sum");}
        const bool same=a.plan.final_mask==a.plan.original_mask;
        const auto reduction=sink_clock::now();
        for(int n:order) {
            const float weight=same?a.plan.original_weights[n]:(native_weights[n]/sum)*routed_scale;
            a.final_weights[n]=weight;const int j=request_for_native[n];
            const float * raw=(gpu_mask&(1u<<n))?gpu_output.data()+size_t(n)*sink->latent:
                sink->expert_output.data()+size_t(j)*sink->latent;
            for(int row=0;row<sink->latent;++row){volatile float scaled=raw[row]*weight;output[row]+=scaled;}
        }
        sink->current.experts=order.size();sink->current.ordered_reduce_us=elapsed_us(reduction);sink->current.compute_us+=sink->current.ordered_reduce_us;
        sink->vram.release_ram_batch(sink->cache);
        if(provider_acquired) {release_leases(gpu_mask);}
        if(provider_begun&&locations.finished){checked(locations.finished(location_context,sink,error,error_size));}
        collect();sink->current.total_us=elapsed_us(start);sink->current.read_completions=sink->ring.completed()-completed_before;sink->current.pending_reads=sink->ring.pending();
        sink->finish_observation();sink->cache_compute_active=false;
        if(stats){*stats=sink->current;}if(decision){*decision=a;}
        cache_compute_print(a,sink->cache.epoch,sink->cache.token,layer,sink->current,true,gpu_mask);return true;
    } catch(const std::exception & e) {
        sink->cancel_and_wait();
        if(provider_acquired) {
            char cancel_error[512]={};
            if(locations.cancel(location_context,cancel_error,sizeof(cancel_error))) {
                std::fprintf(stderr,"cache location provider cannot drain: %s\n",cancel_error);std::abort();
            }
            // Release only after the provider guarantees kernel/DMA ownership ended.
            try {release_leases((1u<<native_count)-1);} catch (...) {std::abort();}
        }
        sink->vram.ram_leases.clear(); // payload owner invalidates only after all work drained
        collect();sink->cache.release();
        sink->current.total_us=elapsed_us(start);sink->current.read_completions=sink->ring.completed()-completed_before;sink->current.pending_reads=sink->ring.pending();
        sink->finish_observation();sink->cache_compute_active=false;
        if(stats){*stats=sink->current;}if(decision){*decision=a;}
        cache_compute_print(a,sink->cache.epoch,sink->cache.token,layer,sink->current,false,gpu_mask);sink_error(error,error_size,e.what());return false;
    }
}
