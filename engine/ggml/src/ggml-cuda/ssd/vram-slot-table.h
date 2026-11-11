#pragma once
#include "ggml-ssd/gpu/vram-tier-api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Deterministic cache decisions, independent of I/O completion order. The
// transport owns bytes; this table publishes a slot only after copy completion.
class vram_slot_table {
public:
    struct entry {
        kimi_vram_request key = {};
        uint64_t touched = 0;
        uint32_t native_refs = 0;
        // uploading: reserved by a deferred upload that outlives its layer; never a victim until published
        bool valid = false, locked = false, uploading = false;
        // seed: put into an empty slot by a prefill (reserve_cold_prefill), not admitted by decode. Its selection
        // count is zero until decode requests it; the first decode request makes it an ordinary entry.
        bool seed = false;
        // keyed: the key names an expert (an admission or a seed set it, and no reset since); see find
        bool keyed = false;
        // preloaded: filled at load (reserve_preload), with an expert no request had named, and not used since. A
        // lookup finds it like any entry; every allocation takes it as if it were empty (available), except a
        // prefill's for a layer its pass has not reached. Its first use makes it what that use makes of an empty
        // slot: a prefill's, a seed of its prompt (adopt_preloaded); decode's, an entry admitted at that request.
        bool preloaded = false;
    };
    struct layer {
        size_t begin = 0, count = 0;
        uint64_t observed = 0;
        std::vector<double> counts; // one entry per routed expert of the layer
        std::vector<uint16_t> held; // per routed expert: the slots that hold it (see holds)
        std::vector<std::vector<uint32_t>> keyed; // per routed expert: the keyed slots with its key, ascending
        mutable uint64_t score_step = 0;
        mutable double score_factor = 1;
    };
    std::vector<entry> slots;
    std::vector<layer> layers;
    std::vector<int> pool_layers;
    std::vector<int> hits, admissions;
    std::vector<kimi_vram_request> requests;
    bool count_decay = true;
    double minimum_count = 0;
    bool global_pool = false;
    double half_life = 2;
    // Kimi-K3 defaults: layers 1..92 with 896 experts. The explicit-format tier
    // passes the model's layer range and expert count.
    int first_layer = 1, last_layer = 92, experts = 896;
    bool in_flight = false;
    uint64_t step = 0, clock = 0, evictions = 0, admitted = 0;
    // What a use of an expert saves when the tier serves it, in microseconds, against one from the RAM cache
    // (ram_saving) and one from the drive (drive_saving); the tier measures both. They price decode's replacements.
    // A candidate is an expert in use: its uses outside the tier come from the RAM cache where the sink has one, and
    // each of them the tier serves saves candidate_saving (below 0 until measured, when the fixed minimum_count rule
    // applies). Replacing an entry gives up what its uses save: against the RAM cache where it keeps a copy of the
    // expert (ram_copy: the RAM cache does not drop its copy of an expert the tier takes, and drops such a copy first
    // when it needs the slot), else against the drive. A miss replaces its victim when its count x candidate_saving
    // exceeds the victim's count x that saving. Both counts cover the same memory, so the rule needs no horizon. A
    // free slot takes any expert the tier saves time on: a deferred upload is off the decode's critical path.
    double candidate_saving = -1, ram_saving = -1, drive_saving = -1;
    bool priced() const { return count_decay && candidate_saving >= 0; }
    // The sink's RAM cache: whether it keeps a copy of an expert (none without one).
    int (*ram_holds)(void *, int32_t layer, int32_t expert) = nullptr;
    void * ram_holds_arg = nullptr;
    bool ram_copy(const kimi_vram_request & key) const { return ram_holds && ram_holds(ram_holds_arg, key.layer, key.expert); }
    uint64_t lease_generation = 0, next_generation = 0;

    explicit vram_slot_table(size_t capacity, const std::vector<int> & priority, bool decay, double minimum=0,
            bool global=false, double half=2, int first=1, int last=92, int experts_per_layer=896) :
            count_decay(decay), minimum_count(minimum), global_pool(global), half_life(half),
            first_layer(first), last_layer(last), experts(experts_per_layer) {
        if (!std::isfinite(minimum) || minimum < 0 || minimum > 4 || (!decay && minimum != 0)) { throw std::runtime_error("VRAM minimum selection count"); }
        if (!std::isfinite(half) || half < 1 || half > 64 || (!decay && half != 2)) { throw std::runtime_error("VRAM decay half-life"); }
        if (first < 0 || last < first || last >= KIMI_VRAM_MAX_LAYERS || experts_per_layer < 1 || experts_per_layer > 65536) { throw std::runtime_error("VRAM layer/expert geometry"); }
        if (priority.empty() || capacity > priority.size()*size_t(experts)) { throw std::runtime_error("VRAM cache capacity"); }
        pool_layers = priority;
        for(int id:priority)if(id<first_layer || id>last_layer)throw std::runtime_error("VRAM layer priority range");
        layers.resize(*std::max_element(priority.begin(),priority.end())+1);
        for(auto & l:layers){
            l.counts.assign(size_t(experts),0.0);l.held.assign(size_t(experts),0);l.keyed.assign(size_t(experts),{});
        }
        std::vector<bool> seen(layers.size());
        size_t assigned=0;
        for(size_t i=0;i<priority.size();++i) {
            int id=priority[i];
            if(id<first_layer || seen[id]) { throw std::runtime_error("VRAM cache layer priority"); }
            seen[id]=true;auto & l=layers[id];l.begin=assigned;
            l.count=capacity/priority.size()+(i<capacity%priority.size());assigned+=l.count;
        }
        slots.resize(capacity);
        // One payload arena, shared by all layers. Source keys retain the
        // layer identity; live compute/readiness leases still protect slots.
        if (global_pool) for (int id : priority) { layers[id].begin=0;layers[id].count=capacity; }
    }
    // Elastic storage is deliberately limited to the global pool. Its logical
    // slot IDs remain stable while tail blocks are added or removed; per-layer
    // heat survives a resize even when every payload slot is reclaimed.
    void reserve_global(size_t capacity) {
        if (!global_pool || capacity > pool_layers.size()*size_t(experts)) {
            throw std::runtime_error("invalid VRAM global reserve");
        }
        slots.reserve(capacity);
    }
    void resize_global(size_t capacity) {
        if (!global_pool || capacity > pool_layers.size()*size_t(experts) ||
                in_flight || lease_generation || native_pinned()) {
            throw std::runtime_error("invalid VRAM global resize phase");
        }
        if (capacity < slots.size() && std::any_of(
                slots.begin() + capacity, slots.end(), [](const entry & e) { return e.uploading || e.locked || e.native_refs; })) {
            throw std::runtime_error("VRAM global shrink with a live tail slot");
        }
        for (size_t i = capacity; i < slots.size(); ++i) change(slots[i], [](entry & x) { x = {}; });
        slots.resize(capacity);
        for (int id : pool_layers) {
            layers[id].begin = 0;
            layers[id].count = capacity;
        }
    }
    static bool same_source(const kimi_vram_source & a,const kimi_vram_source & b) {
        return a.device==b.device && a.inode==b.inode && a.inode_hi==b.inode_hi && a.size==b.size && a.offset==b.offset &&
            a.bytes==b.bytes && a.mtime_s==b.mtime_s && a.mtime_ns==b.mtime_ns &&
            a.ctime_s==b.ctime_s && a.ctime_ns==b.ctime_ns;
    }
    static bool same(const kimi_vram_request & a,const kimi_vram_request & b) {
        if(a.layer!=b.layer || a.expert!=b.expert)return false;
        for(int r=0;r<3;++r)if(!same_source(a.role[r],b.role[r]))return false;
        return true; // Routing weight is not part of immutable payload identity.
    }
    bool native_pinned() const {
        return std::any_of(slots.begin(),slots.end(),[](const entry & e){return e.native_refs!=0;});
    }
    // Whether the tier holds an expert now: a slot is valid with it, or reserved for a deferred upload of it that has
    // not completed (the layer's next prepare collects it first). A seed (reserve_cold_prefill) counts from its
    // publication (publish_cold_prefill), which no prepare does: a prefill's planned seeds are published when the
    // decode phase completes the prefill (phase_complete), Q2's cold-prefill admission right after its own copy. The
    // next-layer prefetch asks before it reads an expert into RAM. (layer, expert) stands for the key: one model's
    // experts have one source each. Every change of a slot's key, valid, uploading or seed goes through change(),
    // which keeps the count and the index of keyed slots (find).
    bool holds(int layer,int expert) const {
        return layer>=0 && size_t(layer)<layers.size() && expert>=0 && size_t(expert)<layers[layer].held.size() &&
            layers[layer].held[size_t(expert)]>0;
    }
    static bool held_by(const entry & e) { return e.valid || (e.uploading && !e.seed); }
    template<class F> void change(entry & e,F && update) {
        const uint32_t slot=uint32_t(&e-slots.data());
        const bool was=e.keyed;const int32_t layer=e.key.layer,expert=e.key.expert;
        if(held_by(e))--layers[e.key.layer].held[size_t(e.key.expert)];
        update(e);
        if(held_by(e))++layers[e.key.layer].held[size_t(e.key.expert)];
        const bool same_expert=layer==e.key.layer && expert==e.key.expert;
        if(was==e.keyed && (!was || same_expert))return; // the index has the slot where it was
        if(was){
            auto & v=layers[size_t(layer)].keyed[size_t(expert)];const auto it=std::find(v.begin(),v.end(),slot);
            if(it==v.end())throw std::runtime_error("VRAM slot index lost a slot");
            v.erase(it);
        }
        if(e.keyed){
            auto & v=layers[size_t(e.key.layer)].keyed[size_t(e.key.expert)];
            v.insert(std::upper_bound(v.begin(),v.end(),slot),slot);
        }
    }
    // The first slot of the request's layer range whose key is the request's and for which take(slot) holds, or -1;
    // the slots it looks at are the keyed ones with the request's expert, in slot order. A slot without a key keeps
    // the default one, which no request's sources match.
    template<class P> int find(const kimi_vram_request & request,P && take) const {
        const auto & l=layers[size_t(request.layer)];
        for(const uint32_t i:l.keyed[size_t(request.expert)]){
            const auto & e=slots[i];
            if(i>=l.begin && i<l.begin+l.count && take(e) && same(e.key,request))return int(i);
        }
        return -1;
    }
    // The native consumer cancelled: no payload is valid any more; heat stays.
    void invalidate() {
        for(auto & e:slots)change(e,[](entry & x){x.valid=false;x.locked=false;});
    }
    void clear() {
        if(in_flight || lease_generation || native_pinned()) { throw std::runtime_error("clear with live VRAM slots"); }
        for(auto & e:slots)e={};
        for(auto & l:layers){
            std::fill(l.counts.begin(),l.counts.end(),0.0);std::fill(l.held.begin(),l.held.end(),0);
            for(auto & v:l.keyed)v.clear();
            l.observed=l.score_step=0;l.score_factor=1;
        }
        hits.clear();admissions.clear();requests.clear();step=clock=0;
    }
    void next_token() { if(in_flight || lease_generation || native_pinned())throw std::runtime_error("token with live VRAM slots");++step; }
    double score(const entry & e) const {
        if (!count_decay) return 0;
        const auto & l=layers[e.key.layer];
        if (!global_pool) return l.counts[e.key.expert];
        // A victim can belong to a layer not visited yet in this token. Bring
        // its score to this token without training its selection history.
        if (l.score_step!=step) {
            l.score_factor=std::exp2(-double(step-l.observed)/half_life);
            l.score_step=step;
        }
        return l.counts[e.key.expert]*l.score_factor;
    }
    void prepare(const kimi_vram_request * input,size_t n,uint32_t demand_mask=UINT32_MAX) {
        if(in_flight || lease_generation || !step || n>16)throw std::runtime_error("invalid VRAM prepare phase");
        if(n)requests.assign(input,input+n);else requests.clear();
        hits.assign(n,-1);admissions.assign(n,-1);
        if(n==0){in_flight=true;return;}
        const int id=requests[0].layer;
        if(id<first_layer || size_t(id)>=layers.size())throw std::runtime_error("VRAM layer out of range");
        auto & l=layers[id];
        for(const auto & r:requests)if(r.layer!=id || r.expert<0 || r.expert>=experts)throw std::runtime_error("VRAM expert out of range");
        if(l.observed!=step) {
            const double factor=std::exp2(-double(step-l.observed)/half_life);
            for(auto & value:l.counts)value*=factor;
            std::vector<bool> seen(static_cast<size_t>(experts), false);
            for(size_t j=0;j<n;++j){const auto & r=requests[j];if((demand_mask&(1u<<j))&&!seen[r.expert]){l.counts[r.expert]+=1;seen[r.expert]=true;}}
            l.observed=step;
            l.score_step=step;l.score_factor=1;
        }
        // Pin all selected hits before choosing any victim, including a hit
        // that appears after a miss in the original router order. A seed's
        // or a preloaded entry's first hit makes it an ordinary entry.
        for(size_t j=0;j<n;++j) {
            if(!(demand_mask&(1u<<j)))continue;
            const int i=find(requests[j],[](const entry & e){return e.valid;});
            if(i<0)continue;
            auto & e=slots[size_t(i)];
            e.locked=true;e.touched=++clock;hits[j]=i;
            e.seed=e.preloaded=false;
        }
        std::vector<size_t> order;
        for(size_t i=0;i<n;++i)if((demand_mask&(1u<<i))&&hits[i]<0)order.push_back(i);
        std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){
            return count_decay && l.counts[requests[a].expert]>l.counts[requests[b].expert];
        });
        for(size_t j:order) {
            const double count=l.counts[requests[j].expert];
            if(count_decay && (priced()?!(candidate_saving>0):count<minimum_count))continue;
            if(find(requests[j],[](const entry & e){return e.locked || e.native_refs || e.uploading;})>=0)continue;
            int victim=-1;
            for(size_t i=l.begin;i<l.begin+l.count;++i) {
                const auto & e=slots[i];if(e.locked || e.native_refs || e.uploading)continue;
                if(!e.valid || e.preloaded){victim=int(i);break;}
                if(victim<0 || score(e)<score(slots[victim]) ||
                   (score(e)==score(slots[victim]) && e.touched<slots[victim].touched))victim=int(i);
            }
            if(victim<0)continue;
            auto & e=slots[victim];
            const bool occupied=e.valid && !e.preloaded;
            if(occupied && count_decay && (priced()?count*candidate_saving<=score(e)*(ram_copy(e.key)?ram_saving:drive_saving):
                                                   count<score(e)))continue;
            change(e,[&](entry & x){evictions+=x.valid;x.valid=false;x.seed=x.preloaded=false;x.locked=true;x.key=requests[j];x.keyed=true;x.touched=++clock;});
            admissions[j]=victim;
        }
        in_flight=true;
    }
    void publish(size_t request_index) {
        if(!in_flight || request_index>=admissions.size() || admissions[request_index]<0)throw std::runtime_error("VRAM publication without reservation");
        auto & e=slots[admissions[request_index]];
        if(!e.locked || e.valid || e.uploading)throw std::runtime_error("duplicate VRAM publication");
        change(e,[](entry & x){x.valid=true;});++admitted;
    }
    // A deferred upload keeps its reservation after finish() unlocks the layer, until publish_upload.
    void reserve_upload(size_t request_index) {
        if(!in_flight || request_index>=admissions.size() || admissions[request_index]<0)throw std::runtime_error("deferred VRAM upload without reservation");
        auto & e=slots[admissions[request_index]];
        if(!e.locked || e.valid || e.uploading)throw std::runtime_error("duplicate deferred VRAM upload");
        change(e,[](entry & x){x.uploading=true;});
    }
    void publish_upload(size_t slot,const kimi_vram_request & key) {
        if(in_flight || lease_generation || slot>=slots.size())throw std::runtime_error("invalid deferred VRAM publication phase");
        auto & e=slots[slot];
        if(!e.uploading || e.valid || !same(e.key,key))throw std::runtime_error("deferred VRAM publication lost its reservation");
        change(e,[](entry & x){x.uploading=false;x.valid=true;});++admitted;
    }
    // A slot an allocation for `layer` may take: an empty one, or one preloaded with an expert no request has used,
    // of a layer up to `layer`. A prefill reaches the layers in order, and a pass has used what it will of a layer it
    // has reached (adopt_preloaded), so this leaves the experts of the layers ahead of it in their slots.
    static bool available(const entry & e,int layer) {
        return !e.locked && !e.native_refs && !e.uploading && (!e.valid || (e.preloaded && e.key.layer<=layer));
    }
    // A prefill's use of the preloaded entries of the experts its pass routes in a layer (the next prepare's union,
    // before its seeds are planned): each becomes a seed of its prompt, as if the prefill had copied it into an empty
    // slot. The others of the layer stay preloaded, and so available to its seeds.
    void adopt_preloaded(const kimi_vram_request * input,size_t n) {
        for(size_t j=0;j<n;++j) {
            const int i=find(input[j],[](const entry & e){return e.valid && e.preloaded;});
            if(i>=0)change(slots[size_t(i)],[](entry & x){x.preloaded=false;x.seed=true;});
        }
    }
    // Cold prefill fills only available slots. It deliberately does not call
    // prepare(), increment counts, or evict an existing decode entry.
    // A seeded entry therefore has score zero until decode actually requests
    // it, making unused prompt heat the first victim of later decode misses.
    int reserve_cold_prefill(const kimi_vram_request & request) {
        return reserve_unrequested(request,request.layer,true);
    }
    // At load: an empty slot for an expert no request has named yet, its copy to follow (publish_cold_prefill).
    int reserve_preload(const kimi_vram_request & request) {
        return reserve_unrequested(request,first_layer-1,false);
    }
    // Slots no entry holds (all of them in a global pool's layer ranges): what a preload may fill.
    size_t empty_slots() const {
        return size_t(std::count_if(slots.begin(),slots.end(),[](const entry & e){
            return !e.valid && !e.locked && !e.native_refs && !e.uploading;}));
    }
    // Slots a prefill may seed in layer `id` (all of them in a global pool).
    size_t free_slots(int id) const {
        if(id<first_layer || size_t(id)>=layers.size())return 0;
        const auto & l=layers[id];
        return size_t(std::count_if(slots.begin()+l.begin,slots.begin()+l.begin+l.count,[&](const entry & e){
            return available(e,id);}));
    }
    // Seeds the decode after their prefill never selected: prompt heat of a request that is over. Freed before the
    // next prefill seeds, so that its own prompt plans with them. Returns how many.
    size_t free_unused_seeds() {
        if(in_flight || lease_generation || native_pinned())throw std::runtime_error("free seeds with live VRAM slots");
        size_t n=0;
        for(auto & e:slots)if(e.valid && e.seed && !e.locked){change(e,[](entry & x){x={};});++n;}
        return n;
    }
    void publish_cold_prefill(size_t slot,const kimi_vram_request & request) {
        if(in_flight || lease_generation || slot>=slots.size())
            throw std::runtime_error("invalid cold-prefill VRAM publication phase");
        auto & e=slots[slot];
        if(!e.uploading || e.valid || !same(e.key,request))
            throw std::runtime_error("cold-prefill VRAM publication lost its reservation");
        change(e,[](entry & x){x.uploading=false;x.valid=true;});++admitted;
    }
    // A reservation of a slot no request asked for: a seed (seed) or a preload. `layer`: the latest layer whose
    // preloaded entries it may take (available); below first_layer, only an empty slot.
    int reserve_unrequested(const kimi_vram_request & request,int layer,bool seed) {
        if(in_flight || lease_generation || native_pinned() || request.layer<first_layer ||
                size_t(request.layer)>=layers.size() || request.expert<0 || request.expert>=experts) {
            throw std::runtime_error("invalid unrequested VRAM reservation phase");
        }
        if(find(request,[](const entry & e){return e.valid || e.uploading;})>=0)return -1;
        const auto & l=layers[request.layer];
        for(size_t i=l.begin;i<l.begin+l.count;++i) {
            auto & e=slots[i];
            if(!available(e,layer))continue;
            change(e,[&](entry & x){evictions+=x.valid;x={};x.key=request;x.keyed=true;x.touched=++clock;x.uploading=true;x.seed=seed;x.preloaded=!seed;});
            return int(i);
        }
        return -1;
    }
    // Drops a reservation whose copy did not complete: a seed's, or a deferred upload's in a block given back.
    void cancel_upload(size_t slot,const kimi_vram_request & request) {
        if(slot>=slots.size())return;
        auto & e=slots[slot];
        if(e.uploading && !e.valid && same(e.key,request))change(e,[](entry & x){x={};});
    }
    void cancel_cold_prefill(size_t slot,const kimi_vram_request & request) { cancel_upload(slot,request); }
    // Only prepare locks a slot in flight: its hits and its admissions.
    void finish() {
        if(!in_flight)throw std::runtime_error("VRAM finish without prepare");
        for(int s:hits)if(s>=0)slots[size_t(s)].locked=false;
        for(int s:admissions)if(s>=0)slots[size_t(s)].locked=false;
        in_flight=false;
    }
    uint64_t acquire_ready(const kimi_vram_request * input, size_t n, int * ready) {
        if (in_flight || lease_generation || native_pinned() || n > 16 || (n && (!input || !ready)) || next_generation == UINT64_MAX)
            throw std::runtime_error("invalid VRAM ready lease phase");
        std::vector<int> found(n, -1);
        const uint64_t generation = acquire_slots(input, n, found.data());
        for (size_t j = 0; j < n; ++j) { ready[j] = found[j] >= 0; }
        return generation;
    }
    // Read-only grouped-prefill lease. Unlike prepare(), this accepts the full
    // distinct-expert union, does not update heat/admission state, and returns
    // physical slot IDs so a later worker can reuse each resident triplet.
    uint64_t acquire_slots(const kimi_vram_request * input, size_t n, int * found) {
        if (in_flight || lease_generation || native_pinned() || n > size_t(experts) ||
                (n && (!input || !found)) || next_generation == UINT64_MAX) {
            throw std::runtime_error("invalid VRAM grouped-prefill lease phase");
        }
        for (size_t j = 0; j < n; ++j) {
            if (input[j].layer < first_layer || size_t(input[j].layer) >= layers.size() || input[j].expert < 0 || input[j].expert >= experts)
                throw std::runtime_error("invalid VRAM residency key");
            found[j] = find(input[j], [](const entry & e) { return e.valid; });
            if (found[j] >= 0) { slots[size_t(found[j])].locked = true; }
        }
        return lease_generation = ++next_generation;
    }
    void validate_ready(uint64_t generation) const {
        if (!generation || generation != lease_generation || in_flight)
            throw std::runtime_error("stale VRAM ready lease");
    }
    void release_ready(uint64_t generation) {
        validate_ready(generation);
        for (auto & e : slots) e.locked = false;
        lease_generation = 0;
    }
    void abort() {
        if(native_pinned())throw std::runtime_error("abort with native VRAM leases");
        for(auto & e:slots){if(e.locked && !e.valid)e={};e.locked=false;}
        in_flight=false;lease_generation=0;
        clear(); // Error paths never preserve partially validated payload.
    }
};
