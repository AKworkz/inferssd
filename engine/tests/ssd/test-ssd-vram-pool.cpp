#include "vram-slot-table.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>

static int checks = 0;
static void require(bool ok, const char * message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
static void rejected(const std::function<void()> & fn) {
    bool failed = false;
    try { fn(); } catch (const std::exception &) { failed = true; }
    require(failed, "invalid pool configuration or lease operation accepted");
}
static kimi_vram_request request(int layer, int expert) {
    kimi_vram_request r = {};
    r.layer=layer;r.expert=expert;r.weight=1;
    for (int i=0;i<3;++i) r.role[i]={1,7,1000000,uint64_t(expert*3+i)*4096,4096,3,4,5,6,0};
    return r;
}
static void publish(vram_slot_table & table) {
    for (size_t i=0;i<table.admissions.size();++i) if (table.admissions[i]>=0) table.publish(i);
    table.finish();
}
// holds() against a scan of the slots: an expert is held when a slot is valid with it or reserved for its deferred
// upload (a seed's copy counts once published).
static void holds_match(const vram_slot_table & t,const char * what) {
    for (size_t layer=0;layer<t.layers.size();++layer) for (int expert=0;expert<int(t.layers[layer].held.size());++expert) {
        const bool scan=std::any_of(t.slots.begin(),t.slots.end(),[&](const vram_slot_table::entry & e){
            return (e.valid || (e.uploading && !e.seed)) && e.key.layer==int(layer) && e.key.expert==expert;});
        require(t.holds(int(layer),expert)==scan,what);
    }
}
int main() { try {
    rejected([]{vram_slot_table t(2,{1,2},true,0,true,0);});
    rejected([]{vram_slot_table t(2,{1,2},true,0,true,std::numeric_limits<double>::infinity());});
    rejected([]{vram_slot_table t(2,{1,2},false,0,true,4);});
    rejected([]{vram_slot_table t(2,{1,1},true,0,true,4);});

    auto a=request(1,0),b=request(1,1),c=request(2,2);
    kimi_vram_request pair[]={a,b};
    vram_slot_table old(2,{1,2},false),pool(2,{1,2},false,0,true);
    for (auto * t:{&old,&pool}) {
        t->next_token();t->prepare(pair,2);publish(*t);
        t->next_token();t->prepare(pair,2);
    }
    require(old.hits[0]>=0 && old.hits[1]<0,"legacy layer quota changed");
    require(pool.hits[0]>=0 && pool.hits[1]>=0,"global pool cannot use an idle layer slot");
    publish(old);publish(pool);

    // A later selected hit must be protected before an earlier miss chooses
    // a victim, including when the arena is shared with other layers.
    auto d=request(1,3);kimi_vram_request reordered[]={d,a};
    pool.next_token();pool.prepare(reordered,2);
    require(pool.hits[1]>=0 && pool.admissions[0]>=0,"selected hit was evicted by an earlier miss");
    publish(pool);

    pool.next_token();const int pinned=pool.hits[1];pool.slots[pinned].native_refs=1;
    pool.prepare(&c,1);
    require(pool.admissions[0]>=0 && pool.admissions[0]!=pinned,"cross-layer admission evicted native lease");
    publish(pool);rejected([&]{pool.clear();});
    pool.slots[pinned].native_refs=0;
    int ready[2]={};kimi_vram_request lookup[]={a,c};
    const auto generation=pool.acquire_slots(lookup,2,ready);
    require(ready[0]>=0 && ready[1]>=0,"global lookup lost a layer identity");
    rejected([&]{pool.prepare(&b,1);});pool.release_ready(generation);
    lookup[0].layer=2;lookup[1].role[0].ctime_ns++;
    const auto stale=pool.acquire_slots(lookup,2,ready);
    require(ready[0]<0 && ready[1]<0,"global lookup accepted wrong layer or stale payload");
    pool.release_ready(stale);

    // Global logical slots grow and shrink at block boundaries without
    // changing layer heat. Live leases and uploading tail entries block shrink.
    const double retained_heat=pool.layers[1].counts[0];
    pool.reserve_global(6);pool.resize_global(6);
    require(pool.slots.size()==6 && pool.layers[1].count==6 && pool.layers[2].count==6,
            "global grow did not publish every slot to every layer");
    require(pool.layers[1].counts[0]==retained_heat,"global grow reset selection heat");
    const auto resize_lease=pool.acquire_slots(&c,1,ready);
    rejected([&]{pool.resize_global(2);});pool.release_ready(resize_lease);
    pool.slots[5].uploading=true;rejected([&]{pool.resize_global(5);});pool.slots[5].uploading=false;
    pool.resize_global(2);
    require(pool.slots.size()==2 && pool.layers[1].count==2 && pool.layers[2].count==2,
            "global shrink left stale layer bounds");
    require(pool.layers[1].counts[0]==retained_heat,"global shrink reset selection heat");
    rejected([&]{old.resize_global(3);});

    vram_slot_table decay(1,{1,2},true,0,true,2);
    for(int i=0;i<4;++i){decay.next_token();decay.prepare(&a,1);publish(decay);}
    const double hot=decay.layers[1].counts[0];
    decay.next_token();decay.prepare(&c,1);
    require(decay.admissions[0]<0,"one cold observation displaced a hotter layer");
    require(decay.layers[1].counts[0]==hot,"victim scoring trained another layer");
    publish(decay);
    // The score cached while another layer was prepared must be invalidated
    // when this layer's history is brought up to date in the same token.
    decay.prepare(&a,1);
    require(decay.score(decay.slots[0])==decay.layers[1].counts[0],"cached decay applied twice in one token");
    publish(decay);
    for(int i=0;i<6;++i)decay.next_token();
    decay.prepare(&c,1);
    require(decay.admissions[0]==0,"unvisited layer did not age across tokens");publish(decay);
    decay.abort();require(!decay.slots[0].valid && decay.layers[1].counts[0]==0,"abort retained cache payload or history");

    vram_slot_table slow(1,{1},true,2.5,false,4),legacy(1,{1},true,2.5);
    for(int i=0;i<3;++i){
        for(auto * t:{&slow,&legacy}){t->next_token();t->prepare(&a,1);}
        if(i==2)require(slow.admissions[0]>=0 && legacy.admissions[0]<0,"half-life did not affect admission only");
        publish(slow);publish(legacy);
    }
    // Prefill seeds take only empty slots and never evict. Their selection count is zero, so the unused ones are
    // decode's first victims; the ones decode never selected are freed before the next prefill. A seed decode
    // selects becomes an ordinary entry.
    {
        vram_slot_table t(3,{1},true,0,true,16);
        t.next_token();
        const auto r5=request(1,5),r7=request(1,7),r8=request(1,8),r9=request(1,9);
        t.prepare(&r5,1);require(t.admissions[0]>=0,"decode did not admit into an empty slot");publish(t);
        const int s7=t.reserve_cold_prefill(r7),s8=t.reserve_cold_prefill(r8);
        require(s7>=0 && s8>=0 && t.free_slots(1)==0 && t.reserve_cold_prefill(r9)<0 && t.reserve_cold_prefill(r5)<0,
                "a seed took a slot that was not empty");
        t.publish_cold_prefill(size_t(s7),r7);t.publish_cold_prefill(size_t(s8),r8);
        require(t.slots[size_t(s7)].seed && t.layers[1].counts[7]==0 && t.layers[1].counts[8]==0,"a seed changed decode's counts");
        t.next_token();t.prepare(&r7,1);require(t.hits[0]==s7,"decode missed a published seed");publish(t);
        require(!t.slots[size_t(s7)].seed,"a seed decode selected did not become an ordinary entry");
        t.next_token();t.prepare(&r7,1);require(t.hits[0]==s7,"decode missed the former seed");publish(t);
        t.next_token();t.prepare(&r9,1);
        require(t.admissions[0]==s8,"decode did not evict the unused seed first");publish(t);
        require(!t.slots[size_t(s8)].seed,"the slot of an evicted seed is still a seed");
        t.next_token();
        const int s10=t.reserve_cold_prefill(request(1,10));
        require(s10<0,"a seed evicted a decode entry");
        vram_slot_table u(2,{1},true,0,true,16);
        const int a=u.reserve_cold_prefill(r7),b=u.reserve_cold_prefill(r8);
        u.publish_cold_prefill(size_t(a),r7);u.publish_cold_prefill(size_t(b),r8);
        u.next_token();u.prepare(&r7,1);publish(u);
        require(u.free_unused_seeds()==1 && !u.slots[size_t(b)].valid && u.slots[size_t(a)].valid,
                "only the seed decode never selected is freed");
    }
    // Preloads (filled at load, no request having named their experts): held like any entry. A prefill's pass makes
    // the ones it routes in a layer seeds of its prompt (adopt_preloaded); its seeds take the others of the layers it
    // has reached as if empty, never those of a layer ahead of it. A decode miss takes one as if empty, at the price
    // of an empty slot; decode's first selection of one makes it an ordinary entry.
    {
        vram_slot_table t(4,{1,2},true,0,true,16);
        const auto p1=request(1,1),p2=request(1,2),p3=request(2,3),p4=request(2,4);
        for (const auto & r : {p1,p2,p3,p4}) {
            const int s=t.reserve_preload(r);require(s>=0,"no empty slot for a preload");t.publish_cold_prefill(size_t(s),r);
        }
        require(t.empty_slots()==0 && t.holds(1,1) && t.holds(2,4) && t.slots[0].preloaded && !t.slots[0].seed,
                "preloads are not held, preloaded entries");
        require(t.reserve_preload(request(1,5))<0,"a preload took a preloaded slot");
        holds_match(t,"held differs from the slots after preloads");
        t.adopt_preloaded(&p1,1);
        const auto valid=[](const vram_slot_table::entry & e){return e.valid;};
        const int s1=t.find(p1,valid);
        require(s1>=0 && t.slots[size_t(s1)].seed && !t.slots[size_t(s1)].preloaded,"a routed preload did not become a seed");
        require(t.free_slots(1)==1 && t.free_slots(2)==3,"the slots a seed may take do not follow the pass's layers");
        const auto r9=request(1,9);
        const int s9=t.reserve_cold_prefill(r9);
        require(s9>=0 && !t.holds(1,2) && t.holds(2,3) && t.holds(2,4) && t.reserve_cold_prefill(request(1,10))<0,
                "a seed took a slot of a layer ahead of its pass, or not the unused preload of its own");
        t.publish_cold_prefill(size_t(s9),r9);
        t.candidate_saving=t.ram_saving=100;t.drive_saving=1000;
        t.next_token();
        const auto r6=request(2,6);t.prepare(&r6,1);
        require(t.admissions[0]>=0 && t.slots[size_t(t.admissions[0])].key.expert==6 && (t.holds(2,3)!=t.holds(2,4)),
                "a decode miss did not take a preloaded slot as if empty");
        publish(t);
        const auto left=t.holds(2,3)?p3:p4;
        t.next_token();t.prepare(&left,1);
        require(t.hits[0]>=0 && !t.slots[size_t(t.hits[0])].preloaded && !t.slots[size_t(t.hits[0])].seed,
                "decode's first selection of a preload did not make it an ordinary entry");
        publish(t);
        holds_match(t,"held differs from the slots after preloads were used");
    }
    // What the tier holds, for the next-layer prefetch: admissions once published, not their victims; deferred
    // uploads from their reservation, until a block given back drops them; seeds from their publication, not while
    // copied or once cancelled or freed unused; a shrink of the global pool; native cancel; abort.
    {
        vram_slot_table t(3,{1,2},true,0,true,16);
        const auto r1=request(1,1),r2=request(1,2),r3=request(2,3),r4=request(2,4),r5=request(1,5);
        t.next_token();t.prepare(&r1,1);
        require(t.admissions[0]>=0 && !t.holds(1,1),"a reservation counted as held before its upload");
        holds_match(t,"held differs from the slots after a reservation");
        publish(t);require(t.holds(1,1) && !t.holds(2,1),"a published admission is not held, or under another layer");
        holds_match(t,"held differs from the slots after a publication");
        t.next_token();t.prepare(&r3,1);const int deferred=t.admissions[0];t.reserve_upload(0);t.finish();
        require(t.holds(2,3),"a deferred upload in flight is not held");holds_match(t,"held differs during a deferred upload");
        t.publish_upload(size_t(deferred),r3);require(t.holds(2,3),"a published deferred upload is not held");
        t.next_token();t.prepare(&r2,1);t.reserve_upload(0);t.finish();
        const int slot2=std::find_if(t.slots.begin(),t.slots.end(),[&](const vram_slot_table::entry & e){return e.uploading;})-t.slots.begin();
        t.cancel_upload(size_t(slot2),r2);
        require(!t.holds(1,2) && t.holds(1,1) && t.holds(2,3),"a deferred upload of a block given back is still held");
        holds_match(t,"held differs after a dropped upload");
        t.next_token();t.prepare(&r2,1);publish(t);require(t.holds(1,2),"the third admission is not held");
        // full: the next admission evicts the coldest entry, which is no longer held
        t.next_token();t.prepare(&r5,1);
        require(t.admissions[0]>=0 && !t.holds(1,5),"no eviction for a new expert in a full pool");
        holds_match(t,"held differs after an eviction");publish(t);holds_match(t,"held differs after an eviction's publication");
        require(t.holds(1,5) && int(t.holds(1,1))+int(t.holds(1,2))+int(t.holds(2,3))==2,"the admission or its victim");
        // seeds: held from their publication, a cancelled one never, an unused published one freed
        vram_slot_table s(4,{1,2},true,0,true,16);
        const int seed1=s.reserve_cold_prefill(r1),seed4=s.reserve_cold_prefill(r4);
        require(seed1>=0 && seed4>=0 && !s.holds(1,1) && !s.holds(2,4),"a seed being copied is held before its publication");
        holds_match(s,"held differs while seeds are copied");
        s.cancel_cold_prefill(size_t(seed4),r4);require(!s.holds(2,4),"a cancelled seed is held");
        s.publish_cold_prefill(size_t(seed1),r1);require(s.holds(1,1),"a published seed is not held");
        holds_match(s,"held differs after a seed's publication");
        require(s.free_unused_seeds()==1 && !s.holds(1,1),"a freed seed is still held");
        holds_match(s,"held differs after the seeds");
        // reclaim: a shrink of the global pool drops what its tail slots held
        vram_slot_table g(2,{1,2},true,0,true,16);g.reserve_global(4);g.resize_global(4);
        const kimi_vram_request four[]={r1,r2,r5,request(1,6)};
        g.next_token();g.prepare(four,4);publish(g);
        require(g.holds(1,1) && g.holds(1,2) && g.holds(1,5) && g.holds(1,6),"four admissions are not held");
        g.resize_global(2);holds_match(g,"held differs after a shrink");
        require(int(g.holds(1,1))+int(g.holds(1,2))+int(g.holds(1,5))+int(g.holds(1,6))==2,"a shrink kept what it dropped");
        g.invalidate();holds_match(g,"held differs after a native cancel");require(!g.holds(1,1) && !g.holds(1,2),"cancelled payload held");
        t.abort();require(!t.holds(1,5) && !t.holds(2,3),"abort kept what the tier held");holds_match(t,"held differs after abort");
        require(!t.holds(0,1) && !t.holds(7,1) && !t.holds(1,-1) && !t.holds(1,999),"held answered outside the table");
    }
    // Decode's replacements, priced by what a use of an expert saves when the tier serves it: a candidate's
    // (candidate_saving), and a victim's against its RAM copy where the RAM cache keeps one (ram_saving), else against
    // the drive (drive_saving). Before the prices are measured only the fixed minimum count takes a free slot; then
    // any expert does, and a full table replaces its least used entry when count x candidate_saving exceeds that
    // entry's count x its saving. Where the tier saves a candidate nothing it admits nothing.
    {
        const auto a=request(1,1),b=request(1,2);
        const auto price=[](vram_slot_table & t,double candidate,double ram,double drive){
            t.candidate_saving=candidate;t.ram_saving=ram;t.drive_saving=drive;};
        vram_slot_table t(1,{1},true,2.5,true,16);
        t.next_token();t.prepare(&a,1);require(t.admissions[0]<0,"an unpriced table admitted below the minimum count");publish(t);
        price(t,100,100,1000);
        t.next_token();t.prepare(&a,1);require(t.admissions[0]>=0,"a priced table refused a free slot");publish(t);
        t.next_token();t.prepare(&b,1);
        require(t.admissions[0]<0,"one selection replaced an entry whose uses would come from the drive");publish(t);
        t.layers[1].counts[2]=20; // as if b had been selected in most of the recent tokens
        t.next_token();t.prepare(&b,1);
        require(t.admissions[0]>=0 && t.evictions==1,"a candidate worth more than its victim's loss was refused");publish(t);
        // a victim the RAM cache keeps a copy of costs only its uses' RAM hits (here a tenth of a drive read's): b at
        // four times a's count replaces a where the RAM cache keeps a copy of a, not where it does not
        for (const bool copy : {true,false}) {
            vram_slot_table u(1,{1},true,2.5,true,16);price(u,100,100,1000);
            if(copy)u.ram_holds=[](void *,int32_t layer,int32_t expert){return int(layer==1 && expert==1);};
            u.next_token();u.prepare(&a,1);publish(u);
            u.layers[1].counts[2]=3;
            u.next_token();u.prepare(&b,1);
            require((u.admissions[0]>=0)==copy,"a victim's RAM copy did not set what replacing it gives up");publish(u);
        }
        vram_slot_table none(1,{1},true,0,true,16);price(none,0,0,1000);
        none.next_token();none.prepare(&a,1);require(none.admissions[0]<0,"a tier that saves nothing on a candidate admitted");publish(none);
    }
    // A shrink gives up the least valuable entries (compaction): the most valuable of the leaving slots' entries move
    // into empty slots below the boundary first, then onto entries worth less; live slots neither move nor take a move.
    {
        vram_slot_table t(4,{1},true,0,true,16);
        const kimi_vram_request four[]={request(1,1),request(1,2),request(1,3),request(1,4)};
        t.next_token();t.prepare(four,4);publish(t);
        const int head_cold=t.slots[0].key.expert,head_warm=t.slots[1].key.expert;
        const int tail_hot=t.slots[2].key.expert,tail_cold=t.slots[3].key.expert;
        auto & counts=t.layers[1].counts;
        counts[size_t(head_cold)]=1;counts[size_t(head_warm)]=3;counts[size_t(tail_hot)]=5;counts[size_t(tail_cold)]=.5;
        const auto moves=t.compaction(2);
        require(moves.size()==1 && moves[0].first==2 && moves[0].second==0,
                "compaction did not put the hottest leaving entry on the coldest kept one, and stop there");
        const auto evicted=t.evictions;
        t.move_entry(2,0);t.resize_global(2);
        require(t.holds(1,tail_hot) && t.holds(1,head_warm) && !t.holds(1,head_cold) && !t.holds(1,tail_cold) &&
                t.evictions==evicted+1,"a shrink did not give up the coldest entries");
        require(t.find(request(1,tail_hot),[](const vram_slot_table::entry & e){return e.valid;})==0,
                "the moved entry is not found in its new slot");
        holds_match(t,"held differs after a compaction");
        vram_slot_table u(4,{1},true,0,true,16);
        u.next_token();u.prepare(four,4);publish(u);
        u.change(u.slots[1],[](vram_slot_table::entry & x){x={};});
        const auto into_empty=u.compaction(2);
        require(into_empty.size()==1 && into_empty[0].second==1,"a leaving entry did not take the empty slot below first");
        u.slots[2].uploading=true;u.slots[3].uploading=true;
        require(u.compaction(2).empty(),"a slot being uploaded moved");
        u.slots[2].uploading=u.slots[3].uploading=false;u.slots[1].locked=true;u.slots[0].native_refs=1;
        require(u.compaction(2).empty(),"a live slot took a move");
        u.slots[1].locked=false;u.slots[0].native_refs=0;
        rejected([&]{u.move_entry(1,0);});
    }
    std::printf("VRAM_GLOBAL_POOL_PASS assertions=%d\n",checks);return 0;
} catch(const std::exception & e) { std::fprintf(stderr,"VRAM_GLOBAL_POOL_FAIL: %s\n",e.what());return 1; } }
