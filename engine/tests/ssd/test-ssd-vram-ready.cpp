#include "vram-slot-table.h"
#include <cstdio>
#include <functional>
#include <vector>

static int checks = 0;
static void require(bool ok, const char * message) { ++checks; if (!ok) throw std::runtime_error(message); }
static void rejected(const std::function<void()> & fn) {
    bool failed = false; try { fn(); } catch (const std::exception &) { failed = true; }
    require(failed, "unsafe lifecycle operation accepted");
}
int main() { try {
    vram_slot_table table(2, {1}, false);
    kimi_vram_request r = {}; r.layer = 1; r.expert = 0; r.weight = .3f;
    for (int i = 0; i < 3; ++i) r.role[i] = {1,2,1000000,uint64_t(i)*4096,4096,3,4,5,6,0};
    table.next_token(); table.prepare(&r,1); int ready = -1;
    rejected([&] { table.acquire_ready(&r,1,&ready); });
    table.publish(0); table.finish();
    auto generation = table.acquire_ready(&r,1,&ready);
    require(ready == 1 && table.slots[0].locked, "ready copy not pinned");
    rejected([&] { table.next_token(); });
    rejected([&] { table.prepare(&r,1); });
    rejected([&] { table.clear(); });
    rejected([&] { table.release_ready(generation+1); });
    table.validate_ready(generation); table.release_ready(generation);
    rejected([&] { table.validate_ready(generation); });
    r.weight = .9f;
    auto next = table.acquire_ready(&r,1,&ready);
    require(next > generation && ready == 1, "routing weight changed payload identity");
    table.release_ready(next);
    std::vector<kimi_vram_request> union_requests(17, r);
    for (int expert = 0; expert < 17; ++expert) { union_requests[size_t(expert)].expert = expert; }
    std::vector<int> slots(union_requests.size(), -2);
    next = table.acquire_slots(union_requests.data(), union_requests.size(), slots.data());
    require(slots[0] == 0 && std::count(slots.begin(), slots.end(), -1) == 16 && table.slots[0].locked,
            "grouped-prefill union did not return only its resident physical slot");
    table.release_ready(next);
    r.role[1].ctime_ns++;
    next = table.acquire_ready(&r,1,&ready);
    require(ready == 0, "stale packed-file contents accepted"); table.release_ready(next);
    table.clear();
    next = table.acquire_ready(&r,1,&ready);
    require(next > generation && ready == 0, "clear recycled a confirmation");
    table.abort(); rejected([&] { table.validate_ready(next); });
    std::printf("VRAM_READY_LEASE_PASS assertions=%d\n", checks); return 0;
} catch (const std::exception & e) { std::fprintf(stderr,"READY_LEASE_FAIL: %s\n",e.what()); return 1; } }
