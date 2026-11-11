#include "vram-slot-table.h"
#include <cstdio>

int main(){try{
    vram_slot_table table(2,{1},true,0);kimi_vram_request r[4]={};
    for(int i=0;i<4;++i){r[i].layer=1;r[i].expert=i;r[i].weight=1;for(auto & role:r[i].role){role.inode=7;role.bytes=4096;role.offset=i*4096;}}
    table.next_token();table.prepare(r,2);table.publish(0);table.publish(1);table.finish();
    table.next_token();table.slots[0].native_refs=1;
    const auto counts=table.layers[1].counts;
    // Query/pin does not alter heat. Only expert 2 is true demand; 0 and 1
    // are cached candidates and 3 is an unselected native candidate.
    table.prepare(r,4,1u<<2);
    if(table.layers[1].counts[0]!=counts[0]*std::exp2(-.5)||table.layers[1].counts[3]!=0||table.layers[1].counts[2]!=1)
        throw std::runtime_error("cache-only query trained native heat");
    if(table.admissions[2]!=1||!table.slots[0].valid||table.slots[0].key.expert!=0)
        throw std::runtime_error("admission evicted a native compute lease");
    table.publish(2);table.finish();
    int rejected=0;
    try{table.next_token();}catch(const std::exception &){++rejected;}
    try{table.clear();}catch(const std::exception &){++rejected;}
    int ready[4]={};try{table.acquire_ready(r,4,ready);}catch(const std::exception &){++rejected;}
    if(rejected!=3)throw std::runtime_error("native lease did not protect lifecycle boundaries");
    table.slots[0].native_refs=0;const auto lease=table.acquire_ready(r,4,ready);
    if(ready[0]!=1||ready[1]!=0||ready[2]!=1||ready[3]!=0)throw std::runtime_error("wrong post-finish residency");
    table.release_ready(lease);table.next_token();table.clear();
    std::puts("VRAM_NATIVE_METADATA_PASS checks=7");return 0;
}catch(const std::exception & e){std::fprintf(stderr,"VRAM_NATIVE_METADATA_FAIL: %s\n",e.what());return 1;}}
