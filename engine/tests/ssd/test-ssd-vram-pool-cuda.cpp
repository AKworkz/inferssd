#include "gpu-services.h"
#include "explicit-fixture-inputs.h"
#define main original_async_test_main
#include "test-ssd-vram-async.cpp"
#undef main

int main(){
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services());try{
    inputs.set("vram_async", "1");
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");
    inputs.set("tier.global", "1");
    // The retired environment zero sentinel resolved to this effective default.
    inputs.set("tier.half_life", "2");
    const size_t role=lifecycle_role_bytes();
    std::vector<unsigned char> payload[4];
    kimi_vram_request requests[4]={};
    for(int j=0;j<4;++j){
        payload[j].resize(role,0);payload[j].back()=static_cast<unsigned char>(31+j);
        requests[j].layer=j<2?1:2;requests[j].expert=j%2;requests[j].weight=.7f;
        for(int i=0;i<3;++i)requests[j].role[i]={1,2,1000000,uint64_t(3*j+i)*role,role,3,4,5,6};
    }
    tier t(2*3*role, inputs.tier_policy());int slots[2]={};std::vector<float> out(512);
    std::vector<float> input(256,0.0f);std::vector<unsigned char> copied(3*role);
    auto upload=[&](int j){const void * data[]={payload[j].data(),payload[j].data(),payload[j].data()};t.admit(requests[j],data,true);};
    auto bytes_equal=[&](int slot,int j){
        cuda_check(cudaMemcpy(copied.data(),t.arena+size_t(slot)*t.stride,copied.size(),cudaMemcpyDeviceToHost));
        for(int i=0;i<3;++i)require(!std::memcmp(copied.data()+i*role,payload[j].data(),role),"cross-layer slot payload mismatch");
    };
    t.begin(17,1,true);lifecycle_prepare(t,requests,2,slots);
    require(t.table->global_pool && t.table->slots.size()==2 && t.table->half_life==2,"pool options or budget not applied");
    upload(0);upload(1);t.finish(out.data(),256);
    t.begin(17,2,true);lifecycle_prepare(t,requests,1,slots);
    require(slots[0]>=0,"warm layer1 copy missing");const int a_slot=slots[0];
    // CUDA loads a kernel's module at its first launch, which waits for the device: one launch before a call is held.
    lifecycle_launch(t,input);t.finish(out.data(),256);
    lifecycle_prepare(t,requests+2,1,slots);
    const int b_slot=t.table->admissions[0];
    require(b_slot>=0 && b_slot!=a_slot,"cross-layer LRU chose wrong victim");upload(2);t.finish(out.data(),256);
    bytes_equal(a_slot,0);bytes_equal(b_slot,2);

    vram_format_compute_api instrumented={};real_compute=t.format_api;instrumented=*real_compute;instrumented.launch=blocked_launch;t.format_api=&instrumented;
    struct cleanup { ~cleanup(){unblock();} } cleanup_on_error;
    t.begin(17,3,true);lifecycle_prepare(t,requests+2,2,slots);
    require(slots[0]==b_slot && slots[1]<0 && t.table->admissions[1]==a_slot,"global hit/admission mapping differs");
    lifecycle_launch(t,input);
    {std::unique_lock<std::mutex> lock(barrier_mutex);require(barrier_cv.wait_for(lock,std::chrono::seconds(5),[]{return entered;}),"global compute did not enter");}
    upload(3);
    require(!t.table->slots[a_slot].valid && t.table->slots[b_slot].valid,"pending overwrite published early or invalidated live hit");
    unblock();t.finish(out.data(),256);bytes_equal(a_slot,3);bytes_equal(b_slot,2);
    const auto valid=[](const vram_slot_table::entry & e){return e.valid;};
    require(t.table->find(requests[0],valid)<0 && t.table->find(requests[1],valid)<0 &&
            t.table->find(requests[2],valid)>=0 && t.table->find(requests[3],valid)>=0,
            "an overwritten payload is still held, or a cross-layer one is not");
    t.format_api=real_compute;
    t.begin(0,0,false);require(!t.arena && !t.table,"prefill did not release global arena");
    std::printf("REAL_CUDA_GLOBAL_POOL_PASS assertions=%d\n",assertions);return 0;
}catch(const std::exception & e){unblock();std::fprintf(stderr,"GLOBAL_POOL_CUDA_FAIL: %s\n",e.what());return 1;}}
