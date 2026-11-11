#include "explicit-fixture-inputs.h"
// CUDA transport lifecycle test. A launch the device holds until the test lets
// it go makes borrowed source lifetime observable, independently of GPU/CPU
// scheduling speed.
#include "gpu-services.h"
#include "vram-tier.cpp"
#include "test-ssd-vram-format-fixture.h"
#include <functional>

namespace {
std::mutex barrier_mutex;
std::condition_variable barrier_cv;
bool entered = false, proceed = false, inject_error = false;
const vram_format_compute_api * real_compute = nullptr;
// A launched call waits on the device for a host function of a gate stream, which returns once `proceed` is set.
cudaStream_t gate_stream = nullptr;
cudaEvent_t gate_event = nullptr;
void CUDART_CB gate_host(void *) {
    std::unique_lock<std::mutex> lock(barrier_mutex); barrier_cv.wait(lock,[] { return proceed; });
}
int blocked_launch(void * p,const vram_compute_expert * w,int n,int first,const void * q,size_t bytes,
                   const kimi_vram_activation * activation,CUevent_st * ready,CUevent_st * start,CUevent_st * end,
                   char * error,size_t size) {
    { std::lock_guard<std::mutex> lock(barrier_mutex); entered = true; barrier_cv.notify_all(); }
    if (inject_error) { std::snprintf(error,size,"injected worker error"); return 1; }
    if (!gate_stream) {
        cuda_check(cudaStreamCreateWithFlags(&gate_stream,cudaStreamNonBlocking));
        cuda_check(cudaEventCreateWithFlags(&gate_event,cudaEventDisableTiming));
    }
    if (ready) { cuda_check(cudaStreamWaitEvent(gate_stream,ready,0)); }
    cuda_check(cudaLaunchHostFunc(gate_stream,gate_host,nullptr));
    cuda_check(cudaEventRecord(gate_event,gate_stream));
    return real_compute->launch(p,w,n,first,q,bytes,activation,gate_event,start,end,error,size);
}
int assertions = 0;
void require(bool ok,const char * what) { ++assertions; if(!ok)throw std::runtime_error(what); }
void rejected(const std::function<void()> & fn) {
    bool error=false;try{fn();}catch(const std::exception &){error=true;}require(error,"unsafe residency operation accepted");
}
void unblock() { std::lock_guard<std::mutex> lock(barrier_mutex); proceed=true;barrier_cv.notify_all(); }
}
int main() {
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services()); try {
    inputs.set("vram_async", "1");
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");
    const size_t role=lifecycle_role_bytes();
    std::vector<unsigned char> payload(role,0);std::vector<float> input(256,0.0f);payload.back()=37;
    const void * data[]={payload.data(),payload.data(),payload.data()};
    kimi_vram_request requests[2]={};
    for(int j=0;j<2;++j){requests[j].layer=1;requests[j].expert=j;requests[j].weight=.7f;
        for(int i=0;i<3;++i)requests[j].role[i]={1,2,1000000,uint64_t(3*j+i)*role,role,3,4,5,6};}
    vram_format_compute_api instrumented={};
    tier t(32ULL<<20, inputs.tier_policy());
    struct cleanup { ~cleanup(){unblock();} } cleanup_on_error;
    int slots[2]={};std::vector<float> out(512);
    auto warm=[&]{t.begin(17,1,true);lifecycle_prepare(t,requests,1,slots);t.admit(requests[0],data,true);t.finish(out.data(),256);};
    warm();
    // CUDA loads a kernel's module at its first launch, which waits for the device: one launch before a call is held.
    t.begin(17,1,true);lifecycle_prepare(t,requests,1,slots);lifecycle_launch(t,input);t.finish(out.data(),256);
    real_compute=t.format_api;instrumented=*real_compute;instrumented.launch=blocked_launch;
    t.format_api=&instrumented;
    t.begin(17,2,true);lifecycle_prepare(t,requests,2,slots);
    require(slots[0]>=0 && slots[1]<0,"mixed hit/admission not exercised");
    const int destination=t.table->admissions[1];
    lifecycle_launch(t,input);
    {std::unique_lock<std::mutex> lock(barrier_mutex);require(barrier_cv.wait_for(lock,std::chrono::seconds(5),[]{return entered;}),"launch did not enter");}
    t.admit(requests[1],data,true);
    // Compute is still held. admit must have returned with source ownership
    // held, without either publishing a hit or copying through the CPU thread.
    require(!proceed && t.pending_uploads.size()==1 && !t.table->slots[destination].valid,"borrowed upload was published early");
    kimi_vram_ready_lease lease={};int ready[2]={};
    rejected([&]{t.acquire_ready(17,requests,2,&lease,ready);});
    unblock();t.finish(out.data(),256);
    require(t.done && t.pending_uploads.empty() && t.table->slots[destination].valid,"finish did not publish completed upload");
    std::vector<unsigned char> copied(3*role);
    cuda_check(cudaMemcpy(copied.data(),t.arena+size_t(destination)*t.stride,copied.size(),cudaMemcpyDeviceToHost));
    for(int i=0;i<3;++i)require(!std::memcmp(copied.data()+i*role,payload.data(),role),"borrowed payload corrupted");
    std::fill(payload.begin(),payload.end(),0xff);
    rejected([&]{t.acquire_ready(18,requests,2,&lease,ready);});
    t.acquire_ready(17,requests,2,&lease,ready);require(ready[0] && ready[1],"ready confirmation missing");
    rejected([&]{t.begin(18,1,true);});
    const auto old=lease;auto stale=lease;++stale.generation;
    rejected([&]{t.release_ready(stale);});t.validate_ready(lease);t.release_ready(lease);
    rejected([&]{t.validate_ready(old);});
    t.release();std::fill(payload.begin(),payload.end(),0);warm();
    t.acquire_ready(17,requests,1,&lease,ready);require(lease.generation>old.generation,"arena recreation reused a confirmation");
    rejected([&]{t.validate_ready(old);});t.release_ready(lease);
    {
        std::lock_guard<std::mutex> lock(barrier_mutex);entered=proceed=false;inject_error=true;
    }
    // A launch refused after a borrowed upload was queued fails the layer at finish, which leaves no borrowed work live.
    t.begin(17,2,true);lifecycle_prepare(t,requests,2,slots);t.admit(requests[1],data,true);lifecycle_launch(t,input);
    {std::unique_lock<std::mutex> lock(barrier_mutex);require(barrier_cv.wait_for(lock,std::chrono::seconds(5),[]{return entered;}),"error launch did not enter");}
    unblock();char error[512]={};
    require(api_finish(&t,out.data(),256,error,sizeof(error))!=0 && std::strstr(error,"injected worker error"),"worker failure was hidden");
    require(!t.arena && !t.table && t.done && t.pending_uploads.empty() && t.upload_jobs.empty(),"error returned with borrowed work live");
    t.format_api=real_compute;
    std::printf("REAL_CUDA_ASYNC_LEASE_PASS assertions=%d\n",assertions);return 0;
}catch(const std::exception & e){unblock();std::fprintf(stderr,"ASYNC_LEASE_FAIL: %s\n",e.what());return 1;} }
