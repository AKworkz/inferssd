#include "explicit-fixture-inputs.h"
// Retired implicit ABI slots must reject callers while draining actual
// explicit-format CUDA work before releasing its cache-owned weights.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gpu-services.h"
#include "vram-tier.cpp"
#include "test-kimi-vram-format-fixture.h"
#include <cstdio>

namespace {
void require(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
}
int main(){
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services());try{
    inputs.set("tier.min_count", "0");
    inputs.set("tier.policy", "lru");
    const auto role=lifecycle_role_bytes();std::vector<unsigned char> payload(role,0);
    const void * data[]={payload.data(),payload.data(),payload.data()};
    kimi_vram_request request={};request.layer=1;request.expert=0;request.weight=1;
    for(int i=0;i<3;++i)request.role[i]={1,2,1000000,uint64_t(i)*role,role,3,4,5,6};
    tier t(32ULL<<20, inputs.tier_policy());int slot=-1;std::vector<float> input(256,0),output(256);char error[256]={};
    auto warm=[&]{t.begin(17,1,true);lifecycle_prepare(t,&request,1,&slot);t.admit(request,data,true);t.finish(output.data(),256);};
    warm();require(t.table && t.arena,"explicit cache was not allocated");
    require(api_prepare(&t,&request,1,256,256,&slot,error,sizeof(error))!=0 &&
        std::strstr(error,"implicit expert format retired") && !t.table && !t.arena && t.done && t.pending_uploads.empty(),
        "implicit prepare did not reject and drain its context");
    warm();t.begin(17,2,true);lifecycle_prepare(t,&request,1,&slot);require(slot>=0,"explicit cache hit absent");
    lifecycle_launch(t,input);
    require(api_launch(&t,input.data(),input.size()*sizeof(float),4,25,error,sizeof(error))!=0 &&
        std::strstr(error,"implicit expert format retired") && !t.table && !t.arena && t.done && t.pending_uploads.empty(),
        "implicit launch returned with explicit CUDA work or cache still live");
    warm();t.begin(17,2,true);lifecycle_prepare(t,&request,1,&slot);lifecycle_launch(t,input);
    int admitted=99;
    require(prefill_cold_admission_active(&t)==0 &&
        prefill_admit_cold(&t,&request,data[0],data[1],data[2],&admitted,error,sizeof(error))!=0 &&
        !admitted && std::strstr(error,"Q2 cold admission retired") && !t.table && !t.arena && t.done && t.pending_uploads.empty(),
        "retired cold admission returned with CUDA work or cache still live");
    std::puts("REAL_CUDA_FORMAT_COMPAT_DRAIN_PASS checks=5");return 0;
}catch(const std::exception & e){std::fprintf(stderr,"FORMAT_COMPAT_DRAIN_FAIL: %s\n",e.what());return 1;}}
