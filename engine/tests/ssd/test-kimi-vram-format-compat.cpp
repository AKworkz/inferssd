#include "explicit-fixture-inputs.h"
// Retired implicit ABI slots must reject callers while draining actual
// explicit-format CUDA work before releasing its cache-owned weights.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gpu-services.h"
#include "vram-tier.cpp"
#include "test-kimi-vram-format-fixture.h"
#include <cstdio>
#include <functional>

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
    const auto * location=kimi_vram_location_get_api();
    require(location && location->version==1 && location->size==sizeof(*location) && location->configure &&
        location->view && location->view->version==1 && location->view->size==sizeof(*location->view),
        "retired location ABI layout changed");
    const auto & view=*location->view;kimi_cache_location_lease native_lease={77,88};
    struct location_case { const char * name; std::function<int()> call; bool rejected; };
    std::vector<location_case> cases={
        {"configure",[&]{return location->configure(&t,256,256,error,sizeof(error));},true},
        {"acquire",[&]{return view.acquire_native(&t,nullptr,1,&native_lease,error,sizeof(error));},true},
        {"begin",[&]{return view.begin_layer(&t,nullptr,0,0,0,error,sizeof(error));},true},
        {"launch",[&]{return view.launch(&t,nullptr,0,nullptr,0,0,0,0,0,error,sizeof(error));},true},
        {"admit",[&]{return view.admit(&t,nullptr,nullptr,nullptr,nullptr,1,error,sizeof(error));},true},
        {"finish",[&]{return view.finish(&t,error,sizeof(error));},true},
        {"cancel",[&]{return view.cancel(&t,error,sizeof(error));},false},
        {"release",[&]{return view.release(&t,&native_lease,1,error,sizeof(error));},true},
        {"empty-release",[&]{return view.release(&t,nullptr,0,error,sizeof(error));},false},
    };
    for(const auto & test:cases){
        warm();t.begin(17,2,true);lifecycle_prepare(t,&request,1,&slot);require(slot>=0,"explicit cache hit absent");
        lifecycle_launch(t,input);native_lease={77,88};error[0]=0;
        const int rc=test.call();
        require((rc!=0)==test.rejected && (!test.rejected || std::strstr(error,"implicit cache-location provider retired")),test.name);
        require(!t.table && !t.arena && t.done && t.pending_uploads.empty() && t.work.empty(),
            "retired location callback returned with explicit CUDA work or borrowed weights still live");
        if(std::strcmp(test.name,"acquire")==0)require(!native_lease.handle && !native_lease.generation,"retired acquire issued a lease");
        if(!test.rejected)require(!error[0],"successful location cleanup retained an error");
    }
    std::puts("REAL_CUDA_FORMAT_COMPAT_DRAIN_PASS checks=5 native_callback_cases=9");return 0;
}catch(const std::exception & e){std::fprintf(stderr,"FORMAT_COMPAT_DRAIN_FAIL: %s\n",e.what());return 1;}}
