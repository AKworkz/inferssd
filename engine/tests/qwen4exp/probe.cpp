// SPDX-License-Identifier: Apache-2.0
#include "llama.h"
#include "ggml-backend.h"
#include <cstring>
#include <iomanip>
#ifdef QWEN_INTERNAL
#include "llama-model.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
#ifndef QWEN_UPSTREAM
#include "llama-ssd.h"
#endif
static void require(bool v, const char * msg) { if (!v) throw std::runtime_error(msg); }
static std::vector<float> decode(llama_context * c, const std::vector<llama_token> & t, int lo, int hi, int chunk, int seq=0) {
    std::vector<float> out;
    const bool last_only = std::getenv("QWEN_LAST_LOGIT_ONLY") != nullptr;
    for (int p=lo;p<hi;p+=chunk) {
        int n=std::min(chunk,hi-p);auto b=llama_batch_init(n,0,1);b.n_tokens=n;
        for(int i=0;i<n;++i){ b.token[i]=t[p+i];b.pos[i]=p+i;b.n_seq_id[i]=1;b.seq_id[i][0]=seq;b.logits[i]=!last_only || i==n-1; }
        int rc=llama_decode(c,b);llama_batch_free(b);require(rc==0,"decode failed");
        for(int i=last_only?n-1:0;i<n;++i){ const float * l=llama_get_logits_ith(c,i);require(l,"missing logits");for(int j=0;j<256;++j){require(std::isfinite(l[j]),"nonfinite logits");out.push_back(l[j]);}}
    }
    return out;
}
static void equal(const std::vector<float>& a,const std::vector<float>& b,const char * label){
    require(a.size()==b.size(),"logit size mismatch");double max=0,err=0,mag=0;
    for(size_t i=0;i<a.size();++i){double d=a[i]-b[i];max=std::max(max,std::abs(d));err+=d*d;mag+=a[i]*a[i];}
    std::cout<<label<<" max_abs="<<max<<" nmse="<<err/mag<<"\n";
    require(max<1e-5,"state restore/copy mismatch");
}
int main(int argc,char **argv){try{
    require(argc==6,"usage: probe fixture real-text logits chunk flash(0/1)");llama_backend_init();
    std::ifstream f(argv[2],std::ios::binary);std::string text((std::istreambuf_iterator<char>(f)),{});const int count=std::getenv("QWEN_TOKEN_COUNT") ? std::atoi(std::getenv("QWEN_TOKEN_COUNT")) : 48;
    require(count>0 && count<=96 && text.size()>=size_t(count),"real text too short or bad token count");
    std::vector<llama_token> t;for(unsigned char x:text.substr(0,count))t.push_back(x);
    std::cout<<"byte-vocab fixture input IDs:";for(auto x:t)std::cout<<" "<<x;std::cout<<"\n";
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;
#ifdef QWEN_UPSTREAM
    mp.load_mode=LLAMA_LOAD_MODE_MMAP;mp.tensor_read_lazy=LLAMA_TENSOR_READ_LAZY_ON;
#else
    mp.use_mmap=true;mp.use_mlock=false;
    mp.defer_experts=std::getenv("QWEN_STREAMED")!=nullptr;
#endif
    if (const char * g=std::getenv("QWEN_GPU_LAYERS")) mp.n_gpu_layers=std::atoi(g);
    auto m=llama_model_load_from_file(argv[1],mp);require(m,"model load failed");
    std::ofstream audit;
    if (const char * path=std::getenv("QWEN_ROUTE_AUDIT")) audit.open(path);
    auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=64;cp.n_ubatch=64;cp.n_seq_max=2;cp.kv_unified=true;cp.offload_kqv=false;cp.op_offload=false;cp.n_threads=2;cp.n_threads_batch=2;
    if (const char * u=std::getenv("QWEN_UBATCH")) cp.n_ubatch=std::atoi(u);
    if (audit.is_open()) {
        cp.cb_eval_user_data=&audit;
        cp.cb_eval=[](ggml_tensor * tensor, bool ask, void * data) {
            const char * name=ggml_get_name(tensor);
            const bool selected=!std::strncmp(name,"ffn_moe_argsort-",16) || !std::strncmp(name,"ffn_moe_weights_norm-",21);
            if (ask) return selected;
            if (!selected) return true;
            auto & out=*static_cast<std::ofstream *>(data);
            const size_t n=ggml_nelements(tensor);
            require(ggml_is_contiguous(tensor),"noncontiguous route audit");
            out<<"{\"tensor\":\""<<name<<"\",\"width\":"<<tensor->ne[0]<<",\"values\":[";
            if(tensor->type==GGML_TYPE_I32){std::vector<int32_t> v(n);ggml_backend_tensor_get(tensor,v.data(),0,n*sizeof(int32_t));for(size_t i=0;i<n;++i)out<<(i?",":"")<<v[i];}
            else {require(tensor->type==GGML_TYPE_F32,"bad route audit type");std::vector<float> v(n);ggml_backend_tensor_get(tensor,v.data(),0,n*sizeof(float));for(size_t i=0;i<n;++i)out<<(i?",":"")<<std::setprecision(9)<<v[i];}
            out<<"]}\n";return true;
        };
    }
    cp.flash_attn_type=std::atoi(argv[5])?LLAMA_FLASH_ATTN_TYPE_ENABLED:LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (const char * u=std::getenv("QWEN_UNIFIED")) cp.kv_unified=std::atoi(u)!=0;
#ifdef QWEN_INTERNAL
    require(m->hparams.expert_weights_scale==1.0f,"Qwen routed scale is not one");
    for (uint32_t il=0;il<m->hparams.n_layer;++il) {
        require(m->hparams.swiglu_clamp_exp[il]==0.0f,"Qwen expert activation is clamped");
        if (std::getenv("QWEN_STREAMED")) for(auto * t:{m->layers[il].ffn_up_exps,m->layers[il].ffn_gate_exps,m->layers[il].ffn_down_exps})
            require(t && !std::strcmp(ggml_backend_buffer_name(t->buffer),"CPU_Mapped"),"deferred expert is not in a mapped CPU buffer");
    }
    if(std::getenv("QWEN_LOAD_ONLY")){std::cout<<"QWEN_DEFERRED_LOAD_PASS\n";llama_model_free(m);llama_backend_free();return 0;}
    std::cout << "PLE_ADDR=" << m->per_layer_tok_embd->data << " PLE_BYTES=" << ggml_nbytes(m->per_layer_tok_embd) << " ROW_BYTES=" << ggml_row_size(m->per_layer_tok_embd->type,m->per_layer_tok_embd->ne[0]) << std::endl;
#endif
    if (std::getenv("QWEN_STREAMED")) cp.n_seq_max=1;
    auto c=llama_init_from_model(m,cp);require(c,"context failed");
    const int chunk=std::atoi(argv[4]);require(chunk>0,"bad chunk");
    auto phase = [&](bool active, int tokens) {
#ifndef QWEN_UPSTREAM
        if (std::getenv("QWEN_STREAMED")) require(llama_ssd_prefill_phase(c,active,tokens,tokens),"phase transition failed");
#endif
    };
    if (std::getenv("QWEN_WARMUP")) {
        phase(false,0);
        decode(c,t,0,std::min(count,24),1);
        llama_memory_clear(llama_get_memory(c),true);
    }
    const int prefill=std::getenv("QWEN_PREFILL_TOKENS") ? std::atoi(std::getenv("QWEN_PREFILL_TOKENS")) : count;
    require(prefill>=0 && prefill<=count,"bad prefill length");
    phase(prefill>0,prefill);
    auto logits=decode(c,t,0,prefill,chunk);
    if (prefill<count) phase(false,0);
    auto tail=decode(c,t,prefill,count,1);logits.insert(logits.end(),tail.begin(),tail.end());std::ofstream o(argv[3],std::ios::binary);o.write((const char*)logits.data(),logits.size()*sizeof(float));o.close();
    if (std::getenv("QWEN_PROBE_ONLY")) { llama_free(c);llama_model_free(m);llama_backend_free();return 0; }
    auto mem=llama_get_memory(c);llama_memory_clear(mem,true);decode(c,t,0,24,8);
    auto ssize=llama_state_seq_get_size(c,0);std::vector<uint8_t> seq(ssize);require(llama_state_seq_get_data(c,seq.data(),seq.size(),0)==ssize,"sequence save failed");
    auto size=llama_state_get_size(c);std::vector<uint8_t> state(size);require(llama_state_get_data(c,state.data(),state.size())==size,"whole state save failed");
    auto continuation=decode(c,t,24,48,1);
    require(llama_state_set_data(c,state.data(),state.size())==size,"whole state restore failed");equal(continuation,decode(c,t,24,48,1),"whole restore");
    // Occupy another sequence before restoration to exercise adopted KV/indexer slots.
    decode(c,t,0,8,8,1);
    require(llama_state_seq_set_data(c,seq.data(),seq.size(),0)==ssize,"sequence restore failed");equal(continuation,decode(c,t,24,48,1),"sequence restore with occupied cells");
    require(llama_state_set_data(c,state.data(),state.size())==size,"whole restore before copy failed");
    llama_memory_seq_cp(mem,0,1,0,-1);equal(continuation,decode(c,t,24,48,1,1),"sequence copy");
    std::cout<<"state_bytes="<<size<<" seq_bytes="<<ssize<<" fixture checks PASS\n";
    llama_free(c);llama_model_free(m);llama_backend_free();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
