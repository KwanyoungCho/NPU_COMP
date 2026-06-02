// NPU c-model reimplementation (+ improvements: uncapped buffers, write-back, halt).
// Storage in float; FP16 .bin I/O + per-op FP16 rounding via half<->float helpers.
// Trace output stays byte-exact with the original a.out for all 55 b_program examples.
//
// Usage: ./mysim [--run N] [--gout FILE] [--gn N] [--gbuf N]
//   --run  N    max instructions before halt (default 30; HALT opcode 0xFF stops earlier)
//   --gout FILE write G-buffer back to FILE as FP16 (enables multi-run chaining)
//   --gn   N    number of G-buffer entries to write back (default = input entry count)
//   --gbuf N    initial G-buffer capacity in FP16 entries (auto-grows anyway)
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
using namespace std;

static inline float h2f(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&0x1F, man=h&0x3FF, f;
    if(exp==0){ if(man==0) f=sign; else { int e=-1; do{e++; man<<=1;}while(!(man&0x400));
        man&=0x3FF; f=sign|((127-15-e)<<23)|(man<<13);} }
    else if(exp==0x1F) f=sign|0x7F800000|(man<<13);
    else f=sign|((exp-15+127)<<23)|(man<<13);
    float r; memcpy(&r,&f,4); return r;
}
static inline uint16_t f2h(float x){
    uint32_t f; memcpy(&f,&x,4);
    uint32_t sign=(f>>16)&0x8000; int exp=((f>>23)&0xFF)-127+15; uint32_t man=f&0x7FFFFF;
    if(((f>>23)&0xFF)==0xFF) return sign|0x7C00|(man?0x200:0);
    if(exp>=0x1F) return sign|0x7C00;
    if(exp<=0){ if(exp<-10) return sign; man|=0x800000;
        uint32_t m=man>>(14-exp); uint32_t rem=man&((1u<<(14-exp))-1), half=1u<<(13-exp);
        if(rem>half||(rem==half&&(m&1))) m++; return sign|m; }
    uint32_t m=man>>13, rem=man&0x1FFF; if(rem>0x1000||(rem==0x1000&&(m&1))) m++;
    uint32_t r=(exp<<10)|m; if(m==0x400) r=((exp+1)<<10); return sign|r;
}
static inline float fp16(float x){ return h2f(f2h(x)); }
static inline float sigmoid(float x){ return 1.0f/(1.0f+expf(-x)); }
static inline float act(float x){ return x*x*sigmoid(x); }   // w_act: x^2 * sigmoid(x)

int main(int argc, char** argv){
    long maxRun=30, gn=-1, gbufCap=1<<16;
    string goutFile;
    for(int i=1;i<argc;i++){ string a=argv[i];
        if(a=="--run"&&i+1<argc) maxRun=stol(argv[++i]);
        else if(a=="--gout"&&i+1<argc) goutFile=argv[++i];
        else if(a=="--gn"&&i+1<argc) gn=stol(argv[++i]);
        else if(a=="--gbuf"&&i+1<argc) gbufCap=stol(argv[++i]); }

    // ---- read G_buffer_data.bin (FP16 LE) — buffer auto-grows, no 8192 cap ----
    ifstream gf("G_buffer_data.bin", ios::binary);
    gf.seekg(0,ios::end); long gbytes=gf.tellg(); gf.seekg(0);
    long ginN = gbytes/2;
    vector<float> G((size_t)max((long)gbufCap, ginN), 0.f);
    { vector<char> raw(gbytes); gf.read(raw.data(),gbytes);
      for(long i=0;i<ginN;i++){ uint16_t b=(uint8_t)raw[2*i]|((uint8_t)raw[2*i+1]<<8); G[i]=h2f(b);} }
    auto ensureG=[&](long idx){ if(idx>=(long)G.size()) G.resize(idx+1024,0.f); };

    // ---- read program_memory.bin (uint32 LE) — no 32768 cap ----
    ifstream pf("program_memory.bin", ios::binary);
    pf.seekg(0,ios::end); long pbytes=pf.tellg(); pf.seekg(0);
    long pn = pbytes/4;
    vector<uint32_t> prog((size_t)max(pn,20L),0);
    { vector<char> raw(pbytes); pf.read(raw.data(),pbytes);
      for(long i=0;i<pn;i++)
        prog[i]=(uint8_t)raw[4*i]|((uint8_t)raw[4*i+1]<<8)|((uint8_t)raw[4*i+2]<<16)|((uint32_t)(uint8_t)raw[4*i+3]<<24); }

    // ---- header (byte-exact with original) ----
    cout << "G_buffer size :  " << gbytes << "\n\n\n\n";
    cout << "0\n";
    cout << "Program memory size :  " << hex << pbytes << dec << "\n\n\n\n";
    cout << "0\n";
    for(int k=0;k<20;k++) cout << hex << prog[k] << dec << "\n";
    cout << "\n\n\n";

    uint32_t blo[3]={0,0,0}, bhi[3]={0,0,0};
    long vlen=0, tA[2]={0,0}, tB[2]={0,0};
    vector<float> pin1, pin2, pout;
    auto base=[&](int o){ return (long)(blo[o]|(bhi[o]<<16)); };
    auto emit=[&](const vector<float>& v){ for(float x:v) cout<<x<<"\n"; };

    for(long pc=0; pc<maxRun; pc++){
        uint32_t instr = (pc<(long)prog.size())?prog[pc]:0;
        uint32_t op=instr&0xFF, mode=(instr>>30)&0x3;
        bool actf=(instr>>29)&0x1;
        cout << "p_counter :  " << pc << "\n";
        cout << "instruction :  " << hex << instr << dec << "\n";

        if(op==0xFF){ cout<<"HALT --- \n\n\n\n"; break; }     // NEW: clean halt
        else if(instr==0){ cout<<"NOP --- \n"; }
        else if(op==0x80){ int o=(instr>>30)&3, hi=(instr>>29)&1; uint32_t v=(instr>>8)&0xFFFF;
            if(hi) bhi[o]=v; else blo[o]=v; }
        else if(op==0x82){ vlen=(instr>>8)&0xFFFF; }
        else if(op==0x88){ int m=(instr>>31)&1; tA[m]=(instr>>8)&0xFF; tB[m]=(instr>>16)&0xFF; }
        else if(op==0x90){ int matrix=(instr>>31)&1, o=(instr>>30)&1; long b=base(o);
            long n=matrix? tA[o]*tB[o] : vlen;
            vector<float>& d=(o==0)?pin1:pin2; d.assign(n,0.f);
            for(long i=0;i<n;i++){ ensureG(b+i); d[i]=G[b+i]; }
            const char* lbl=(o==0)?"PE_in_data_1_array :  ":"PE_in_data_2_array :  ";
            for(long i=0;i<n;i++) cout<<lbl<<d[i]<<"\n"; }
        else if(op==0x98){ long b=base(2),n=(long)pout.size();
            for(long i=0;i<n;i++){ ensureG(b+i); G[b+i]=fp16(pout[i]); }   // FP16 round on store
            for(long i=0;i<n;i++) cout<<"PE_out_array :  "<<pout[i]<<"\n"; }
        else { // ---- compute (float32 internally; FP16 only at store) ----
            bool matrix = (op>=0x40 && op<=0x43);
            int cst=(instr>>8)&0xFFFF;
            auto B=[&](long i)->float{ return (mode==2)? pin2[i] : (float)cst; };
            if(matrix && op==0x42 && mode==2){          // real matrix multiply
                long rA=tA[0],cA=tB[0],cB=tB[1]; pout.assign(rA*cB,0.f);
                for(long i=0;i<rA;i++)for(long j=0;j<cB;j++){ float a=0;
                    for(long k=0;k<cA;k++) a+=pin1[i*cA+k]*pin2[k*cB+j]; pout[i*cB+j]=a; }
                if(actf) for(auto&v:pout) v=act(v);
            } else {
                long n = matrix ? tA[0]*tB[0] : vlen;
                if((long)pout.size()!=n) pout.assign(n,0.f);
                for(long i=0;i<n;i++){
                    float a=pin1[i], r=a;
                    switch(op){
                        case 0x01: case 0x40: r=a+B(i); break;
                        case 0x02: case 0x41: r=a-B(i); break;
                        case 0x0A: case 0x42: r=a*B(i); break;
                        case 0x0B: r=a/B(i); break;
                        case 0x0C: r=pout[i]+a*B(i); break;
                        case 0x0D: case 0x43: r=B(i); break;
                        case 0x0E: r=sqrtf(a); break;
                        case 0x0F: r=expf(a); break;
                        case 0x11: r=(a==B(i))?1.f:0.f; break;
                        case 0x12: { float b=B(i); int mx=(instr>>28)&1; r=mx?(a>b?a:b):(a<b?a:b);} break;
                        case 0x13: r=(float)(long)a; break;
                        case 0x09: { int16_t s=(int16_t)cst; if(mode==2) s=(int16_t)(int)pin2[i];
                                     r=a*powf(2.f,(float)s);} break;
                        case 0x08: { int sub=(instr>>27)&0x7; long ia=(long)a, ib=(long)B(i);
                                     switch(sub){case 0:r=ia&ib;break;case 1:r=ia|ib;break;
                                       case 2:r=~ia;break;case 3:r=ia^ib;break;
                                       case 4:r=~(ia&ib);break;case 5:r=~(ia|ib);break;} } break;
                    }
                    if(actf && matrix) r=act(r);
                    pout[i]=r;
                }
            }
            emit(pout);
        }
        cout << "\n\n\n";
    }

    // ---- NEW: write-back G-buffer to file (FP16 LE + trailing newline, like the input .bin) ----
    if(!goutFile.empty()){
        long cnt = (gn>=0)? gn : ginN;
        ofstream of(goutFile, ios::binary);
        for(long i=0;i<cnt;i++){ uint16_t h=f2h(i<(long)G.size()?G[i]:0.f);
            of.put((char)(h&0xFF)); of.put((char)((h>>8)&0xFF)); }
        of.put('\n');
        cerr << "[mysim] wrote " << cnt << " FP16 entries -> " << goutFile << "\n";
    }
    return 0;
}
