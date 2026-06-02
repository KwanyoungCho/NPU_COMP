"""Full Llama3 transformer layer codegen for the NPU c-model (reimplemented mysim).
Assumes NO tiling / one-shot ops at reduced dims. Validated against a float reference.

  h = x + Attn(RMSNorm(x))           # GQA, RoPE, causal
  y = h + FFN(RMSNorm(h))            # SwiGLU
"""
import random, math, os
from npu import Asm, GBuf, matmul, run, fp16

# ---- reduced config (one-shot; all dims <=255) ----
SEQ, D, H, KV, HD, F = 8, 64, 4, 2, 16, 128

# ---- Llama3-8B config (Needs tiling!!!) ----
# SEQ, D, H, KV, HD, F = 1, 4096, 32, 8, 128, 14336

GPK = H // KV                       # q-heads per kv-head
EPS = 0.0                           # immediate can't encode tiny eps; ref uses 0 too
random.seed(7)
def rnd(n,a=-1.,b=1.): return [fp16(random.uniform(a,b)) for _ in range(n)]
def flat(m): return [v for r in m for v in r]

g = GBuf(1<<16)
# inputs / weights
WS=0.2                                                   # small weights (FP16-exp-safe scores)
X   = [rnd(D) for _ in range(SEQ)]
Wn1 = rnd(D, .8, 1.2); Wn2 = rnd(D, .8, 1.2)
Wq  = [[rnd(HD,-WS,WS) for _ in range(D)] for _ in range(H)]    # per q-head [D,HD]
Wk  = [[rnd(HD,-WS,WS) for _ in range(D)] for _ in range(KV)]   # per kv-head
Wv  = [[rnd(HD,-WS,WS) for _ in range(D)] for _ in range(KV)]
Wo  = [[rnd(D,-WS,WS) for _ in range(HD)] for _ in range(H)]    # per q-head [HD,D]
Wg  = [rnd(F,-WS,WS) for _ in range(D)]; Wu=[rnd(F,-WS,WS) for _ in range(D)]; Wd=[rnd(D,-WS,WS) for _ in range(F)]
# RoPE tables
def rope_cos_sin():
    cos=[[0.]*HD for _ in range(SEQ)]; sin=[[0.]*HD for _ in range(SEQ)]
    for p in range(SEQ):
        for i in range(HD//2):
            th=p*(10000.0**(-2.0*i/HD)); c=fp16(math.cos(th)); s=fp16(math.sin(th))
            cos[p][i]=c; cos[p][i+HD//2]=c; sin[p][i]=s; sin[p][i+HD//2]=s
    return cos,sin
COS,SIN=rope_cos_sin()

# ---- G-buffer placement ----
aX=g.alloc(SEQ*D);  g.put(aX,flat(X))
aXt=g.alloc(D*SEQ)                          # filled at runtime? no -> we transpose host-side of NORMED x
aWn1=g.alloc(D); g.put(aWn1,Wn1); aWn2=g.alloc(D); g.put(aWn2,Wn2)
aWq=[g.alloc(D*HD) for _ in range(H)];  [g.put(aWq[h],flat(Wq[h])) for h in range(H)]
aWkt=[g.alloc(HD*D) for _ in range(KV)]; [g.put(aWkt[k],[Wk[k][i][j] for j in range(HD) for i in range(D)]) for k in range(KV)]  # Wk^T
aWv=[g.alloc(D*HD) for _ in range(KV)]; [g.put(aWv[k],flat(Wv[k])) for k in range(KV)]
aWo=[g.alloc(HD*D) for _ in range(H)];  [g.put(aWo[h],flat(Wo[h])) for h in range(H)]
aWg=g.alloc(D*F); g.put(aWg,flat(Wg)); aWu=g.alloc(D*F); g.put(aWu,flat(Wu)); aWd=g.alloc(F*D); g.put(aWd,flat(Wd))
aCosQ=g.alloc(SEQ*HD); g.put(aCosQ,flat(COS)); aSinQ=g.alloc(SEQ*HD); g.put(aSinQ,flat(SIN))
aCosKt=g.alloc(HD*SEQ); g.put(aCosKt,[COS[p][d] for d in range(HD) for p in range(SEQ)])
aSinKt=g.alloc(HD*SEQ); g.put(aSinKt,[SIN[p][d] for d in range(HD) for p in range(SEQ)])
aOneSEQ=g.alloc(SEQ); g.put(aOneSEQ,[1.]*SEQ)
aOneD=g.alloc(D); g.put(aOneD,[1.]*D)
aOneF=g.alloc(SEQ*F); g.put(aOneF,[1.]*(SEQ*F))
aMask=g.alloc(SEQ*SEQ); g.put(aMask,[(0. if j<=i else -30000.) for i in range(SEQ) for j in range(SEQ)])
aZeroF=g.alloc(SEQ*F)
# scratch (reused)
sN=g.alloc(SEQ*D); sNt=g.alloc(D*SEQ)
sQ=g.alloc(SEQ*HD); sRh=g.alloc(SEQ*HD); sQ2=g.alloc(SEQ*HD)
sKt=[g.alloc(HD*SEQ) for _ in range(KV)]; sV=[g.alloc(SEQ*HD) for _ in range(KV)]; sKtmp=g.alloc(HD*SEQ)
sS=g.alloc(SEQ*SEQ); sE=g.alloc(SEQ*SEQ); sSum=g.alloc(SEQ); sSB=g.alloc(SEQ*SEQ); sP=g.alloc(SEQ*SEQ)
sCtx=g.alloc(SEQ*HD); sPart=g.alloc(SEQ*D); sAttn=g.alloc(SEQ*D)
sH=g.alloc(SEQ*D); sHt=g.alloc(D*SEQ)
sGate=g.alloc(SEQ*F); sNg=g.alloc(SEQ*F); sDen=g.alloc(SEQ*F); sSig=g.alloc(SEQ*F); sSilu=g.alloc(SEQ*F); sUp=g.alloc(SEQ*F); sFh=g.alloc(SEQ*F); sFout=g.alloc(SEQ*D)
aY=g.alloc(SEQ*D)
ssq=g.alloc(D); sss=g.alloc(1); smean=g.alloc(1); srms=g.alloc(1); sinv=g.alloc(1); sscale=g.alloc(D); stmp=g.alloc(SEQ*D)

a=Asm()
def vec(op,s1,s2,dst,n,mode=2,imm=0):
    a.vlen(n); a.addr(0,s1); a.load(0,0)
    if mode==2: a.addr(1,s2); a.load(0,1)
    op(mode=mode,imm=imm); a.addr(2,dst); a.save(0)
def unary(uop,s,dst,n):
    a.vlen(n); a.addr(0,s); a.load(0,0); uop(); a.addr(2,dst); a.save(0)
def copy(s,dst,n): vec(a.v_add,s,0,dst,n,mode=0,imm=0)

def rmsnorm(src, w, dst):           # per-token RMSNorm
    for t in range(SEQ):
        xt=src+t*D; ot=dst+t*D
        vec(a.v_mul,xt,xt,ssq,D)
        matmul(a,ssq,aOneD,sss,1,D,1)
        vec(a.v_div,sss,0,smean,1,mode=0,imm=D)
        unary(a.v_sqrt,smean,srms,1)
        vec(a.v_div,aOneD,srms,sinv,1)          # 1/rms (uses ones[0])
        matmul(a,aOneD,sinv,sscale,D,1,1)       # broadcast
        vec(a.v_mul,xt,sscale,stmp+t*D,D)
        vec(a.v_mul,stmp+t*D,w,ot,D)

def transpose(src, dst, R, C):      # [R,C] -> [C,R] via per-elem copy (no transpose op)
    for r in range(R):
        for c in range(C):
            copy(src+r*C+c, dst+c*R+r, 1)

def rope_Q(qaddr):                  # Q [SEQ,HD] rotate_half RoPE, in place -> sQ2
    half=HD//2
    for t in range(SEQ):
        base=qaddr+t*HD
        vec(a.v_sub,aZeroF,base+half,sRh+t*HD,half)     # rh[:half] = -q[half:]
        copy(base, sRh+t*HD+half, half)                 # rh[half:] = q[:half]
    vec(a.v_mul,qaddr,aCosQ,sQ,SEQ*HD)                  # q*cos
    vec(a.v_mul,sRh,aSinQ,sQ2,SEQ*HD)                   # rh*sin
    vec(a.v_add,sQ,sQ2,sQ2,SEQ*HD)                      # q_embed

def rope_Kt(ktaddr,dst):            # Kt [HD,SEQ] rotate_half over HD(row) -> contiguous blocks
    half=HD//2
    vec(a.v_sub,aZeroF,ktaddr+half*SEQ,sKtmp,half*SEQ)  # rh[:half rows] = -Kt[half:]
    copy(ktaddr, sKtmp+half*SEQ, half*SEQ)              # rh[half:] = Kt[:half]
    vec(a.v_mul,ktaddr,aCosKt,dst,HD*SEQ)               # kt*cos
    vec(a.v_mul,sKtmp,aSinKt,sKtmp,HD*SEQ)              # rh*sin
    vec(a.v_add,dst,sKtmp,dst,HD*SEQ)

# ===== Attention block =====
rmsnorm(aX, aWn1, sN)
transpose(sN, sNt, SEQ, D)                              # X_norm^T for K^T = Wk^T @ Xn^T
# zero attn accumulator
for t in range(SEQ): copy(aZeroF, sAttn+t*D, D)
# kv projections (per kv head)
for k in range(KV):
    matmul(a, aWkt[k], sNt, sKt[k], HD, D, SEQ)         # K^T = Wk^T @ Xn^T
    rope_Kt(sKt[k], sKt[k])
    matmul(a, sN, aWv[k], sV[k], SEQ, D, HD)            # V
for h in range(H):
    kv=h//GPK
    matmul(a, sN, aWq[h], sQ, SEQ, D, HD)               # Q_h
    rope_Q(sQ)                                          # -> sQ2
    matmul(a, sQ2, sKt[kv], sS, SEQ, HD, SEQ)           # scores
    vec(a.v_div, sS, 0, sS, SEQ*SEQ, mode=0, imm=int(math.isqrt(HD)))
    vec(a.v_add, sS, aMask, sS, SEQ*SEQ)
    unary(a.v_exp, sS, sE, SEQ*SEQ)
    matmul(a, sE, aOneSEQ, sSum, SEQ, SEQ, 1)
    matmul(a, sSum, aOneSEQ, sSB, SEQ, 1, SEQ)
    vec(a.v_div, sE, sSB, sP, SEQ*SEQ)
    matmul(a, sP, sV[kv], sCtx, SEQ, SEQ, HD)
    matmul(a, sCtx, aWo[h], sPart, SEQ, HD, D)
    vec(a.v_add, sAttn, sPart, sAttn, SEQ*D)            # accumulate output proj
# residual 1
vec(a.v_add, aX, sAttn, sH, SEQ*D)

# ===== FFN block =====
rmsnorm(sH, aWn2, sN)
matmul(a, sN, aWg, sGate, SEQ, D, F)
vec(a.v_sub, aZeroF, sGate, sNg, SEQ*F)
unary(a.v_exp, sNg, sDen, SEQ*F)
vec(a.v_add, sDen, 0, sDen, SEQ*F, mode=0, imm=1)
vec(a.v_div, aOneF, sDen, sSig, SEQ*F)
vec(a.v_mul, sGate, sSig, sSilu, SEQ*F)
matmul(a, sN, aWu, sUp, SEQ, D, F)
vec(a.v_mul, sSilu, sUp, sFh, SEQ*F)
matmul(a, sFh, aWd, sFout, SEQ, F, D)
vec(a.v_add, sH, sFout, aY, SEQ*D)
a.halt()

out = run(a, g, g.top, rundir=os.path.dirname(__file__) or '.', maxrun=20000)

# =========================== float reference ===========================
def mm(A,B,M,K,N): return [[sum(A[i][k]*B[k][j] for k in range(K)) for j in range(N)] for i in range(M)]
def rms(row,w):
    ms=sum(v*v for v in row)/D; inv=1.0/math.sqrt(ms+EPS); return [row[j]*inv*w[j] for j in range(D)]
def rope(vecrow,p):
    half=HD//2; out=[0.]*HD
    for d in range(HD):
        c=math.cos(p*(10000.0**(-2.0*(d%half)/HD))); s=math.sin(p*(10000.0**(-2.0*(d%half)/HD)))
        rh = -vecrow[d+half] if d<half else vecrow[d-half]
        out[d]=vecrow[d]*c+rh*s
    return out
Xn=[rms(X[i],Wn1) for i in range(SEQ)]
attn=[[0.]*D for _ in range(SEQ)]
Kt={}; Vh={}
for k in range(KV):
    Kk=mm(Xn,[[Wk[k][i][j] for j in range(HD)] for i in range(D)],SEQ,D,HD)   # [SEQ,HD]
    Kk=[rope(Kk[p],p) for p in range(SEQ)]
    Kt[k]=[[Kk[p][d] for p in range(SEQ)] for d in range(HD)]                  # [HD,SEQ]
    Vh[k]=mm(Xn,[[Wv[k][i][j] for j in range(HD)] for i in range(D)],SEQ,D,HD)
for h in range(H):
    kv=h//GPK
    Q=mm(Xn,[[Wq[h][i][j] for j in range(HD)] for i in range(D)],SEQ,D,HD)
    Q=[rope(Q[p],p) for p in range(SEQ)]
    S=mm(Q,Kt[kv],SEQ,HD,SEQ)
    sc=math.sqrt(HD)
    for i in range(SEQ):
        row=[ (S[i][j]/sc + (0. if j<=i else -30000.)) for j in range(SEQ)]
        m=[math.exp(v) for v in row]; tot=sum(m); p=[v/tot for v in m]
        for j in range(SEQ): S[i][j]=p[j]
    Ctx=mm(S,Vh[kv],SEQ,SEQ,HD)
    Part=mm(Ctx,[[Wo[h][i][j] for j in range(D)] for i in range(HD)],SEQ,HD,D)
    for i in range(SEQ):
        for j in range(D): attn[i][j]+=Part[i][j]
Hr=[[X[i][j]+attn[i][j] for j in range(D)] for i in range(SEQ)]
Hn=[rms(Hr[i],Wn2) for i in range(SEQ)]
Gt=mm(Hn,Wg,SEQ,D,F); Up=mm(Hn,Wu,SEQ,D,F)
Silu=[[Gt[i][j]/(1+math.exp(-Gt[i][j]))*Up[i][j] for j in range(F)] for i in range(SEQ)]
Fout=mm(Silu,Wd,SEQ,F,D)
Yr=[[Hr[i][j]+Fout[i][j] for j in range(D)] for i in range(SEQ)]

got=[out[aY+i*D+j] for i in range(SEQ) for j in range(D)]
exp=[Yr[i][j] for i in range(SEQ) for j in range(D)]
maxabs=max(abs(v) for v in exp)
maxerr=max(abs(x-y) for x,y in zip(got,exp))
print(f"FULL Llama3 layer [SEQ={SEQ},D={D},H={H},KV={KV},HD={HD},F={F}] instr={len(a.p)}")
print(f"  maxerr={maxerr:.5g}  max|out|={maxabs:.4g}  rel={maxerr/maxabs:.4g}")
print("  got[:6]:", [round(v,3) for v in got[:6]])
print("  exp[:6]:", [round(v,3) for v in exp[:6]])
print("RESULT:", "PASS (FP16 tol)" if maxerr < 0.05*maxabs+0.05 else "FAIL")
