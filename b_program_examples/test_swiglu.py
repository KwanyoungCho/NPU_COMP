"""Validate SwiGLU FFN vs python ref.
FFN(x) = (SiLU(x@Wg) * (x@Wu)) @ Wd ,  SiLU(z)=z/(1+exp(-z))."""
import random, math, os
from npu import Asm, GBuf, matmul, run, fp16

random.seed(4)
SEQ, D, F = 8, 64, 128
g = GBuf()
aX=g.alloc(SEQ*D); aWg=g.alloc(D*F); aWu=g.alloc(D*F); aWd=g.alloc(F*D)
aZero=g.alloc(SEQ*F)                 # zeros (for negate)
aOne =g.alloc(SEQ*F); g.put(aOne,[1.0]*(SEQ*F))
aG=g.alloc(SEQ*F); aNg=g.alloc(SEQ*F); aDen=g.alloc(SEQ*F); aSig=g.alloc(SEQ*F)
aSilu=g.alloc(SEQ*F); aU=g.alloc(SEQ*F); aH=g.alloc(SEQ*F); aOut=g.alloc(SEQ*D)

def rnd(n): return [fp16(random.uniform(-1,1)) for _ in range(n)]
X=[rnd(D) for _ in range(SEQ)]; Wg=[rnd(F) for _ in range(D)]; Wu=[rnd(F) for _ in range(D)]; Wd=[rnd(D) for _ in range(F)]
def flat(m): return [v for r in m for v in r]
g.put(aX,flat(X)); g.put(aWg,flat(Wg)); g.put(aWu,flat(Wu)); g.put(aWd,flat(Wd))

a=Asm()
def vec(op,s1,s2,dst,n,mode=2,imm=0):
    a.vlen(n); a.addr(0,s1); a.load(0,0)
    if mode==2: a.addr(1,s2); a.load(0,1)
    op(mode=mode,imm=imm); a.addr(2,dst); a.save(0)

matmul(a, aX, aWg, aG, SEQ, D, F)                  # gate = x@Wg
vec(a.v_sub, aZero, aG, aNg, SEQ*F)                # -gate
a.vlen(SEQ*F); a.addr(0,aNg); a.load(0,0); a.v_exp(); a.addr(2,aDen); a.save(0)  # exp(-g)
vec(a.v_add, aDen, 0, aDen, SEQ*F, mode=0, imm=1)  # 1+exp(-g)
vec(a.v_div, aOne, aDen, aSig, SEQ*F)              # sigmoid = 1/(1+exp(-g))
vec(a.v_mul, aG, aSig, aSilu, SEQ*F)               # silu = g*sigmoid
matmul(a, aX, aWu, aU, SEQ, D, F)                  # up = x@Wu
vec(a.v_mul, aSilu, aU, aH, SEQ*F)                 # h = silu*up
matmul(a, aH, aWd, aOut, SEQ, F, D)                # out = h@Wd
a.halt()

out=run(a,g,g.top,rundir=os.path.dirname(__file__) or '.',maxrun=4000)

def mm(A,B,M,K,N): return [fp16(sum(A[i*K+k]*B[k*N+j] for k in range(K))) for i in range(M) for j in range(N)]
Xf=flat(X)
G=mm(Xf,flat(Wg),SEQ,D,F)
Sig=[fp16(1.0/fp16(1.0+fp16(math.exp(-G[k])))) for k in range(SEQ*F)]
Silu=[fp16(G[k]*Sig[k]) for k in range(SEQ*F)]
U=mm(Xf,flat(Wu),SEQ,D,F)
H=[fp16(Silu[k]*U[k]) for k in range(SEQ*F)]
Outr=mm(H,flat(Wd),SEQ,F,D)
got=[out[aOut+k] for k in range(SEQ*D)]
mism=sum(1 for x,y in zip(got,Outr) if x!=y); maxerr=max(abs(x-y) for x,y in zip(got,Outr))
print(f"SwiGLU FFN [SEQ={SEQ},D={D},F={F}]: mismatch={mism}/{SEQ*D} maxerr={maxerr:.6g} instr={len(a.p)}")
print("got[:5]:", [round(v,4) for v in got[:5]]); print("exp[:5]:", [round(v,4) for v in Outr[:5]])
print("RESULT:", "PASS" if maxerr<3e-2 else "FAIL")
