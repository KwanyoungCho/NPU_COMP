"""Validate single-head causal attention (pre-transposed K) vs python ref.
scores=Q@K^T/sqrt(HD)+mask -> softmax -> @V -> @Wo.  (RoPE added later.)"""
import random, math, os
from npu import Asm, GBuf, matmul, run, fp16

random.seed(3)
SEQ, D, HD = 8, 16, 16            # single head: D==HD
g = GBuf()
aX  = g.alloc(SEQ*D); aXt = g.alloc(D*SEQ)
aWq = g.alloc(D*HD);  aWkt= g.alloc(HD*D); aWv = g.alloc(D*HD); aWo = g.alloc(HD*D)
aMask = g.alloc(SEQ*SEQ)
aOcol = g.alloc(SEQ); g.put(aOcol,[1.0]*SEQ)        # ones [SEQ,1] and [1,SEQ]
# scratch
aQ=g.alloc(SEQ*HD); aKt=g.alloc(HD*SEQ); aV=g.alloc(SEQ*HD)
aS=g.alloc(SEQ*SEQ); aE=g.alloc(SEQ*SEQ); aSum=g.alloc(SEQ); aSB=g.alloc(SEQ*SEQ)
aP=g.alloc(SEQ*SEQ); aCtx=g.alloc(SEQ*HD); aOut=g.alloc(SEQ*D)

def rnd(n,a=-1,b=1): return [fp16(random.uniform(a,b)) for _ in range(n)]
X  = [rnd(D) for _ in range(SEQ)]
Wq = [rnd(HD) for _ in range(D)]; Wk=[rnd(HD) for _ in range(D)]; Wv=[rnd(HD) for _ in range(D)]
Wo = [rnd(D) for _ in range(HD)]
g.put(aX, [X[i][j] for i in range(SEQ) for j in range(D)])
g.put(aXt,[X[i][j] for j in range(D) for i in range(SEQ)])          # X^T
g.put(aWq,[Wq[i][j] for i in range(D) for j in range(HD)])
g.put(aWkt,[Wk[i][j] for j in range(HD) for i in range(D)])         # Wk^T  [HD,D]
g.put(aWv,[Wv[i][j] for i in range(D) for j in range(HD)])
g.put(aWo,[Wo[i][j] for i in range(HD) for j in range(D)])
NEG=-30000.0
g.put(aMask,[(0.0 if j<=i else NEG) for i in range(SEQ) for j in range(SEQ)])

a = Asm()
def vec(op, s1, s2, dst, n, mode=2, imm=0):
    a.vlen(n); a.addr(0,s1); a.load(0,0)
    if mode==2: a.addr(1,s2); a.load(0,1)
    op(mode=mode, imm=imm); a.addr(2,dst); a.save(0)

matmul(a, aX, aWq, aQ, SEQ, D, HD)          # Q  [SEQ,HD]
matmul(a, aWkt, aXt, aKt, HD, D, SEQ)       # K^T[HD,SEQ] = Wk^T @ X^T
matmul(a, aX, aWv, aV, SEQ, D, HD)          # V  [SEQ,HD]
matmul(a, aQ, aKt, aS, SEQ, HD, SEQ)        # scores [SEQ,SEQ]
vec(a.v_div, aS, 0, aS, SEQ*SEQ, mode=0, imm=int(math.isqrt(HD)))   # / sqrt(HD)=4
vec(a.v_add, aS, aMask, aS, SEQ*SEQ)        # + causal mask
a.vlen(SEQ*SEQ); a.addr(0,aS); a.load(0,0); a.v_exp(); a.addr(2,aE); a.save(0)  # exp
matmul(a, aE, aOcol, aSum, SEQ, SEQ, 1)     # rowsum [SEQ,1]
matmul(a, aSum, aOcol, aSB, SEQ, 1, SEQ)    # broadcast [SEQ,SEQ]
vec(a.v_div, aE, aSB, aP, SEQ*SEQ)          # softmax
matmul(a, aP, aV, aCtx, SEQ, SEQ, HD)       # context [SEQ,HD]
matmul(a, aCtx, aWo, aOut, SEQ, HD, D)      # output  [SEQ,D]
a.halt()

out = run(a, g, g.top, rundir=os.path.dirname(__file__) or '.', maxrun=4000)

# ---- python reference (same op order, fp16 round at each "save") ----
def mm(A,B,M,K,N):
    return [[fp16(sum(A[i*K+k]*B[k*N+j] for k in range(K))) for j in range(N)] for i in range(M)]
def flat(m): return [v for r in m for v in r]
Xf=flat(X); Xt=[X[i][j] for j in range(D) for i in range(SEQ)]
Wqf=flat(Wq); Wktf=[Wk[i][j] for j in range(HD) for i in range(D)]; Wvf=flat(Wv); Wof=flat(Wo)
Q=flat(mm(Xf,Wqf,SEQ,D,HD)); Kt=flat(mm(Wktf,Xt,HD,D,SEQ)); V=flat(mm(Xf,Wvf,SEQ,D,HD))
S=flat(mm(Q,Kt,SEQ,HD,SEQ))
S=[fp16(v/4) for v in S]
S=[fp16(S[i*SEQ+j]+(0.0 if j<=i else NEG)) for i in range(SEQ) for j in range(SEQ)]
E=[fp16(math.exp(v)) for v in S]
Sum=flat(mm(E,[1.0]*SEQ,SEQ,SEQ,1))
SB=flat(mm(Sum,[1.0]*SEQ,SEQ,1,SEQ))
P=[fp16(E[k]/SB[k]) for k in range(SEQ*SEQ)]
Ctx=flat(mm(P,V,SEQ,SEQ,HD))
Outr=flat(mm(Ctx,Wof,SEQ,HD,D))

got=[out[aOut+k] for k in range(SEQ*D)]
mism=sum(1 for x,y in zip(got,Outr) if x!=y); maxerr=max(abs(x-y) for x,y in zip(got,Outr))
print(f"attention 1-head [SEQ={SEQ},HD={HD}]: mismatch={mism}/{SEQ*D} maxerr={maxerr:.6g} instr={len(a.p)}")
print("got[:5]:", [round(v,4) for v in got[:5]])
print("exp[:5]:", [round(v,4) for v in Outr[:5]])
print("RESULT:", "PASS" if maxerr<2e-2 else "FAIL")
