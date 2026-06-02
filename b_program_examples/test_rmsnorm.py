"""Validate RMSNorm codegen (reduction + broadcast via matmul-with-ones) vs python ref.
RMSNorm(x)_j = x_j / sqrt(mean_k x_k^2) * w_j   (eps=0 for this check)."""
import random, math, os
from npu import Asm, GBuf, matmul, run, fp16

random.seed(2)
SEQ, D = 8, 64
g = GBuf()
aX   = g.alloc(SEQ*D)          # input  [SEQ,D]
aW   = g.alloc(D)              # norm weight [D]
aOne = g.alloc(D)             ; g.put(aOne, [1.0]*D)        # ones vector
aOut = g.alloc(SEQ*D)          # output
# per-token scratch
aSq  = g.alloc(D); aSS=g.alloc(1); aMean=g.alloc(1); aRms=g.alloc(1); aInv=g.alloc(1)
aScale=g.alloc(D); aTmp=g.alloc(SEQ*D)

X = [[fp16(random.uniform(-2,2)) for _ in range(D)] for _ in range(SEQ)]
W = [fp16(random.uniform(0.5,1.5)) for _ in range(D)]
g.put(aX, [X[i][j] for i in range(SEQ) for j in range(D)])
g.put(aW, W)

a = Asm()
def vec(op, src1, src2, dst, n, mode=2, imm=0):
    a.vlen(n); a.addr(0, src1); a.load(0,0)
    if mode==2: a.addr(1, src2); a.load(0,1)
    op(mode=mode, imm=imm); a.addr(2, dst); a.save(0)

for t in range(SEQ):
    xt = aX + t*D; ot = aOut + t*D
    vec(a.v_mul, xt, xt, aSq, D)                 # sq = x*x
    matmul(a, aSq, aOne, aSS, 1, D, 1)           # ss = sum(sq)  (1xD @ Dx1)
    vec(a.v_div, aSS, 0, aMean, 1, mode=0, imm=D) # mean = ss/D
    a.vlen(1); a.addr(0,aMean); a.load(0,0); a.v_sqrt(); a.addr(2,aRms); a.save(0)  # rms
    vec(a.v_div, aOne, aRms, aInv, 1)            # inv = 1/rms (ones[1]/rms)
    matmul(a, aOne, aInv, aScale, D, 1, 1)       # scale = broadcast inv to D (Dx1 @ 1x1)
    vec(a.v_mul, xt, aScale, aTmp+t*D, D)        # x * scale
    vec(a.v_mul, aTmp+t*D, aW, ot, D)            # * weight
a.halt()

gn = g.top
out = run(a, g, gn, rundir=os.path.dirname(__file__) or '.', maxrun=2000)

# reference
def ref_row(x):
    ms = sum(v*v for v in x)/D
    inv = 1.0/math.sqrt(ms)
    return [fp16(fp16(fp16(x[j]*fp16(inv)))*W[j]) for j in range(D)]
exp = [ref_row(X[i]) for i in range(SEQ)]
got = [[out[aOut+i*D+j] for j in range(D)] for i in range(SEQ)]
flat_g=[got[i][j] for i in range(SEQ) for j in range(D)]
flat_e=[exp[i][j] for i in range(SEQ) for j in range(D)]
mism=sum(1 for x,y in zip(flat_g,flat_e) if x!=y)
maxerr=max(abs(x-y) for x,y in zip(flat_g,flat_e))
print(f"RMSNorm [{SEQ},{D}]: mismatch={mism}/{len(flat_e)} maxerr={maxerr:.6g}  instr={len(a.p)}")
print("got[0][:5]:", [round(v,4) for v in got[0][:5]])
print("exp[0][:5]:", [round(v,4) for v in exp[0][:5]])
print("RESULT:", "PASS" if maxerr<1e-2 else "FAIL")
