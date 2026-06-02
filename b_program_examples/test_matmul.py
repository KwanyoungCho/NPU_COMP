"""Validate one-shot matmul (no tiling) on mysim vs pure-python reference."""
import random, struct, os
from npu import Asm, GBuf, matmul, run, fp16

random.seed(1)
M, K, N = 8, 64, 64
g = GBuf()
aA = g.alloc(M*K); aB = g.alloc(K*N); aC = g.alloc(M*N)
A = [[fp16(random.uniform(-1,1)) for _ in range(K)] for _ in range(M)]
B = [[fp16(random.uniform(-1,1)) for _ in range(N)] for _ in range(K)]
g.put(aA, [A[i][k] for i in range(M) for k in range(K)])
g.put(aB, [B[k][j] for k in range(K) for j in range(N)])

asm = Asm()
matmul(asm, aA, aB, aC, M, K, N)
asm.halt()

gn = aC + M*N
out = run(asm, g, gn, rundir=os.path.dirname(__file__) or '.')

# reference: float32 accumulate, FP16 round at store (matches mysim save)
ref = [[fp16(sum(A[i][k]*B[k][j] for k in range(K))) for j in range(N)] for i in range(M)]
got = [out[aC + i*N + j] for i in range(M) for j in range(N)]
exp = [ref[i][j] for i in range(M) for j in range(N)]

mism = sum(1 for a,b in zip(got,exp) if a!=b)
maxerr = max(abs(a-b) for a,b in zip(got,exp))
print(f"matmul {M}x{K} @ {K}x{N}: mismatches={mism}/{len(exp)} maxerr={maxerr:.6g}")
print("sample got:", [round(x,4) for x in got[:6]])
print("sample exp:", [round(x,4) for x in exp[:6]])
print("RESULT:", "PASS (byte-exact)" if mism==0 else "FAIL")
