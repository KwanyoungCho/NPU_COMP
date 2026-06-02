"""NPU c-model assembler + helpers (targets reimplemented mysim).
HW: vector unit 256 lanes, matrix PE array 64x64. Here we assume NO tiling
(one instruction per logical op) at reduced dims that fit the encoding.
"""
import struct, subprocess, os

def fp16(x):                       # round a python float through FP16
    return struct.unpack('<e', struct.pack('<e', float(x)))[0]

class Asm:
    """Emits 32-bit NPU instructions. operand: 0=1st,1=2nd,2=dest."""
    def __init__(self): self.p = []
    def emit(self, w): self.p.append(w & 0xFFFFFFFF)
    def nop(self): self.emit(0)
    def halt(self): self.emit(0xFF)
    def addr(self, operand, a):
        self.emit((operand<<30)|(0<<29)|((a & 0xFFFF)<<8)|0x80)        # low 16
        self.emit((operand<<30)|(1<<29)|(((a>>16)&0xFFFF)<<8)|0x80)    # high 16
    def vlen(self, n): self.emit(((n & 0xFFFF)<<8)|0x82)
    def tile(self, matrix, d1, d2): self.emit((matrix<<31)|((d2&0xFF)<<16)|((d1&0xFF)<<8)|0x88)
    def load(self, matrix, operand): self.emit((matrix<<31)|(operand<<30)|0x90)
    def save(self, matrix): self.emit((matrix<<31)|0x98)
    # ---- vector compute (mode: 0=imm,1=scalar,2=vector) ----
    def v_add(self, mode=2, imm=0): self.emit((mode<<30)|((imm&0xFFFF)<<8)|0x01)
    def v_sub(self, mode=2, imm=0): self.emit((mode<<30)|((imm&0xFFFF)<<8)|0x02)
    def v_mul(self, mode=2, imm=0): self.emit((mode<<30)|((imm&0xFFFF)<<8)|0x0A)
    def v_div(self, mode=2, imm=0): self.emit((mode<<30)|((imm&0xFFFF)<<8)|0x0B)
    def v_muladd(self, mode=2, imm=0): self.emit((mode<<30)|((imm&0xFFFF)<<8)|0x0C)
    def v_move(self, mode=2, imm=0): self.emit((mode<<30)|((imm&0xFFFF)<<8)|0x0D)
    def v_sqrt(self): self.emit(0x0E)
    def v_exp(self): self.emit(0x0F)
    def v_max(self, mode=2, imm=0): self.emit((mode<<30)|(1<<28)|((imm&0xFFFF)<<8)|0x12)
    def v_min(self, mode=2, imm=0): self.emit((mode<<30)|(0<<28)|((imm&0xFFFF)<<8)|0x12)
    # ---- matrix compute ----
    def m_add(self, mode=2, imm=0, act=False): self.emit((mode<<30)|((1 if act else 0)<<29)|((imm&0xFFFF)<<8)|0x40)
    def m_mul(self, mode=2, imm=0, act=False): self.emit((mode<<30)|((1 if act else 0)<<29)|((imm&0xFFFF)<<8)|0x42)
    def bytes(self):
        return b''.join(struct.pack('<I', w) for w in self.p) + b'\n'

# ---- high-level macros ----
def matmul(asm, aAddr, bAddr, cAddr, M, K, N, act=False):
    """C[M,N] = A[M,K] @ B[K,N], one-shot (no tiling)."""
    asm.tile(0, M, K); asm.tile(1, K, N)
    asm.addr(0, aAddr); asm.addr(1, bAddr)
    asm.load(1, 0); asm.load(1, 1)
    asm.m_mul(mode=2, act=act)
    asm.addr(2, cAddr); asm.save(1)

class GBuf:
    """G-buffer memory: bump-allocate regions, fill with data."""
    def __init__(self, size=1<<16):
        self.mem = [0.0]*size; self.top = 0
    def alloc(self, n):
        a = self.top; self.top += n; return a
    def put(self, addr, flat):
        for i, v in enumerate(flat): self.mem[addr+i] = fp16(v)
    def get(self, addr, n):
        return self.mem[addr:addr+n]
    def bytes(self, count):
        return b''.join(struct.pack('<e', fp16(self.mem[i])) for i in range(count)) + b'\n'

def run(asm, gbuf, gn, rundir='.', maxrun=400):
    """Write program+gbuffer, run mysim, return written-back G-buffer as floats."""
    open(os.path.join(rundir,'program_memory.bin'),'wb').write(asm.bytes())
    open(os.path.join(rundir,'G_buffer_data.bin'),'wb').write(gbuf.bytes(gn))
    subprocess.run([os.path.join(rundir,'mysim'),'--run',str(maxrun),
                    '--gout',os.path.join(rundir,'gout.bin'),'--gn',str(gn)],
                   cwd=rundir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = open(os.path.join(rundir,'gout.bin'),'rb').read()
    return list(struct.unpack('<%de'%gn, d[:gn*2]))
