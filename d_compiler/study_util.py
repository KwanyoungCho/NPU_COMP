"""학습용 walkthrough 공용 헬퍼: 배너 출력 + 명령어 디스어셈블러.

disasm(word): 32비트 NPU 명령어 하나를 사람이 읽는 한 줄로 바꾼다.
(isa.py의 인코딩 규칙을 거꾸로 해석. STUDY.md §3 참고)
"""

def banner(n, title):
    print(f"\n{'='*72}\n[단계 {n}] {title}\n{'='*72}")


def disasm(w):
    op = w & 0xFF
    mode = (w >> 30) & 3
    modes = {0: "즉시값", 1: "스칼라", 2: "벡터"}
    if w == 0:
        return "NOP"
    if op == 0xFF:
        return "HALT  (실행 종료)"
    if op == 0x80:
        operand = (w >> 30) & 3; hi = (w >> 29) & 1; val = (w >> 8) & 0xFFFF
        who = {0: "입력1", 1: "입력2", 2: "출력"}[operand]
        return f"ADDR   {who} 주소 {'상위' if hi else '하위'}16 = {val}"
    if op == 0x82:
        return f"VLEN   벡터길이 = {(w >> 8) & 0xFFFF}"
    if op == 0x88:
        sel = (w >> 31) & 1; d1 = (w >> 8) & 0xFF; d2 = (w >> 16) & 0xFF
        return f"TILE   {'B(둘째)' if sel else 'A(첫째)'} 크기 = {d1}x{d2}"
    if op == 0x90:
        mat = (w >> 31) & 1; o = (w >> 30) & 1
        return f"LOAD   {'행렬' if mat else '벡터'}, {'입력2' if o else '입력1'} ← G-buffer"
    if op == 0x98:
        return f"SAVE   {'행렬' if (w >> 31) & 1 else '벡터'} → G-buffer (FP16 반올림)"
    if op == 0x42:
        return f"MATMUL 행렬곱 ({modes[mode]})"
    if op == 0x40:
        return f"M_ADD  행렬덧셈 ({modes[mode]})"
    names = {0x01: "VADD ", 0x02: "VSUB ", 0x0A: "VMUL ", 0x0B: "VDIV "}
    if op in names:
        imm = (w >> 8) & 0xFFFF
        extra = f", imm={imm}" if mode != 2 else ""
        tag = "  (복사: a+0)" if (op == 0x01 and mode == 0 and imm == 0) else ""
        return f"{names[op]} 원소별 ({modes[mode]}{extra}){tag}"
    if op == 0x0E:
        return "VSQRT  원소별 제곱근 (단항)"
    if op == 0x0F:
        return "VEXP   원소별 exp (단항)"
    return f"op=0x{op:02x} (기타)"


def print_program(asm, limit=None):
    """asm.words를 번호+hex+해석으로 출력. limit이 있으면 앞부분만."""
    n = len(asm.words)
    show = n if limit is None else min(limit, n)
    print(f"{'#':<4}{'hex':<13}의미")
    for i in range(show):
        print(f"{i:<4}0x{asm.words[i]:08x}  {disasm(asm.words[i])}")
    if show < n:
        print(f"... (총 {n}개 중 {show}개만 표시)")
