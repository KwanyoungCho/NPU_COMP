"""Regenerate editable SVG diagrams and measured charts for the manual/PPT.

Conceptual diagrams use only the standard library. Charts use matplotlib and
read the retained measurement JSON, never illustrative performance numbers.
"""
from html import escape
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
COLORS = {'host': '#edf0f5', 'dram': '#ffe1b8', 'sram': '#cceafb',
          'unit': '#e2d9ff', 'ir': '#d6f2e5', 'note': '#fff3c7'}


class Diagram:
    def __init__(self, title, subtitle):
        self.parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700">',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#4b5b70"/></marker></defs>',
            '<rect width="1200" height="700" fill="white"/>',
            '<g font-family="DejaVu Sans,Arial,sans-serif" fill="#142239">']
        self.text(45, 52, title, 30, weight='bold')
        self.text(45, 89, subtitle, 17)

    def text(self, x, y, value, size=19, anchor='start', weight='normal'):
        for i, line in enumerate(value.split('\n')):
            self.parts.append(f'<text x="{x}" y="{y+i*(size+10)}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(line)}</text>')

    def box(self, x, y, w, h, label, kind='ir', size=19):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{COLORS[kind]}" stroke="#66758b" stroke-width="1.5"/>')
        lines = len(label.split('\n'))
        self.text(x+w/2, y+h/2-(lines-1)*(size+10)/2+size*.35, label, size, 'middle')

    def arrow(self, x1, y1, x2, y2, label=None):
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="#4b5b70" stroke-width="2.5"/>')
        self.head(x1,y1,x2,y2)
        if label:
            if x1==x2:self.text(x1+14,(y1+y2)/2,label,15)
            else:self.text((x1+x2)/2,(y1+y2)/2-12,label,15,'middle')

    def head(self,x1,y1,x2,y2):
        distance=math.hypot(x2-x1,y2-y1)
        ux,uy=(x2-x1)/distance,(y2-y1)/distance
        a=(x2-ux*11-uy*5,y2-uy*11+ux*5)
        b=(x2-ux*11+uy*5,y2-uy*11-ux*5)
        self.parts.append(f'<polygon points="{x2},{y2} {a[0]},{a[1]} {b[0]},{b[1]}" fill="#4b5b70"/>')

    def path(self, points):
        self.parts.append(f'<polyline points="{points}" fill="none" stroke="#4b5b70" stroke-width="2.5"/>')
        last=[float(v) for p in points.split()[-2:] for v in p.split(',')]
        self.head(*last)

    def save(self, filename, footnote):
        self.text(45, 662, footnote, 16)
        (OUT/filename).write_text('\n'.join(self.parts+['</g></svg>'])+'\n')


def diagrams():
    d=Diagram('01  From a model to an NPU program', 'Current v09 | compile time, load time and inference time have different jobs')
    for x, label in [(45,'Model / shapes\nRelax graph'),(435,'Graph passes\nHost / device split'),(825,'Legalize + fusion\nSemantic TIR')]:
        d.box(x,155,330,95,label)
    d.arrow(375,202,435,202);d.arrow(765,202,825,202)
    for x,label in [(45,'Link + ISA\nProgram + memory plan'),(435,'SRAM / DRAM plan\nLifetime + retention'),(825,'Schedule search\nTiles / loops / cache')]:
        d.box(x,340,330,105,label,'sram')
    d.arrow(990,250,990,340);d.arrow(825,392,765,392);d.arrow(435,392,375,392)
    d.text(600,300,'Candidate scoring revisits schedule, memory and code generation',17,'middle')
    d.box(45,520,330,90,'Host parameter transform\nFill weight / input image','host')
    d.box(435,520,330,90,'Prefill / decode execution\nPatch input + append KV','dram')
    d.box(825,520,330,90,'Logits / token selection\nNext decode step','host')
    d.arrow(210,445,210,520);d.arrow(375,565,435,565);d.arrow(765,565,825,565)
    d.save('01_pipeline.svg','Conceptual flow. The graph is not compiled exactly once per stage.')

    d=Diagram('02  One Llama layer', 'Llama 3.2 3B widths | S tokens | 28 layers in the full model')
    d.box(45,145,250,80,'X [S,3072]','sram')
    d.box(380,145,330,80,'RMSNorm + Q / K / V','ir')
    d.box(795,145,360,80,'RoPE + attention + O','unit')
    d.arrow(295,185,380,185);d.arrow(710,185,795,185)
    d.box(795,305,360,80,'R = X + attention output','sram')
    d.arrow(975,225,975,305)
    d.path('170,225 170,345 795,345');d.text(405,325,'Residual X must survive',18)
    d.box(45,460,250,100,'RMSNorm(R)\n[S,3072]','ir')
    d.box(380,435,340,130,'SiLU(gate) * up\n[S,8192]\nthen down [S,3072]','unit')
    d.box(795,460,360,100,'Y = R + MLP output\n[S,3072]','sram')
    d.path('975,385 975,408 170,408 170,460')
    d.arrow(295,510,380,510);d.arrow(720,510,795,510)
    d.path('1155,345 1175,345 1175,510 1155,510')
    d.text(600,607,'Prefill S > 1; decode S = 1. Attention reads the accumulated KV context.',18,'middle')
    d.save('02_llama_layer.svg','Shapes describe the model. SRAM retention depends on lifetimes and available capacity.')

    d=Diagram('03  LegalizeOps keeps the Relax call graph', 'High-level operators become calls to concrete TIR computations')
    d.box(45,160,460,210,'Before\nRelax: matmul -> add -> SiLU\nTensor shapes + data dependencies','ir')
    d.box(695,135,460,190,'After: Relax\ncall_tir(matmul)\n-> call_tir(add) -> call_tir(silu)','ir')
    d.arrow(505,255,695,255,'LegalizeOps')
    d.box(695,440,460,150,'TIR PrimFuncs\nLoops / indices / reductions / casts\nStill before hardware scheduling','unit')
    d.arrow(925,325,925,440,'calls')
    d.box(45,450,460,140,'FuseOps groups the calls\nFuseTIR combines their TIR bodies','note')
    d.save('03_relax_tir.svg','One module can contain both Relax functions and TIR PrimFuncs.')

    d=Diagram('04  K accumulation and FP16 boundaries', 'X[64,128] @ B[128,64] = P[64,64] | two K tiles, one output tile')
    d.box(45,175,330,100,'X[:,0:64] @ B[0:64,:]\nM_MUL: accumulate = 0','unit')
    d.box(435,175,330,100,'X[:,64:128] @ B[64:128,:]\nM_MUL: accumulate = 1','unit',18)
    d.box(825,175,330,100,'SAVE\nP to SRAM as FP16','sram')
    d.arrow(375,225,435,225);d.arrow(765,225,825,225)
    d.box(45,345,720,90,'FP32 accumulator stays inside the matrix unit across K','unit')
    d.arrow(210,275,210,345);d.arrow(600,275,600,345)
    d.box(825,345,330,90,'Add / SiLU after all K\nFP16 stage boundaries','sram')
    d.arrow(990,275,990,345)
    d.box(150,505,900,80,'DMA panel size may cover many 64x64 tiles; it is a separate schedule choice.','note',18)
    d.save('04_matmul_accumulator.svg','Current automatic M_MUL path uses activation OFF. Fusion must preserve the numeric contract.')

    d=Diagram('05  Two kernels can share H in SRAM', 'Recorded 44 KiB example | X[64,64] @ W1[64,128] = H; H @ W2[128,64] = Y')
    origin=105;scale=1000/45056
    segments=[(0,1548,'Reserved','host'),(1548,8192,'X / Y','sram'),(9740,16384,'W1 / W2','dram'),(26124,16384,'H','unit'),(42508,2548,'','host')]
    for row,(y,title) in enumerate([(210,'Kernel 1: produce H'),(410,'Kernel 2: consume H')]):
        d.text(105,y-30,title,22,weight='bold')
        for start,size,label,kind in segments:
            label=label.replace('X / Y','X' if row==0 else 'Y').replace('W1 / W2','W1' if row==0 else 'W2')
            d.box(origin+start*scale,y,size*scale,85,label if size>5000 else '',kind)
        for address in [1548,9740,26124,42508]:
            d.text(origin+address*scale,y+116,str(address),15,'middle')
    d.arrow(866,326,866,403,'same H allocation')
    d.text(105,565,'H is written directly into its retained region. No H GSTORE / GLOAD is emitted.',20)
    d.text(105,603,'Y reuses dead X storage. W2 reuses dead W1 storage. Addresses are bytes.',18)
    d.save('05_sram_retention.svg','SRAM peak 42,508 bytes; capacity 45,056 bytes. The narrow first region includes scratch/constants.')

    d=Diagram('06  Query-blocked attention', 'Current fused attention path | scratch is reduced along Q; K context is still complete')
    d.box(45,170,275,220,'Q block\n[query_rows,D]\n\nNext block later','sram')
    d.box(435,170,330,100,'Full K context\n[context,D]','sram')
    d.box(435,355,330,110,'Score + softmax\n[query_rows,context]','unit')
    d.box(865,170,290,100,'Full V context\n[context,D]','sram')
    d.box(865,355,290,110,'Output block\n[query_rows,D]','sram')
    d.arrow(320,280,435,410,'QK^T');d.arrow(600,270,600,355)
    d.arrow(765,410,865,410,'PV');d.arrow(1010,270,1010,355)
    d.box(110,535,980,70,'Key-blocked online softmax (FlashAttention-style streaming) is not implemented.','note',19)
    d.save('06_attention_blocks.svg','This path has shape/scratch restrictions. Its existence does not mean full Llama always selects it.')

    d=Diagram('07  Decode capacity and the append slot', 'Example: 7 prompt tokens, capacity C=9, first decode position L=7')
    d.text(45,155,'Persistent DRAM cache, capacity 9',22,weight='bold')
    for i in range(9):d.box(45+i*122,185,110,90,f'{i}\n'+('valid' if i<7 else 'empty'),'dram' if i<7 else 'host',18)
    d.box(45,380,300,100,'Decode graph\nTemporary slot at index C','unit')
    d.box(445,380,300,100,'New K / V output\nSeparate DRAM output','dram')
    d.box(845,380,310,100,'NPU DMA append tail\nWrite cache slot L = 7','sram',18)
    d.arrow(345,430,445,430);d.arrow(745,430,845,430)
    d.path('1000,380 1000,320 954,320 954,280')
    d.text(45,563,'Mask: invalid capacity slots do not participate in softmax.',22)
    d.text(45,608,'After this step L=8. The static decode graph is reused while its capacity fits.',20)
    d.save('07_kv_decode.svg','Graph temporary index C is different from the persistent write position L. Host patches inputs/mask.')

    d=Diagram('08  Quantization is a separate validation track', 'Weights stored as INT8 do not imply that every calculation is INT8')
    d.box(45,175,310,100,'W8A16\nINT8 W + FP16 X','dram')
    d.box(445,175,310,100,'Dequantize W\nFP16 tile in SRAM','sram')
    d.box(845,175,310,100,'FP16 matrix multiply\nFP32 accumulation','unit',18)
    d.arrow(355,225,445,225);d.arrow(755,225,845,225)
    d.box(45,370,310,120,'W8A8\nQuantize X per group\nINT8 W + INT8 X','dram')
    d.box(445,370,310,120,'64-element INT8 dot\nINT32 group sum','unit')
    d.box(845,370,310,120,'Apply scales\nFP32 sum -> FP16','sram')
    d.arrow(355,430,445,430);d.arrow(755,430,845,430)
    d.box(95,550,1010,65,'Small kernel/model checks exist. End-to-end quantized generation is not validated.','note',19)
    d.save('08_quantization.svg','See the quantization appendix for scale layout, workspace constraints and historical test scope.')


def charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12,'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white'})
    records=json.loads((ROOT/'examples/memory/measurements.json').read_text())['records']
    labels=[f"{r['sram_capacity_bytes']//1024} KiB\n{'joint on' if r['joint_memory'] else 'joint off'}" for r in records]
    fig,axes=plt.subplots(1,2,figsize=(12,6.2))
    for ax,key,title in zip(axes,['dma_bytes','words'],['DMA traffic (bytes)','Program storage (32-bit words)']):
        values=[r['statistics'][key] for r in records]
        bars=ax.bar(labels,values,color=['#8294a8','#8294a8','#8294a8','#388ab6','#388ab6'])
        ax.bar_label(bars,labels=[f'{v:,}' for v in values],padding=4,fontsize=11)
        ax.set_ylim(0,max(values)*1.2);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('09  SRAM retention: measured small-example programs',fontsize=18)
    fig.text(.5,.02,'X[64,64] @ W1[64,128] @ W2[128,64] | fusion off | all outputs bit-exact | no hardware timing',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(OUT/'09_memory_measurements.svg');plt.close(fig)
    rows=json.loads((ROOT/'evidence/current/compile_comparison.json').read_text())['rows']
    by_label={r['label']:r for r in rows}
    groups=[('M=7',['instrumented_mlp7','bounded_mlp7']),('M=64',['unbounded_mlp64','bounded_mlp64'])]
    fig,axes=plt.subplots(1,2,figsize=(12,6.2))
    for ax,(name,keys) in zip(axes,groups):
        values=[by_label[k]['seconds'] for k in keys]
        bars=ax.bar(['Earlier configuration','Bounded search'],values,color=['#8294a8','#388ab6'])
        ax.bar_label(bars,labels=[f'{v:.3f} s' for v in values],padding=5)
        ax.set_ylim(0,100);ax.set_ylabel('Compilation seconds');ax.set_title(name+' | same program binary');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('10  Compile-time observations: actual-width Llama MLP',fontsize=18)
    fig.text(.5,.02,'Historical single samples; concurrent jobs affected timing. Not hardware execution latency or a repeated benchmark.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(OUT/'10_compile_observations.svg');plt.close(fig)


if __name__=='__main__':
    diagrams();charts()
    print('Wrote eight conceptual diagrams and two evidence-based charts.')
