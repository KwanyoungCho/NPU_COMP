"""Build provenance for exported programs and regression evidence."""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def describe():
    import tvm
    from .native import library_path
    from .v09_contract import CONTRACT_HASH
    root=Path(__file__).resolve().parents[2]
    files=sorted([*root.joinpath('d_compiler/npu_compiler').rglob('*.py'),
                  *root.joinpath('d_compiler/npu_codegen').glob('*.cc'),
                  *root.joinpath('d_compiler/npu_codegen').glob('*.h')])
    sources={str(path.relative_to(root)):sha256(path) for path in files}
    digest=hashlib.sha256()
    for name,value in sources.items():digest.update((name+'\0'+value+'\n').encode())
    packages={}
    for name in ('numpy','torch','transformers','safetensors','cmake','ninja'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    info=tvm.support.libinfo()
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    native=library_path()
    return dict(python=platform.python_version(),packages=packages,tvm_version=tvm.__version__,
        tvm_commit=info.get('GIT_COMMIT_HASH'),llvm_version=info.get('LLVM_VERSION'),
        compiler_head=revision.stdout.strip() if revision.returncode==0 else None,
        compiler_source_sha256=digest.hexdigest(),sources=sources,
        contract_sha256=CONTRACT_HASH,native_path=str(native),
        native_sha256=sha256(native) if native.exists() else None,
        cmodel_source_sha256=sha256(root/'_poc/mysim_v09.cpp'))
