"""Run unchanged upstream performance source without importing accuracy packages."""
import sys, types, pathlib, json, time, hashlib, subprocess, shutil, os
from datetime import datetime, timezone
root=pathlib.Path(os.environ.get('PIMBA_ORIGINAL_ROOT', pathlib.Path(__file__).resolve().parents[2])).resolve()
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() != '0a902eeafddbfddaa34723ed74a761e4453b870d':
    raise RuntimeError('Baseline requires the pinned original upstream commit')
if subprocess.check_output(['git','status','--porcelain','--','src','csrc'],cwd=root,text=True).strip():
    raise RuntimeError('Baseline performance/accuracy/C++ source must be pristine')
b=int(sys.argv[1]); out=pathlib.Path(sys.argv[2]).resolve(); lout=int(sys.argv[3]) if len(sys.argv)>3 else 17
if out.exists(): raise FileExistsError(out)
out.parent.mkdir(parents=True,exist_ok=True)
pkg=types.ModuleType('pimba'); pkg.__path__=[str(root/'src/pimba')];sys.modules['pimba']=pkg
from pimba.performance.exp import Performance
from ruamel.yaml import YAML
yaml=YAML(typ='safe'); orig_run=subprocess.run; calls=[]
def record_run(*args,**kwargs):
    cmd=args[0] if args else kwargs.get('args')
    if isinstance(cmd,str) and 'PimbaSim -f' in cmd:
        import shlex
        parts=shlex.split(cmd); config=pathlib.Path(parts[parts.index('-f')+1]); target=pathlib.Path(parts[parts.index('-o')+1])
        name=out.with_suffix(''); audit=name.parent/(name.name+'-pim'); audit.mkdir(exist_ok=True)
        conf=yaml.load(config); trace=pathlib.Path(conf['Frontend']['path'])
        shutil.copyfile(trace,audit/'trace.txt'); shutil.copyfile(config,audit/'config.yaml')
        start=time.perf_counter(); result=orig_run(*args,**kwargs)
        (audit/'stdout.log').write_bytes(result.stdout); (audit/'stderr.log').write_bytes(result.stderr)
        if target.exists(): shutil.copyfile(target,audit/'stats.yaml')
        calls.append({'command':cmd,'wall_seconds':time.perf_counter()-start,'returncode':result.returncode})
        return result
    return orig_run(*args,**kwargs)
subprocess.run=record_run
start=time.perf_counter();exp=Performance('mamba2-2.7b','Pimba',1,b,'A100',2048,lout);r=exp.run()
t=sum(r['res']['time'].values())
r.update(wall_seconds=time.perf_counter()-start,decode_latency_ms=t*1000,tpot_ms=t*1000/(lout-1),throughput=b*(lout-1)/t,lin=2048,lout=lout,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),binary_sha256=hashlib.sha256((root/'build/PimbaSim').read_bytes()).hexdigest(),pim_calls=calls,source_diff=subprocess.check_output(['git','diff','--','src','csrc'],cwd=root,text=True),utc=datetime.now(timezone.utc).isoformat())
out.write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
