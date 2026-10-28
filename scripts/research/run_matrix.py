"""Resumable matrix: each case is a fresh interpreter, never overwrites results."""
import subprocess
import sys
import argparse
from pathlib import Path
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--lout',type=int,default=17)
parser.add_argument('--output-dir',type=Path,default=root.parent/'results/run-20261028')
a=parser.parse_args()
out=a.output_dir.resolve()
out.mkdir(parents=True,exist_ok=True)
for batch in (32,64,128):
    for size in (8,16,32,64):
        if size>batch:continue
        name=f'modified-b{batch}-s{size}'
        target=out/(name+'.json')
        if target.exists():continue
        cmd=[sys.executable,str(root/'scripts/research/run_case.py'),'--batch',str(batch),'--sub-batch',str(size),'--output',str(target),'--lout',str(a.lout)]
        if a.lout>17 or (batch,size)!=(128,32):cmd+=['--no-timeline']
        print('RUN',name,flush=True)
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
# Same partition and profiling, only overlap disabled: isolate scheduler benefit.
for name,extra in [('control-serialized-b128-s32',['--serialize']),
                   ('sensitivity-sync0-b128-s32',['--sync-ns','0']),
                   ('sensitivity-sync5000-b128-s32',['--sync-ns','5000']),
                   ('sensitivity-pim16-b128-s32',['--pim-channels','16']),
                   ('sensitivity-pim24-b128-s32',['--pim-channels','24'])]:
    target=out/(name+'.json')
    if target.exists():continue
    cmd=[sys.executable,str(root/'scripts/research/run_case.py'),'--batch','128','--sub-batch','32','--output',str(target),'--no-timeline','--lout',str(a.lout),*extra]
    print('RUN',name,flush=True)
    with (out/(name+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
