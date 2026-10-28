"""One experiment per process, with CPU-only performance imports and raw PIM audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'src'))
parser=argparse.ArgumentParser()
parser.add_argument('--batch',type=int,required=True)
parser.add_argument('--sub-batch',type=int)
parser.add_argument('--lout',type=int,default=17)
parser.add_argument('--pim-channels',type=int,default=20)
parser.add_argument('--sync-ns',type=float,default=1000)
parser.add_argument('--serialize',action='store_true')
parser.add_argument('--no-timeline',action='store_true')
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
a.output.parent.mkdir(parents=True,exist_ok=True)
os.environ['PIMBA_AUDIT_DIR']=str(a.output.resolve().with_suffix(''))+'-pim'
from pimba import Performance
start=time.perf_counter()
exp=Performance('mamba2-2.7b','Pimba',1,a.batch,'A100',2048,a.lout,
                sub_batch_interleaving=a.sub_batch is not None,sub_batch_size=a.sub_batch or 32,
                pim_channels=a.pim_channels,sync_ns=a.sync_ns,
                execution_overlap=not a.serialize,record_timeline=not a.no_timeline)
r=exp.run()
r.update(wall_seconds=time.perf_counter()-start,utc=datetime.now(timezone.utc).isoformat(),
         command=sys.argv,python=sys.version,binary_sha256=hashlib.sha256((root/'build/PimbaSim').read_bytes()).hexdigest(),
         lin=2048,lout=a.lout)
if 'timing' not in r:
    t=sum(r['res']['time'].values()); r.update(decode_latency_ms=t*1000,tpot_ms=t*1000/(a.lout-1),throughput=a.batch*(a.lout-1)/t)
a.output.write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k not in ('timing',)},indent=2))
if 'timing' in r: print(json.dumps({k:v for k,v in r['timing'].items() if k not in ('timeline','profiles','state_versions')},indent=2))
