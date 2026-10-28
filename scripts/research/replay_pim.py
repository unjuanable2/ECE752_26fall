"""Replay one archived PIM trace and verify all exported counters exactly."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import json
from ruamel.yaml import YAML
p=argparse.ArgumentParser();p.add_argument('audit_dir',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[2];audit=a.audit_dir.resolve();y=YAML(typ='safe')
config=y.load(audit/'config.yaml');config['Frontend']['path']=str(audit/'trace.txt')
with tempfile.TemporaryDirectory() as temp:
    c=Path(temp)/'config.yaml';out=Path(temp)/'stats.yaml';y.dump(config,c)
    run=subprocess.run([str(root/'build/PimbaSim'),'-f',str(c),'-o',str(out)],capture_output=True,text=True,check=True)
    expected=y.load(audit/'stats.yaml');actual=y.load(out)
    if expected!=actual:raise AssertionError((expected,actual))
    print(json.dumps({'audit_dir':str(audit),'expected':expected,'actual':actual,'matched':True},indent=2))
