"""Generate only from archived simulator executions; refuse incomplete matrices."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2];workspace=root.parent
p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,default=workspace/'results/run-20261028/full2048');p.add_argument('--figures',type=Path,default=workspace/'figures');a=p.parse_args()
a.figures.mkdir(exist_ok=True,parents=True)
rows=[]
for b in (32,64,128):
    base=json.loads((a.input_dir/f'baseline-b{b}.json').read_text())
    for s in (8,16,32,64):
        if s>b:continue
        raw=json.loads((a.input_dir/f'modified-b{b}-s{s}.json').read_text());t=raw['timing']
        rows.append(dict(model='mamba2-2.7b',batch=b,sub_batch=s,lin=raw['lin'],lout=raw['lout'],
            baseline_throughput=base['throughput'],modified_throughput=t['throughput_tokens_s'],
            speedup=t['throughput_tokens_s']/base['throughput'],
            baseline_decode_latency_ms=base['decode_latency_ms'],modified_decode_latency_ms=t['decode_batch_latency_ms'],
            latency_reduction_pct=100*(base['decode_latency_ms']-t['decode_batch_latency_ms'])/base['decode_latency_ms'],
            baseline_tpot_ms=base['tpot_ms'],modified_tpot_ms=t['tpot_ms'],
            gpu_utilization=t['gpu_utilization'],pim_utilization=t['pim_utilization'],overlap_pct=100*t['overlap_fraction'],
            gpu_idle_ms=t['gpu_idle_s']*1000,pim_idle_ms=t['pim_idle_s']*1000,
            gpu_hbm_bandwidth_utilization=t['gpu_hbm_bandwidth_utilization'],
            baseline_wall_seconds=base['wall_seconds'],modified_wall_seconds=raw['wall_seconds'],
            expected_operations=t['expected_operations'],completed_operations=t['completed_operations']))
with (a.input_dir/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(a.input_dir/'comparison.json').write_text(json.dumps(rows,indent=2))
labels=[f"B{r['batch']}/S{r['sub_batch']}" for r in rows]
plt.rcParams.update({'font.size':10})
for name,key1,key2,ylabel in [('throughput_comparison','baseline_throughput','modified_throughput','Decode throughput (tokens/s)'),('latency_comparison','baseline_decode_latency_ms','modified_decode_latency_ms','Decode batch completion latency (ms)')]:
    fig,ax=plt.subplots(figsize=(12,4));x=list(range(len(rows)))
    ax.bar([i-.18 for i in x],[r[key1] for r in rows],.36,label='Original Pimba')
    ax.bar([i+.18 for i in x],[r[key2] for r in rows],.36,label='Interleaving: static channel partition estimate')
    ax.set_xticks(x,labels,rotation=45);ax.set_ylabel(ylabel);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(a.figures/(name+'.png'),dpi=180);fig.savefig(a.figures/(name+'.pdf'));plt.close(fig)
for name,key,basekey,ylabel in [('sub_batch_vs_throughput','modified_throughput','baseline_throughput','Decode throughput (tokens/s)'),('sub_batch_vs_latency','modified_decode_latency_ms','baseline_decode_latency_ms','Decode batch completion latency (ms)')]:
    fig,ax=plt.subplots(figsize=(7,4))
    for b,color in zip((32,64,128),('tab:blue','tab:orange','tab:green')):
        subset=[r for r in rows if r['batch']==b]
        ax.plot([r['sub_batch'] for r in subset],[r[key] for r in subset],'-o',color=color,label=f'Interleaving B={b}')
        ax.axhline(subset[0][basekey],color=color,linestyle='--',alpha=.6,label=f'Original B={b}')
    ax.set_xticks([8,16,32,64]);ax.set_xlabel('Sub-batch size');ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(a.figures/(name+'.png'),dpi=180);fig.savefig(a.figures/(name+'.pdf'));plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4));x=list(range(len(rows)))
ax.bar([i-.2 for i in x],[100*r['gpu_utilization'] for r in rows],.4,label='GPU operation occupancy')
ax.bar([i+.2 for i in x],[100*r['pim_utilization'] for r in rows],.4,label='PIM operation occupancy')
ax.set_xticks(x,labels,rotation=45);ax.set_ylabel('Modeled resource occupancy (%)');ax.legend();fig.tight_layout();fig.savefig(a.figures/'utilization.png',dpi=180);plt.close(fig)
# Timeline from the separate 17-output validation run; never pretend it is full2048.
source=workspace/'results/run-20261028/modified-b128-s32.json'
r=json.loads(source.read_text());timeline=r['timing']['timeline']
fig,ax=plt.subplots(figsize=(12,3));colors=plt.get_cmap('tab10');limit=.003
for event in timeline:
    if event['start_s']>=limit:continue
    y=1 if event['resource']=='GPU' else 0
    ax.broken_barh([(event['start_s']*1000,(event['end_s']-event['start_s'])*1000)],(y-.3,.6),facecolors=colors(event['sub_batch']))
ax.set_yticks([0,1],['PIM [channels 20..39]','GPU [channels 0..19]']);ax.set_xlim(0,limit*1000);ax.set_xlabel('Simulated time (ms)');ax.set_title('B=128, S=32: first 3ms of validation run (lout=17), colors=sub-batches');fig.tight_layout();fig.savefig(a.figures/'execution_timeline.png',dpi=200);fig.savefig(a.figures/'execution_timeline.pdf');plt.close(fig)
header='| Model | Batch | Sub-batch | Original tokens/s | Modified tokens/s | Speedup | Original latency ms | Modified latency ms | Latency reduction |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for r in rows:
    header+=f"| {r['model']} | {r['batch']} | {r['sub_batch']} | {r['baseline_throughput']:.2f} | {r['modified_throughput']:.2f} | {r['speedup']:.3f}× | {r['baseline_decode_latency_ms']:.2f} | {r['modified_decode_latency_ms']:.2f} | {r['latency_reduction_pct']:.2f}% |\n"
(a.input_dir/'comparison.md').write_text(header)
print(header)
