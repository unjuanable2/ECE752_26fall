import json
import math
from pathlib import Path
import unittest

from pimba import Performance
from pimba.performance.schedulers import InterleavingConfig, simulate_interleaved
from pimba.performance.systems import create_system

ROOT=Path(__file__).resolve().parents[2]

class InterleavingTests(unittest.TestCase):
    def run_model(self,batch=32,size=16,lout=3,**kwargs):
        return Performance('mamba2-2.7b','Pimba',1,batch,'A100',2048,lout,
                           sub_batch_interleaving=True,sub_batch_size=size,**kwargs).run()['timing']

    def test_default_off_exact_baseline(self):
        for batch in (32,64,128):
            expected=json.loads((ROOT/f'results/run-20261028/baseline-b{batch}.json').read_text())
            actual=Performance('mamba2-2.7b','Pimba',1,batch,'A100',2048,17).run()
            self.assertEqual(expected['res'],actual['res'])

    def check_timeline(self,r,lout):
        rows=r['timeline']; self.assertFalse(r['timeline_truncated'])
        ids={(x['sub_batch'],x['token'],x['decoder'],x['op_index']) for x in rows}
        self.assertEqual(len(ids),len(rows))
        self.assertEqual(len(rows),r['expected_operations'])
        self.assertEqual(r['completed_operations'],r['expected_operations'])
        for x in rows:
            self.assertGreater(x['end_s'],x['start_s'])
            self.assertGreaterEqual(x['start_s']+1e-14,x['predecessor_end_s'])
        for resource in ('GPU','PIM'):
            seq=sorted((x for x in rows if x['resource']==resource),key=lambda x:x['start_s'])
            for a,b in zip(seq,seq[1:]):
                self.assertLessEqual(a['end_s'],b['start_s']+1e-14)
        for sb in r['sub_batches']:
            seq=sorted((x for x in rows if x['sub_batch']==sb['id']),key=lambda x:(x['token'],x['decoder'],x['op_index']))
            for a,b in zip(seq,seq[1:]): self.assertLessEqual(a['end_s'],b['start_s']+1e-14)
            self.assertEqual(seq[-1]['end_s'],sb['completion_s'])
        self.assertTrue(all(v==lout-1 for vs in r['state_versions'] for v in vs))
        # Reconstruct interval intersection independently from scheduler counters.
        endpoints=[]
        for x in rows:
            endpoints.extend(((x['start_s'],1,x['resource']),(x['end_s'],-1,x['resource'])))
        endpoints.sort(); active={'GPU':0,'PIM':0}; last=0; overlap=0
        for t,change,resource in endpoints:
            if all(active.values()): overlap+=t-last
            active[resource]+=change; last=t
        self.assertAlmostEqual(overlap,r['overlap_s'],places=10)
        self.assertEqual(r['gpu_channels']+r['pim_channels'],40)
        self.assertLessEqual(r['gpu_hbm_bandwidth_utilization'],1)

    def test_real_overlap_and_dependencies(self):
        r=self.run_model();self.check_timeline(r,3)
        self.assertGreater(r['overlap_fraction'],0)
        serial=self.run_model(execution_overlap=False)
        self.assertEqual(serial['overlap_s'],0)
        self.assertLess(r['makespan_s'],serial['makespan_s'])
        self.assertAlmostEqual(serial['makespan_s'],serial['gpu_busy_s']+serial['pim_busy_s'])

    def test_tail_and_state_recurrence(self):
        r=self.run_model(batch=35,size=16,lout=4);self.check_timeline(r,4)
        self.assertEqual([x['size'] for x in r['sub_batches']],[16,16,3])
        self.assertEqual(r['sub_batches'][-1]['padded_pim_size'],4)
        self.assertEqual(r['sub_batches'][-1]['request_end'],35)

    def test_single_sub_batch_no_overlap(self):
        r=self.run_model(batch=32,size=32);self.check_timeline(r,3)
        self.assertEqual(r['overlap_s'],0)
        self.assertAlmostEqual(r['makespan_s'],r['gpu_busy_s']+r['pim_busy_s'])

    def test_reject_invalid(self):
        for size in (0,-1,33):
            with self.assertRaises(ValueError):self.run_model(size=size)
        with self.assertRaises(ValueError):self.run_model(lout=1)
        for channels in (0,40,41):
            with self.assertRaises(ValueError):self.run_model(pim_channels=channels)
        for sync in (-1,float('nan')):
            with self.assertRaises(ValueError):self.run_model(sync_ns=sync)
        system=create_system('Pimba','A100',2)
        with self.assertRaises(ValueError):
            simulate_interleaved(system,'mamba2-2.7b',32,2048,2,InterleavingConfig(16))

    def test_sync_sensitivity(self):
        a=self.run_model(sync_ns=0); b=self.run_model(sync_ns=5000)
        self.assertGreater(b['makespan_s'],a['makespan_s'])

if __name__=='__main__': unittest.main(verbosity=2)
