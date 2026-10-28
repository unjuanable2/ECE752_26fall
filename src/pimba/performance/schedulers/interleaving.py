"""Dependency/event simulation with exclusive, static HBM channel partitions.

System-level estimate: upstream GPU analytical model + unchanged isolated PIM
command-timing runs. No same-bank GPU/PIM concurrency or dual row buffer assumed.
"""
from dataclasses import dataclass, replace
import copy
import heapq
import math

from ..models import create_model
from ..models.utils import models_configs
from ..layers import SU


@dataclass(frozen=True)
class InterleavingConfig:
    sub_batch_size: int
    pim_channels: int = 20
    sync_ns: float = 1000.0
    overlap: bool = True
    record_timeline: bool = True
    timeline_limit: int = 200000


@dataclass(frozen=True)
class Operation:
    name: str
    resource: str
    duration: float
    gpu_bytes: float = 0.0


def simulate_interleaved(system, model_name, batch_size, lin, lout, config):
    if not model_name.startswith('mamba2-') or system.pim is None or system.pim.hw != 'Pimba' or system.tp != 1:
        raise ValueError('Prototype supports single-GPU Pimba Mamba2 only')
    if not isinstance(config.sub_batch_size, int) or not 0 < config.sub_batch_size <= batch_size:
        raise ValueError('sub_batch_size must be a positive integer <= batch_size')
    if lout < 2 or not isinstance(lout, int):
        raise ValueError('lout must be >= 2 (at least one decode step)')
    total_channels = system.pim.num_channels
    if not isinstance(config.pim_channels, int) or not 0 < config.pim_channels < total_channels:
        raise ValueError('Both GPU and PIM require nonempty disjoint channel partitions')
    if not math.isfinite(config.sync_ns) or config.sync_ns < 0:
        raise ValueError('sync_ns must be finite and nonnegative')
    if config.timeline_limit < 0:
        raise ValueError('timeline_limit must be nonnegative')
    gpu_channels = total_channels - config.pim_channels
    gpu = replace(system.gpu, peak_memory_bandwidth=system.gpu.peak_memory_bandwidth * gpu_channels / total_channels,
                  _table_tiles={})
    pim = replace(system.pim, num_channels=config.pim_channels, _cache={})
    preset = models_configs['mamba2'][model_name.split('-', 1)[1]]
    decoders = preset['num_decoders']
    sizes = [min(config.sub_batch_size, batch_size-i) for i in range(0, batch_size, config.sub_batch_size)]
    templates = {}
    padding = {}
    # Build real smaller shapes; no duration is divided by the number of sub-batches.
    for size in sorted(set(sizes)):
        model = create_model(model_name, size, lin, 2, True, 1, pim.state_dbyte)
        ops = []
        for original in model:
            layer = copy.copy(original)
            layer.num_layers = 1
            if isinstance(layer, SU):
                # Upstream trace requires all bank lanes filled. Pad only dummy SU
                # state slots for an incomplete physical wave, charging full timing.
                lanes = config.pim_channels * 2 * 4 * 4
                rows_per_request = layer.m * math.ceil(layer.n / (32 // layer.dbyte))
                quantum = lanes // math.gcd(lanes, rows_per_request)
                padded = math.ceil(size / quantum) * quantum
                layer.num_op = padded
                padding[size] = padded
                time, _, _ = pim.simulate(layer)
                # Register input/result traffic on PIM-side is already in its trace.
                # Charge extra GPU-side staging; conservative one head_dim result
                # per request/head. Not extra PIM-side register traffic.
                input_bytes = size * layer.m * (3 * layer.k + layer.n) * layer.dbyte
                output_bytes = size * layer.m * layer.n * 2
                bw = gpu.peak_memory_bandwidth * gpu.max_memory_util
                ops.append(Operation('handoff_in', 'GPU', input_bytes / bw + config.sync_ns * 1e-9, input_bytes))
                ops.append(Operation(layer.name, 'PIM', time))
                ops.append(Operation('handoff_out', 'GPU', output_bytes / bw + config.sync_ns * 1e-9, output_bytes))
            else:
                time, off_energy, _ = gpu.simulate(layer)
                ops.append(Operation(layer.name, 'GPU', time, off_energy / gpu.energy['mem']))
        if not all(math.isfinite(op.duration) and op.duration > 0 for op in ops):
            raise ValueError('Nonpositive or nonfinite operation timing')
        templates[size] = ops
    result = schedule(templates, sizes, decoders, lout-1, config, gpu, total_channels, padding)
    result['gpu_channel_range'] = [0, gpu_channels]
    result['pim_channel_range'] = [gpu_channels, total_channels]
    result['pim_logical_state_bytes_including_dummy_slots'] = (
        sum(padding[size] for size in sizes) * preset['num_heads'] * preset['head_dim']
        * preset['state_dim'] * decoders * pim.state_dbyte
    )
    return result


def schedule(templates, sizes, decoders, tokens, config, gpu, total_channels, padding):
    """Lazy chains: only the next dependent operation of each sub-batch is ready.

    Each chain expands token -> decoder -> operator, enforcing all state recurrence
    and inter-decoder edges. FIFO ready queues; completion events release successors.
    """
    ready = {'GPU': [], 'PIM': []}
    busy = {'GPU': False, 'PIM': False}
    events = []
    timeline = []
    sequence = 0
    completed = [0] * len(sizes)
    predecessor_finish = [0.0] * len(sizes)
    completion = [None] * len(sizes)
    versions = [[0] * decoders for _ in sizes]
    service = {'GPU': 0.0, 'PIM': 0.0}
    bytes_gpu = 0.0
    overlap = 0.0
    now = 0.0
    num_events = 0
    nops = len(next(iter(templates.values())))
    if any(len(v) != nops for v in templates.values()):
        raise ValueError('All sub-batches must have identical operation schemas')
    target = tokens * decoders * nops

    def push(sb, index, time):
        nonlocal sequence
        op = templates[sizes[sb]][index % nops]
        heapq.heappush(ready[op.resource], (time, sequence, sb, index))
        sequence += 1

    for sb in range(len(sizes)):
        push(sb, 0, 0.0)
    while any(x is None for x in completion):
        for resource in ('GPU', 'PIM'):
            other = 'PIM' if resource == 'GPU' else 'GPU'
            if busy[resource] or not ready[resource] or (not config.overlap and busy[other]):
                continue
            _, _, sb, index = heapq.heappop(ready[resource])
            op = templates[sizes[sb]][index % nops]
            if index != completed[sb] or now + 1e-14 < predecessor_finish[sb]:
                raise AssertionError('Dependency or duplicate dispatch violation')
            token, rem = divmod(index, decoders * nops)
            decoder, stage = divmod(rem, nops)
            if op.resource == 'PIM' and versions[sb][decoder] != token:
                raise AssertionError('State update is out of token order')
            end = now + op.duration
            heapq.heappush(events, (end, sb, index, resource))
            busy[resource] = True
            service[resource] += op.duration
            bytes_gpu += op.gpu_bytes
            if config.record_timeline and len(timeline) < config.timeline_limit:
                timeline.append(dict(sub_batch=sb, size=sizes[sb], token=token, decoder=decoder,
                                     op_index=stage, op=op.name, resource=resource,
                                     start_s=now, end_s=end, predecessor_end_s=predecessor_finish[sb],
                                     gpu_bytes=op.gpu_bytes))
        if not events:
            raise AssertionError('Deadlock')
        next_time = events[0][0]
        if busy['GPU'] and busy['PIM']:
            overlap += next_time - now
        now = next_time
        # Complete all simultaneous operations before dispatching successors.
        while events and events[0][0] <= now:
            end, sb, index, resource = heapq.heappop(events)
            busy[resource] = False
            completed[sb] += 1
            num_events += 1
            predecessor_finish[sb] = end
            if resource == 'PIM':
                token, rem = divmod(index, decoders*nops)
                decoder = rem // nops
                versions[sb][decoder] += 1
            if completed[sb] == target:
                completion[sb] = end
            else:
                push(sb, index+1, end)
    if any(n != target for n in completed) or any(v != tokens for vv in versions for v in vv):
        raise AssertionError('Incomplete operations or state versions')
    if not math.isclose(sum(service.values())-overlap, now, rel_tol=1e-9, abs_tol=1e-12):
        raise AssertionError('Busy/overlap/makespan accounting mismatch')
    return dict(makespan_s=now, throughput_tokens_s=sum(sizes)*tokens/now,
                decode_batch_latency_ms=1000*now, tpot_ms=1000*now/tokens,
                gpu_busy_s=service['GPU'], pim_busy_s=service['PIM'],
                gpu_idle_s=max(0.0,now-service['GPU']), pim_idle_s=max(0.0,now-service['PIM']),
                gpu_utilization=service['GPU']/now, pim_utilization=service['PIM']/now,
                overlap_s=overlap, overlap_fraction=overlap/now,
                gpu_hbm_bytes=bytes_gpu,
                gpu_hbm_bandwidth_utilization=bytes_gpu/(now*gpu.peak_memory_bandwidth),
                expected_operations=len(sizes)*target, completed_operations=num_events,
                sub_batches=[dict(id=i, request_start=sum(sizes[:i]), request_end=sum(sizes[:i+1]),
                                  size=s, completion_s=completion[i], padded_pim_size=padding[s]) for i,s in enumerate(sizes)],
                request_mean_completion_ms=1000*sum(s*t for s,t in zip(sizes,completion))/sum(sizes),
                state_versions=versions, timeline=timeline,
                timeline_truncated=config.record_timeline and num_events>config.timeline_limit,
                gpu_channels=total_channels-config.pim_channels, pim_channels=config.pim_channels,
                total_channels=total_channels, sync_ns=config.sync_ns,
                model_scope='static-channel-partition system-level estimate; isolated PIM command timing',
                profiles={str(size):[op.__dict__ for op in ops] for size,ops in templates.items()})
