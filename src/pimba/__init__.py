"""Keep accuracy CUDA packages optional for CPU-only performance simulation."""
import importlib

_EXPORTS = {
    'Performance': ('.performance.exp', 'Performance'),
    'create_performance_exps': ('.performance.exps', 'create_performance_exps'),
    'get_root_path': ('.performance.utils.file', 'get_root_path'),
    'Accuracy': ('.accuracy.exp', 'Accuracy'),
    'create_accuracy_exps': ('.accuracy.exps.utils', 'create_accuracy_exps'),
    'draw': ('.draw.exps', 'draw'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    if name in ('Accuracy', 'create_accuracy_exps'):
        # Preserve upstream deterministic accuracy behavior when actually requested.
        import triton
        def _dummy_tuner(**kwargs):
            return lambda fn: fn
        triton.autotune = _dummy_tuner
        import logging
        logging.disable()
    module, attr = _EXPORTS[name]
    value = getattr(importlib.import_module(module, __name__), attr)
    globals()[name] = value
    return value
