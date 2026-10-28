# Archived research evidence

Contains actual baseline/modified simulation records, PIM traces/configurations/logs, comparisons, plots and reports. No Python environment, binaries or nested original checkout is included.

The research scripts expect a workspace containing sibling `pimba/`, `results/` and `figures/`. After cloning this branch into `pimba/`, copy this directory's `results/` and `figures/` into the parent workspace (without overwriting existing measurements), then follow EXPERIMENTS.md. Recreate the pinned original checkout as documented there.

The original baseline JSON files and their hashes are unchanged. Absolute paths in archived command logs describe the original execution environment; replay_pim.py rewrites the saved trace path for replay.
