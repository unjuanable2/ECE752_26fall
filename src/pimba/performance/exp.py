from contextlib import contextmanager

from rich import print, tree

from .models import create_model
from .systems import create_system


class Performance:
    def __init__(
        self,
        model: str,
        system: str,
        num_gpus: int,
        batch_size: int,
        gpu: str,
        lin: int,
        lout: int,
        sub_batch_interleaving: bool = False,
        sub_batch_size: int = 32,
        pim_channels: int = 20,
        sync_ns: float = 1000.0,
        execution_overlap: bool = True,
        record_timeline: bool = True,
    ):
        config = tree.Tree(
            "Config",
            style="bold blue",
        )

        # system
        self.system_name = system
        self.gpu = gpu
        self.num_gpus = num_gpus
        self.system = create_system(system, gpu, num_gpus)

        system_config = tree.Tree("System")
        system_config.add(f"system: {system}", style="white")
        system_config.add(f"gpu: {gpu}", style="white")
        system_config.add(f"num_gpus: {num_gpus}", style="white")
        config.add(system_config, style="yellow")

        # model
        if self.system.pim is not None:
            state_dbyte = self.system.pim.state_dbyte
        else:
            state_dbyte = self.system.gpu.state_dbyte

        self.model_name = model
        self.model = create_model(
            model,
            batch_size,
            lin,
            lout,
            self.system.hetero_system,
            self.system.tp,
            state_dbyte,
        )

        model_config = tree.Tree("Model")
        model_config.add(f"model: {model}", style="white")
        config.add(model_config, style="yellow")

        # workload
        self.batch_size = batch_size
        self.lin, self.lout = lin, lout
        self.sub_batch_interleaving = sub_batch_interleaving
        self.sub_batch_size = sub_batch_size
        self.pim_channels = pim_channels
        self.sync_ns = sync_ns
        self.execution_overlap = execution_overlap
        self.record_timeline = record_timeline

        workload_config = tree.Tree("Workload")
        workload_config.add(f"batch: {batch_size}", style="white")
        workload_config.add(f"in: {lin}", style="white")
        workload_config.add(f"out: {lout}", style="white")
        config.add(workload_config, style="yellow")

        # config
        self.config = config

    def print_config(self):
        print(self.config)

    def run(self):
        if self.sub_batch_interleaving:
            from ..performance.schedulers import InterleavingConfig, simulate_interleaved

            timing = simulate_interleaved(
                self.system, self.model_name, self.batch_size, self.lin, self.lout,
                InterleavingConfig(self.sub_batch_size, self.pim_channels, self.sync_ns,
                                   self.execution_overlap, self.record_timeline),
            )
            return {"exp": {"model": self.model_name, "batch_size": self.batch_size,
                            "system_name": self.system_name, "num_gpus": self.num_gpus,
                            "gpu": self.gpu, "lin": self.lin, "lout": self.lout,
                            "sub_batch_interleaving": True, "sub_batch_size": self.sub_batch_size},
                    "timing": timing}
        res = self.system.simulate(self.model)
        res = (
            res.groupby(["phase", "category"])
            .agg({"time": "sum", "power_1": "sum", "power_2": "sum"})
            .loc["gen"]
            .to_dict()
        )

        return {
            "exp": {
                "model": self.model_name,
                "batch_size": self.batch_size,
                "system_name": self.system_name,
                "num_gpus": self.num_gpus,
                "gpu": self.gpu,
            },
            "res": res,
        }

    @classmethod
    @contextmanager
    def setup(
        cls,
        model: str,
        system: str,
        num_gpus: int,
        batch_size: int,
        gpu: str,
        lin: int = 2048,
        lout: int = 2048,
        sub_batch_interleaving: bool = False,
        sub_batch_size: int = 32,
        pim_channels: int = 20,
        sync_ns: float = 1000.0,
        execution_overlap: bool = True,
        record_timeline: bool = True,
    ):
        yield Performance(model, system, num_gpus, batch_size, gpu, lin, lout,
                          sub_batch_interleaving, sub_batch_size, pim_channels,
                          sync_ns, execution_overlap, record_timeline)
