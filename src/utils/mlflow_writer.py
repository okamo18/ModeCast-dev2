from typing import Any, Optional

import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.figure import Figure
from mlflow import pytorch, set_tracking_uri
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig


class MlflowWriter:
    def __init__(self, experiment_name: str, out_dir: str = "", **kwargs: dict):
        set_tracking_uri(out_dir + "mlruns/")
        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except Exception:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

        self.experiment = self.client.get_experiment(self.experiment_id)
        print("New experiment started")
        print(f"Name: {self.experiment.name}")
        print(f"Experiment_id: {self.experiment.experiment_id}")
        print(f"Artifact Location: {self.experiment.artifact_location}")

    def log_params_from_omegaconf_dict(self, params: DictConfig) -> None:
        for param_name, element in params.items():
            self._explore_recursive(str(param_name), element)

    def _explore_recursive(self, parent_name: str, element: dict) -> None:
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f"{parent_name}{k}", v)
                else:
                    self.client.log_param(self.run_id, f"{parent_name}.{k}", v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f"{parent_name}.{i}", v)
        else:
            self.client.log_param(self.run_id, f"{parent_name}", element)

    def log_torch_model(self, model: nn.Module) -> None:
        pytorch.log_model(model, "models")

    def log_param(self, key: str, value: float) -> None:
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key: str, value: float) -> None:
        self.client.log_metric(self.run_id, key, value)

    def log_metric_step(self, key: str, value: float, step: Optional[int] = None) -> None:
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, local_path: str) -> None:
        self.client.log_artifact(self.run_id, local_path)

    def log_dict(self, dictionary: dict[str, Any], file: str) -> None:
        self.client.log_dict(self.run_id, dictionary, file)

    def log_figure(self, figure: Figure, file: str) -> None:
        self.client.log_figure(self.run_id, figure, file)

    def set_terminated(self) -> None:
        self.client.set_terminated(self.run_id)
        plt.close()
