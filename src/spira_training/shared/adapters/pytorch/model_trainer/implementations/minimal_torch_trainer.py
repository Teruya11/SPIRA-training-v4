import torch

from src.spira_training.shared.core.models.base_model import BaseModel
from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.ports.model_trainer import ModelTrainer


class MinimalTorchTrainer(ModelTrainer):
    def __init__(self, model: BaseModel, optimizer: torch.optim.Optimizer, loss_fn):
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn

    def train_model(
        self, train_dataset: Dataset, test_dataset: Dataset, epochs: int
    ) -> BaseModel:
        self._model.train()

        for _ in range(epochs):
            for feature, label in zip(train_dataset.features, train_dataset.labels):
                x = torch.as_tensor(feature.wav.tensor, dtype=torch.float32)
                y = self._to_target(label, x)

                self._optimizer.zero_grad()
                prediction = self._model.predict(x)
                loss = self._loss_fn(prediction, y)
                loss.backward()
                self._optimizer.step()

        return self._model

    def _to_target(self, label, x):
        if hasattr(label, "value"):
            label_value = label.value
        else:
            label_value = label

        target = torch.as_tensor(float(label_value), dtype=torch.float32)
        model_output = self._model.predict(x) if hasattr(self._model, "predict") else self._model(x)
        if model_output.ndim == 0:
            return target
        return target.view(1)
