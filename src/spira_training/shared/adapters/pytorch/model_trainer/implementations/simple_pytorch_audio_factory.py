from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)
from src.spira_training.shared.core.models.audio import Audio
import torch
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_tensor_factory import (
    PytorchTensorFactory,
)


class SimplePytorchTensorFactory(PytorchTensorFactory):
    def create_tensor_from_audio(self, audio: Audio) -> PytorchTensor:
        tensor = audio.wav.tensor.detach().clone().to(dtype=torch.float32)
        return PytorchTensor(tensor)
