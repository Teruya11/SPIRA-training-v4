from src.spira_training.shared.core.models.event import Event
from src.spira_training.shared.ports.train_logger import TrainLogger


class DebugTrainLogger(TrainLogger):
    def log_event(self, event: Event) -> None:
        print(event)
