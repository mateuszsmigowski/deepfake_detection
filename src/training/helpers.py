import torch
import random
from dataclasses import dataclass, field
from src.loaders.config import RuntimeConfig, RuntimeDevice
from torch.optim import AdamW
from src.loaders.config import TrainingConfig, ConfigModel

@dataclass
class EarlyStopping:
    patience: int
    counter: int = field(default=0, init=False)

    def step(self, improved: bool) -> bool:
        if improved:
            self.counter = 0
            return False
        
        self.counter += 1
        return self.counter >= self.patience

def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

def resolve_device(device: RuntimeConfig.Device) -> torch.device:

        match device:
            case RuntimeDevice.AUTO:
                if torch.cuda.is_available():
                    return torch.device("cuda")
                elif torch.backends.mps.is_available():
                    return torch.device("mps")
                else:
                    return torch.device("cpu")
            case RuntimeDevice.CPU:
                return torch.device("cpu")
            case RuntimeDevice.CUDA:
                return torch.device("cuda")
            case RuntimeDevice.MPS:
                return torch.device("mps")
            case _:
                raise ValueError(f"Invalid runtime device: {device}")

def configure_reproducibility(seed: int, deterministic: bool):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

def build_label_criterion(config: ConfigModel, device: torch.device) -> torch.nn.BCEWithLogitsLoss:

    if not config.training.pos_weight:
        return torch.nn.BCEWithLogitsLoss()
    
    if config.data.train_fake == 0:
        raise ValueError("Pos weight is not supported when train fake is 0")

    pos_weight = torch.tensor(
        [config.data.train_real / config.data.train_fake],
        device=device,
    )
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)