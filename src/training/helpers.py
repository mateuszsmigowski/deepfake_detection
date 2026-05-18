import torch
import random
from src.loaders.config import RuntimeConfig, RuntimeDevice

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