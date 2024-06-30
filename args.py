from dataclasses import dataclass
from modules import * 
from typing import Optional


@dataclass
class Args:
    epochs: int = 1
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-5
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_class: Optional[int] = 128
    t_max: int = 50
    eta_min: float = 1e-4