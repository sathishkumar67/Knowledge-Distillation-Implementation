from modules import * 

@dataclass
class Args:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-5
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_class: int = 128
    temperature: int = 2
    device_count: int = 1