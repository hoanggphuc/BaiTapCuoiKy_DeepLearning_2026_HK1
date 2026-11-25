from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent[1]

@dataclass
class Config:
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"

    #Audio
    sample_rate: int = 25050 #tần số lấy mẫu mặc định
    duration: float = 5.0 #độ dài mặc định của âm thanh
    mono: bool = True #chuyển channel âm thanh sang về 1 channel
    backend: str = "soundfile" #đọc file âm thanh bằng thư viện soundfile

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError("Sample rate phải là một số nguyên dương.")
        if self.duration <= 0:
            raise ValueError("Duration phải là một số thực dương.")

    @property
    def num_samples(self):
        return round(self.sample_rate * self.duration)