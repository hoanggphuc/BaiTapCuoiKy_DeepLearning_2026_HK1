from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class AudioConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    audio_dir: Path = data_dir / "audio"
    metadata_file: Path = data_dir / "meta"/ "metadata.csv"

    #Audio
    sample_rate: int = 22050 #tần số lấy mẫu mặc định
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

class FeatureConfig:
    n_fft: int = 1024 #kích thước của FFT window
    hop_length: int = 512 #số lượng samples giữa các frames liên tiếp
    n_mels: int = 128 #số lượng dải tần Mel
    norm: bool = True #chuẩn hóa các giá trị của Mel spectrogram về khoảng [0, 1]

    def __post_init__(self):
        if self.n_fft <= 0:
            raise ValueError("n_fft phải là một số nguyên dương.")
        if self.hop_length <= 0:
            raise ValueError("hop_length phải là một số nguyên dương.")
        if self.n_mels <= 0:
            raise ValueError("n_mels phải là một số nguyên dương.")