import torch
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib.pyplot as plt

class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=22050,n_fft=2048, hop_length=512, n_mels=128, norm=True):
        super().__init__()

        self.norm = norm
        self.sample_rate = sample_rate

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.db_transform = T.AmplitudeToDB()

    def forward(self, waveform):
        mel = self.mel_transform(waveform)
        log_mel = self.db_transform(mel)
        return log_mel

def normalize_features(feature, eps=1e-6):
    mean = feature.mean(dim=(-2, -1), keepdim=True)
    std = feature.std(dim=(-2, -1), keepdim=True)
    return (feature - mean) / (std + eps)

def plot_spectrogram(features, duration=5.0, title="Log-Mel Spectrogram", normalized=True):
    """Trực quan hóa đặc trưng của một file audio."""
    spec = features.squeeze(0).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(11, 4))

    im = ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, spec.shape[0]],
        cmap="magma"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title(title)

    color_label = (
        "Normalized value" if normalized else "dB"
    )
    fig.colorbar(im, ax=ax, label=color_label)

    plt.tight_layout()
    return fig