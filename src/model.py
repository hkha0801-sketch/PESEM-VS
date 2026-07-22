import torch
import torch.nn as nn

from utils import stft, istft


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(5, 2), stride=(2, 1), padding=(2, 0)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(5, 2), stride=(2, 1), padding=(2, 0), out_padding=(1, 0), last=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride,
                                          padding=padding, output_padding=out_padding)
        self.last = last
        if not last:
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.PReLU()

    def forward(self, x):
        x = self.deconv(x)
        if not self.last:
            x = self.act(self.bn(x))
        return x


class CRN(nn.Module):
    """Convolutional Recurrent Network dự đoán mask trên magnitude spectrogram.
    Encoder (Conv2d) -> LSTM (bottleneck) -> Decoder (ConvTranspose2d, U-Net skip).
    Output: mask nhân trực tiếp với magnitude noisy, giữ nguyên phase noisy khi ISTFT.
    """

    def __init__(self, n_fft=512, hop_length=128, win_length=512,
                 base_channels=16, rnn_hidden=128, rnn_layers=2, **kwargs):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        c = base_channels

        # Encoder: 5 conv layers, downsample theo trục tần số
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.enc5 = ConvBlock(c * 8, c * 8)

        # số bin tần số còn lại sau 5 lần stride=2 theo freq: n_fft/2+1 -> chia 2^5
        freq_bins = n_fft // 2 + 1
        for _ in range(5):
            freq_bins = (freq_bins + 2 * 2 - 5) // 2 + 1  # theo công thức conv output
        self.rnn_input_size = c * 8 * freq_bins
        self.freq_bins_bottleneck = freq_bins

        self.lstm = nn.LSTM(self.rnn_input_size, rnn_hidden, num_layers=rnn_layers, batch_first=True)
        self.lstm_out_proj = nn.Linear(rnn_hidden, self.rnn_input_size)

        self.dec5 = DeconvBlock(c * 8 * 2, c * 8)
        self.dec4 = DeconvBlock(c * 8 * 2, c * 4)
        self.dec3 = DeconvBlock(c * 4 * 2, c * 2)
        self.dec2 = DeconvBlock(c * 2 * 2, c)
        self.dec1 = DeconvBlock(c * 2, 1, last=True)

        self.mask_act = nn.Sigmoid()

    def forward(self, noisy_waveform: torch.Tensor) -> torch.Tensor:
        """noisy_waveform: (B, T) -> enhanced_waveform: (B, T)"""
        length = noisy_waveform.shape[-1]
        spec = stft(noisy_waveform, self.n_fft, self.hop_length, self.win_length)  # (B,F,T') complex
        mag = torch.abs(spec)          # (B,F,T')
        phase = torch.angle(spec)

        x = mag.unsqueeze(1)           # (B,1,F,T')

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b, c, f, t = e5.shape
        rnn_in = e5.permute(0, 3, 1, 2).reshape(b, t, c * f)   # (B,T',C*F)
        rnn_out, _ = self.lstm(rnn_in)
        rnn_out = self.lstm_out_proj(rnn_out)                  # (B,T',C*F)
        rnn_out = rnn_out.reshape(b, t, c, f).permute(0, 2, 3, 1)  # (B,C,F,T')

        d5 = self.dec5(torch.cat([rnn_out, e5], dim=1))
        d5 = self._match(d5, e4)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d4 = self._match(d4, e3)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d3 = self._match(d3, e2)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d2 = self._match(d2, e1)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        d1 = self._match(d1, x)
        mask = self.mask_act(d1.squeeze(1))   # (B,F,T')

        enhanced_mag = mag * mask
        enhanced_spec = torch.polar(enhanced_mag, phase)
        enhanced_wav = istft(enhanced_spec, self.n_fft, self.hop_length, self.win_length, length=length)
        return enhanced_wav

    @staticmethod
    def _match(x, ref):
        """Cắt/pad x cho khớp shape (F,T') với ref, do conv/deconv có thể lệch 1 vài bin."""
        _, _, fh, fw = ref.shape
        _, _, xh, xw = x.shape
        if xh > fh:
            x = x[:, :, :fh, :]
        elif xh < fh:
            x = torch.nn.functional.pad(x, (0, 0, 0, fh - xh))
        if xw > fw:
            x = x[:, :, :, :fw]
        elif xw < fw:
            x = torch.nn.functional.pad(x, (0, fw - xw))
        return x


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    stft_cfg = cfg["stft"]
    name = model_cfg.get("name", "CRN")
    if name == "CRN":
        return CRN(
            n_fft=stft_cfg["n_fft"],
            hop_length=stft_cfg["hop_length"],
            win_length=stft_cfg["win_length"],
            base_channels=model_cfg.get("base_channels", 16),
            rnn_hidden=model_cfg.get("rnn_hidden", 128),
            rnn_layers=model_cfg.get("rnn_layers", 2),
        )
    raise ValueError(f"Unknown model name: {name}")
