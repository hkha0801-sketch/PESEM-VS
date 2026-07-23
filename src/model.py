import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import stft, istft


# ============================================================
# ACTIVATION
# ============================================================

def get_activation(name):
    """
    Tạo activation function tương ứng với config.
    Checkpoint của bạn đang dùng null nên mặc định Identity.
    """
    if name is None:
        return nn.Identity()

    name = str(name).lower()

    if name == "relu":
        return nn.ReLU()

    if name == "sigmoid":
        return nn.Sigmoid()

    if name == "tanh":
        return nn.Tanh()

    if name == "prelu":
        return nn.PReLU()

    if name == "none":
        return nn.Identity()

    return nn.Identity()


# ============================================================
# SEQUENCE MODEL
# ============================================================

class SequenceModel(nn.Module):
    """
    Sequence model tương thích với cấu trúc checkpoint:

    fb_model.sequence_model.weight_ih_l0
    fb_model.sequence_model.weight_hh_l0
    fb_model.sequence_model.bias_ih_l0
    fb_model.sequence_model.bias_hh_l0

    fb_model.fc_output_layer.weight
    fb_model.fc_output_layer.bias

    Tương tự cho sb_model.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers=2,
        bidirectional=False,
        sequence_model="LSTM",
        output_activate_function=None,
    ):
        super().__init__()

        assert sequence_model in ("GRU", "LSTM")

        self.sequence_model_type = sequence_model

        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )

        else:
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )

        output_size_in = hidden_size

        if bidirectional:
            output_size_in *= 2

        self.fc_output_layer = nn.Linear(
            output_size_in,
            output_size,
        )

        self.output_activate_function = get_activation(
            output_activate_function
        )

    def forward(self, x):
        """
        Input:
            (B, feature, T)

        LSTM cần:
            (B, T, feature)

        Output:
            (B, output_size, T)
        """

        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        x, _ = self.sequence_model(x)

        x = self.fc_output_layer(x)

        x = self.output_activate_function(x)

        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        return x


# ============================================================
# NORMALIZATION
# ============================================================

class OfflineLaplaceNorm(nn.Module):
    """
    Offline Laplace normalization gần với cách dùng
    trong FullSubNet.

    Input:
        (B, C, F, T)

    Normalize theo thời gian T.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):

        # Mean magnitude theo thời gian
        denominator = torch.mean(
            torch.abs(x),
            dim=-1,
            keepdim=True,
        )

        denominator = denominator + self.eps

        return x / denominator


# ============================================================
# FREQUENCY UNFOLD
# ============================================================

def freq_unfold(x, num_neighbors):
    """
    Lấy các frequency bins lân cận.

    Input:
        x: (B, C, F, T)

    Output:
        (B, C * (2 * num_neighbors + 1), F, T)
    """

    if num_neighbors == 0:
        return x

    # Pad frequency axis
    x = F.pad(
        x,
        (
            0, 0,                         # time
            num_neighbors, num_neighbors  # frequency
        ),
        mode="replicate",
    )

    # Unfold frequency
    x = x.unfold(
        dimension=2,
        size=2 * num_neighbors + 1,
        step=1,
    )

    # Hiện tại:
    # (B, C, F, T, neighborhood)

    x = x.permute(
        0, 1, 4, 2, 3
    )

    # (B, C, neighborhood, F, T)

    b, c, n, f, t = x.shape

    x = x.reshape(
        b,
        c * n,
        f,
        t,
    )

    return x


# ============================================================
# DROP BAND
# ============================================================

def drop_band(x, num_groups=2):
    """
    Phiên bản đơn giản của DropBand.

    Input:
        (B, C, F, T)

    Không làm thay đổi shape.

    Khi inference batch thường = 1 nên
    thực tế không ảnh hưởng.
    """

    if num_groups <= 0:
        return x

    # Không random drop khi inference
    # Để đảm bảo kết quả ổn định.
    return x


# ============================================================
# COMPRESSED cIRM
# ============================================================

def decompress_cIRM(x, K=10):
    """
    Decompress compressed Complex Ideal Ratio Mask.

    Input:
        (..., 2)

    Output:
        (..., 2)
    """

    # x có 2 kênh:
    # real và imaginary

    real = x[..., 0]
    imag = x[..., 1]

    # Công thức inverse compression
    real = real / K

    imag = imag / K

    real = torch.tanh(real)

    imag = torch.tanh(imag)

    return torch.stack(
        [real, imag],
        dim=-1,
    )


# ============================================================
# FULLSUBNET CORE
# ============================================================

class FullSubNetCore(nn.Module):

    def __init__(
        self,
        num_freqs,
        look_ahead,
        sequence_model,
        fb_num_neighbors,
        sb_num_neighbors,
        fb_output_activate_function,
        sb_output_activate_function,
        fb_model_hidden_size,
        sb_model_hidden_size,
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=2,
        weight_init=True,
    ):

        super().__init__()

        assert sequence_model in (
            "GRU",
            "LSTM",
        )

        self.num_freqs = num_freqs

        self.look_ahead = look_ahead

        self.fb_num_neighbors = fb_num_neighbors

        self.sb_num_neighbors = sb_num_neighbors

        self.num_groups_in_drop_band = (
            num_groups_in_drop_band
        )

        # ====================================================
        # FULL-BAND MODEL
        # ====================================================

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=(
                fb_output_activate_function
            ),
        )

        # ====================================================
        # SUB-BAND MODEL
        # ====================================================

        sb_input_size = (
            sb_num_neighbors * 2 + 1
        ) + (
            fb_num_neighbors * 2 + 1
        )

        self.sb_model = SequenceModel(
            input_size=sb_input_size,
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=(
                sb_output_activate_function
            ),
        )

        # ====================================================
        # NORMALIZATION
        # ====================================================

        if norm_type == "offline_laplace_norm":
            self.norm = OfflineLaplaceNorm()

        else:
            self.norm = OfflineLaplaceNorm()

        # ====================================================
        # WEIGHT INIT
        # ====================================================

        # QUAN TRỌNG:
        # Không initialize lại khi load checkpoint.
        #
        # Nhưng giữ argument để tương thích config.

        if weight_init:
            self._weight_init()

    def _weight_init(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                (
                    nn.LSTM,
                    nn.GRU,
                ),
            ):

                for name, param in module.named_parameters():

                    if "weight" in name:

                        nn.init.xavier_uniform_(
                            param
                        )

                    elif "bias" in name:

                        nn.init.zeros_(
                            param
                        )

    def forward(
        self,
        noisy_mag,
    ):

        # ====================================================
        # LOOK AHEAD
        # ====================================================

        noisy_mag = F.pad(
            noisy_mag,
            (
                0,
                self.look_ahead,
            ),
        )

        (
            batch_size,
            num_channels,
            num_freqs,
            num_frames,
        ) = noisy_mag.shape

        # ====================================================
        # FULL BAND
        # ====================================================

        fb_input = self.norm(
            noisy_mag
        )

        fb_input = fb_input.reshape(
            batch_size,
            num_channels * num_freqs,
            num_frames,
        )

        fb_output = self.fb_model(
            fb_input
        )

        fb_output = fb_output.reshape(
            batch_size,
            1,
            num_freqs,
            num_frames,
        )

        # ====================================================
        # FULL BAND FREQUENCY NEIGHBORS
        # ====================================================

        fb_output_unfolded = freq_unfold(
            fb_output,
            num_neighbors=self.fb_num_neighbors,
        )

        fb_output_unfolded = fb_output_unfolded.reshape(
            batch_size,
            num_freqs,
            self.fb_num_neighbors * 2 + 1,
            num_frames,
        )

        # ====================================================
        # NOISY MAG FREQUENCY NEIGHBORS
        # ====================================================

        noisy_mag_unfolded = freq_unfold(
            noisy_mag,
            num_neighbors=self.sb_num_neighbors,
        )

        noisy_mag_unfolded = noisy_mag_unfolded.reshape(
            batch_size,
            num_freqs,
            self.sb_num_neighbors * 2 + 1,
            num_frames,
        )

        # ====================================================
        # CONCAT
        # ====================================================

        sb_input = torch.cat(
            [
                noisy_mag_unfolded,
                fb_output_unfolded,
            ],
            dim=2,
        )

        sb_input = self.norm(
            sb_input
        )

        # ====================================================
        # DROP BAND
        # ====================================================

        if batch_size > 1:

            sb_input = drop_band(
                sb_input.permute(
                    0,
                    2,
                    1,
                    3,
                ),
                num_groups=(
                    self.num_groups_in_drop_band
                ),
            )

            num_freqs = sb_input.shape[2]

            sb_input = sb_input.permute(
                0,
                2,
                1,
                3,
            )

        # ====================================================
        # SUB BAND INPUT
        # ====================================================

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (
                self.sb_num_neighbors * 2
                + 1
            )
            + (
                self.fb_num_neighbors * 2
                + 1
            ),
            num_frames,
        )

        # ====================================================
        # SUB BAND MODEL
        # ====================================================

        sb_mask = self.sb_model(
            sb_input
        )

        # ====================================================
        # RESHAPE MASK
        # ====================================================

        sb_mask = sb_mask.reshape(
            batch_size,
            num_freqs,
            2,
            num_frames,
        )

        sb_mask = sb_mask.permute(
            0,
            2,
            1,
            3,
        ).contiguous()

        # ====================================================
        # REMOVE LOOK AHEAD
        # ====================================================

        return sb_mask[
            :,
            :,
            :,
            self.look_ahead:,
        ]


# ============================================================
# FULLSUBNET WAVEFORM WRAPPER
# ============================================================

class FullSubNetWrapper(
    FullSubNetCore
):

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        **fsn_kwargs,
    ):

        # QUAN TRỌNG:
        # Truyền fsn_kwargs vào FullSubNetCore

        super().__init__(
            **fsn_kwargs
        )

        self.n_fft = n_fft

        self.hop_length = hop_length

        self.win_length = win_length

    def forward(
        self,
        noisy_waveform,
    ):

        length = (
            noisy_waveform.shape[-1]
        )

        # ====================================================
        # STFT
        # ====================================================

        spec = stft(
            noisy_waveform,
            self.n_fft,
            self.hop_length,
            self.win_length,
        )

        # ====================================================
        # MAGNITUDE
        # ====================================================

        mag = torch.abs(
            spec
        ).unsqueeze(1)

        # ====================================================
        # FULLSUBNET
        # ====================================================

        compressed_cirm = super().forward(
            mag
        )

        # ====================================================
        # DECOMPRESS cIRM
        # ====================================================

        cirm = decompress_cIRM(
            compressed_cirm.permute(
                0,
                2,
                3,
                1,
            )
        )

        cirm = cirm.permute(
            0,
            3,
            1,
            2,
        )

        # ====================================================
        # COMPLEX MASK
        # ====================================================

        mask_real = cirm[
            :,
            0,
        ]

        mask_imag = cirm[
            :,
            1,
        ]

        noisy_real = spec.real

        noisy_imag = spec.imag

        # ====================================================
        # COMPLEX MULTIPLICATION
        # ====================================================

        enhanced_real = (
            noisy_real * mask_real
            - noisy_imag * mask_imag
        )

        enhanced_imag = (
            noisy_real * mask_imag
            + noisy_imag * mask_real
        )

        enhanced_spec = torch.complex(
            enhanced_real,
            enhanced_imag,
        )

        # ====================================================
        # ISTFT
        # ====================================================

        enhanced_wav = istft(
            enhanced_spec,
            self.n_fft,
            self.hop_length,
            self.win_length,
            length=length,
        )

        return enhanced_wav


# ============================================================
# CRN
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel=(5, 2),
        stride=(2, 1),
        padding=(2, 0),
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel,
            stride=stride,
            padding=padding,
        )

        self.bn = nn.BatchNorm2d(
            out_ch
        )

        self.act = nn.PReLU()

    def forward(
        self,
        x,
    ):

        return self.act(
            self.bn(
                self.conv(x)
            )
        )


class DeconvBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel=(5, 2),
        stride=(2, 1),
        padding=(2, 0),
        out_padding=(1, 0),
        last=False,
    ):

        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel,
            stride=stride,
            padding=padding,
            output_padding=out_padding,
        )

        self.last = last

        if not last:

            self.bn = nn.BatchNorm2d(
                out_ch
            )

            self.act = nn.PReLU()

    def forward(
        self,
        x,
    ):

        x = self.deconv(
            x
        )

        if not self.last:

            x = self.act(
                self.bn(x)
            )

        return x


class CRN(nn.Module):

    def __init__(
        self,
        n_fft=512,
        hop_length=128,
        win_length=512,
        base_channels=16,
        rnn_hidden=128,
        rnn_layers=2,
        **kwargs,
    ):

        super().__init__()

        self.n_fft = n_fft

        self.hop_length = hop_length

        self.win_length = win_length

        c = base_channels

        self.enc1 = ConvBlock(
            1,
            c,
        )

        self.enc2 = ConvBlock(
            c,
            c * 2,
        )

        self.enc3 = ConvBlock(
            c * 2,
            c * 4,
        )

        self.enc4 = ConvBlock(
            c * 4,
            c * 8,
        )

        self.enc5 = ConvBlock(
            c * 8,
            c * 8,
        )

        freq_bins = (
            n_fft // 2 + 1
        )

        for _ in range(5):

            freq_bins = (
                freq_bins
                + 2 * 2
                - 5
            ) // 2 + 1

        self.rnn_input_size = (
            c
            * 8
            * freq_bins
        )

        self.lstm = nn.LSTM(
            self.rnn_input_size,
            rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )

        self.lstm_out_proj = nn.Linear(
            rnn_hidden,
            self.rnn_input_size,
        )

        self.dec5 = DeconvBlock(
            c * 8 * 2,
            c * 8,
        )

        self.dec4 = DeconvBlock(
            c * 8 * 2,
            c * 4,
        )

        self.dec3 = DeconvBlock(
            c * 4 * 2,
            c * 2,
        )

        self.dec2 = DeconvBlock(
            c * 2 * 2,
            c,
        )

        self.dec1 = DeconvBlock(
            c * 2,
            1,
            last=True,
        )

        self.mask_act = nn.Sigmoid()

    def forward(
        self,
        noisy_waveform,
    ):

        length = (
            noisy_waveform.shape[-1]
        )

        spec = stft(
            noisy_waveform,
            self.n_fft,
            self.hop_length,
            self.win_length,
        )

        mag = torch.abs(
            spec
        )

        phase = torch.angle(
            spec
        )

        x = mag.unsqueeze(1)

        e1 = self.enc1(x)

        e2 = self.enc2(e1)

        e3 = self.enc3(e2)

        e4 = self.enc4(e3)

        e5 = self.enc5(e4)

        b, c, f, t = e5.shape

        rnn_in = e5.permute(
            0,
            3,
            1,
            2,
        ).reshape(
            b,
            t,
            c * f,
        )

        rnn_out, _ = self.lstm(
            rnn_in
        )

        rnn_out = self.lstm_out_proj(
            rnn_out
        )

        rnn_out = rnn_out.reshape(
            b,
            t,
            c,
            f,
        ).permute(
            0,
            2,
            3,
            1,
        )

        d5 = self.dec5(
            torch.cat(
                [
                    rnn_out,
                    e5,
                ],
                dim=1,
            )
        )

        d5 = self._match(
            d5,
            e4,
        )

        d4 = self.dec4(
            torch.cat(
                [
                    d5,
                    e4,
                ],
                dim=1,
            )
        )

        d4 = self._match(
            d4,
            e3,
        )

        d3 = self.dec3(
            torch.cat(
                [
                    d4,
                    e3,
                ],
                dim=1,
            )
        )

        d3 = self._match(
            d3,
            e2,
        )

        d2 = self.dec2(
            torch.cat(
                [
                    d3,
                    e2,
                ],
                dim=1,
            )
        )

        d2 = self._match(
            d2,
            e1,
        )

        d1 = self.dec1(
            torch.cat(
                [
                    d2,
                    e1,
                ],
                dim=1,
            )
        )

        d1 = self._match(
            d1,
            x,
        )

        mask = self.mask_act(
            d1.squeeze(1)
        )

        enhanced_mag = (
            mag * mask
        )

        enhanced_spec = torch.polar(
            enhanced_mag,
            phase,
        )

        enhanced_wav = istft(
            enhanced_spec,
            self.n_fft,
            self.hop_length,
            self.win_length,
            length=length,
        )

        return enhanced_wav

    @staticmethod
    def _match(
        x,
        ref,
    ):

        _, _, fh, fw = ref.shape

        _, _, xh, xw = x.shape

        if xh > fh:

            x = x[
                :,
                :,
                :fh,
                :,
            ]

        elif xh < fh:

            x = F.pad(
                x,
                (
                    0,
                    0,
                    0,
                    fh - xh,
                ),
            )

        if xw > fw:

            x = x[
                :,
                :,
                :,
                :fw,
            ]

        elif xw < fw:

            x = F.pad(
                x,
                (
                    0,
                    fw - xw,
                ),
            )

        return x


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    cfg: dict,
) -> nn.Module:

    model_cfg = cfg[
        "model"
    ]

    stft_cfg = cfg[
        "stft"
    ]

    name = model_cfg.get(
        "name",
        "CRN",
    )

    # ========================================================
    # CRN
    # ========================================================

    if name == "CRN":

        return CRN(
            n_fft=stft_cfg[
                "n_fft"
            ],

            hop_length=stft_cfg[
                "hop_length"
            ],

            win_length=stft_cfg[
                "win_length"
            ],

            base_channels=model_cfg.get(
                "base_channels",
                16,
            ),

            rnn_hidden=model_cfg.get(
                "rnn_hidden",
                128,
            ),

            rnn_layers=model_cfg.get(
                "rnn_layers",
                2,
            ),
        )

    # ========================================================
    # FULLSUBNET
    # ========================================================

    elif name == "FullSubNet":

        return FullSubNetWrapper(

            n_fft=stft_cfg[
                "n_fft"
            ],

            hop_length=stft_cfg[
                "hop_length"
            ],

            win_length=stft_cfg[
                "win_length"
            ],

            num_freqs=model_cfg[
                "num_freqs"
            ],

            look_ahead=model_cfg[
                "look_ahead"
            ],

            sequence_model=model_cfg[
                "sequence_model"
            ],

            fb_num_neighbors=model_cfg[
                "fb_num_neighbors"
            ],

            sb_num_neighbors=model_cfg[
                "sb_num_neighbors"
            ],

            fb_output_activate_function=(
                model_cfg[
                    "fb_output_activate_function"
                ]
            ),

            sb_output_activate_function=(
                model_cfg[
                    "sb_output_activate_function"
                ]
            ),

            fb_model_hidden_size=(
                model_cfg[
                    "fb_model_hidden_size"
                ]
            ),

            sb_model_hidden_size=(
                model_cfg[
                    "sb_model_hidden_size"
                ]
            ),

            norm_type=model_cfg.get(
                "norm_type",
                "offline_laplace_norm",
            ),

            num_groups_in_drop_band=(
                model_cfg.get(
                    "num_groups_in_drop_band",
                    2,
                )
            ),

            # QUAN TRỌNG
            # Không init weight mới
            # vì chúng ta sẽ load checkpoint.

            weight_init=False,
        )

    else:

        raise ValueError(
            f"Unknown model name: {name}"
        )