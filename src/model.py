import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import stft, istft


# ============================================================
# ACTIVATION
# ============================================================

def get_activation(name):

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
# SHARED WEIGHT INIT
# ============================================================

def generic_weight_init(module_root):
    """
    Weight init dùng chung cho FullSubNet và InterSubNet.

    Chỉ nên gọi khi KHÔNG load checkpoint (train from scratch),
    vì gọi sau khi load_state_dict sẽ ghi đè trọng số đã học.
    """

    for module in module_root.modules():

        if isinstance(module, nn.Linear):

            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LSTM, nn.GRU)):

            for name, param in module.named_parameters():

                if "weight" in name:
                    nn.init.xavier_uniform_(param)

                elif "bias" in name:
                    nn.init.zeros_(param)


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


        if weight_init:
            self._weight_init()

    def _weight_init(self):
        generic_weight_init(self)

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
# FULLSUBNET+ — CÁC MODULE PHỤ TRỢ
# ============================================================

# ------------------------------------------------------------
# CHANNEL ATTENTION LAYERS (theo attention_model.py gốc)
# ------------------------------------------------------------

class ChannelSELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) block, hoạt động theo chiều channel.

    Input:
        (B, num_channels, T)
    """

    def __init__(self, num_channels, reduction_ratio=2):
        super().__init__()

        num_channels_reduced = num_channels // reduction_ratio

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)

        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        squeeze_tensor = input_tensor.mean(dim=2)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))

        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()

        return torch.mul(
            input_tensor,
            fc_out_2.view(a, b, 1),
        )


class ChannelCBAMLayer(nn.Module):
    """
    Channel attention kiểu CBAM (dùng cả avg-pool và max-pool).

    Input:
        (B, num_channels, T)
    """

    def __init__(self, num_channels, reduction_ratio=2):
        super().__init__()

        num_channels_reduced = num_channels // reduction_ratio

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)

        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        mean_squeeze_tensor = input_tensor.mean(dim=2)

        max_squeeze_tensor, _ = torch.max(input_tensor, dim=2)

        mean_fc_out_1 = self.relu(self.fc1(mean_squeeze_tensor))

        max_fc_out_1 = self.relu(self.fc1(max_squeeze_tensor))

        fc_out_1 = mean_fc_out_1 + max_fc_out_1

        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = mean_squeeze_tensor.size()

        return torch.mul(
            input_tensor,
            fc_out_2.view(a, b, 1),
        )


class ChannelECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA).

    Input:
        (B, num_channels, T)
    """

    def __init__(self, channel, k_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.avg_pool(x)

        y = self.conv(
            y.transpose(-1, -2)
        ).transpose(-1, -2)

        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ChannelTimeSenseSELayer(nn.Module):
    """
    MulCA: multi-scale time-sensitive channel attention (SE).

    Dùng 3 nhánh Conv1d (kernel nhỏ/vừa/lớn) để lấy đặc trưng đa tỉ lệ
    theo chiều thời gian, sau đó gộp lại bằng 1 lớp FC rồi mới đi qua
    squeeze-excite như SE thông thường.

    Input:
        (B, num_channels, T)
    """

    def __init__(
        self,
        num_channels,
        reduction_ratio=2,
        kersize=(3, 5, 10),
        subband_num=1,
    ):
        super().__init__()

        num_channels_reduced = num_channels // reduction_ratio

        groups = num_channels // subband_num

        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=groups),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True),
        )

        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=groups),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True),
        )

        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=groups),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True),
        )

        self.feature_concate_fc = nn.Linear(3, 1, bias=True)

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)

        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        small_feature = self.smallConv1d(input_tensor)

        middle_feature = self.middleConv1d(input_tensor)

        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat(
            [small_feature, middle_feature, large_feature],
            dim=2,
        )

        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))

        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()

        return torch.mul(
            input_tensor,
            fc_out_2.view(a, b, 1),
        )


def get_plus_channel_attention(name, num_channels, kersize=(3, 5, 10)):
    """
    Factory chọn loại channel attention cho FullSubNet+.

    Hỗ trợ: "SE", "ECA", "CBAM", "TSSE" (không phân biệt hoa/thường).
    Mặc định (và cũng là kiến trúc dùng trong checkpoint gốc): "TSSE".
    """

    key = str(name).upper() if name else "TSSE"

    if key == "SE":
        return ChannelSELayer(num_channels=num_channels)

    if key == "ECA":
        return ChannelECALayer(channel=num_channels)

    if key == "CBAM":
        return ChannelCBAMLayer(num_channels=num_channels)

    # "TSSE" hoặc bất kỳ giá trị nào khác -> mặc định TSSE (MulCA)
    return ChannelTimeSenseSELayer(
        num_channels=num_channels,
        kersize=kersize,
    )


# ------------------------------------------------------------
# TCN BLOCK + FULL-BAND EXTRACTOR (theo causal_conv.py gốc)
# ------------------------------------------------------------

class TCNBlockPlus(nn.Module):
    """
    1 khối TCN (Temporal Convolutional Network), dùng làm full-band
    model trong FullSubNet+ thay cho LSTM/GRU.

    Input / Output:
        (B, channels, T)
    """

    def __init__(
        self,
        in_channels=257,
        hidden_channel=512,
        out_channels=257,
        kernel_size=3,
        dilation=1,
        use_skip_connection=True,
        causal=False,
    ):
        super().__init__()

        self.conv1x1 = nn.Conv1d(in_channels, hidden_channel, 1)

        self.prelu1 = nn.PReLU()

        self.norm1 = nn.GroupNorm(1, hidden_channel, eps=1e-8)

        padding = (
            (dilation * (kernel_size - 1)) // 2
            if not causal
            else (dilation * (kernel_size - 1))
        )

        self.depthwise_conv = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel_size=kernel_size,
            stride=1,
            groups=hidden_channel,
            padding=padding,
            dilation=dilation,
        )

        self.prelu2 = nn.PReLU()

        self.norm2 = nn.GroupNorm(1, hidden_channel, eps=1e-8)

        self.sconv = nn.Conv1d(hidden_channel, out_channels, 1)

        self.causal = causal

        self.padding = padding

        self.use_skip_connection = use_skip_connection

    def forward(self, x):

        y = self.conv1x1(x)

        y = self.norm1(self.prelu1(y))

        y = self.depthwise_conv(y)

        if self.causal:
            y = y[:, :, :-self.padding]

        y = self.norm2(self.prelu2(y))

        output = self.sconv(y)

        if self.use_skip_connection:
            return x + output

        return output


class TCNSequenceModel(nn.Module):
    """
    Full-band model dạng TCN dùng trong FullSubNet+
    (tương ứng SequenceModel(sequence_model="TCN") trong repo gốc).

    Gồm 8 khối TCNBlockPlus (dilation lần lượt 1,2,5,9,1,2,5,9) rồi
    ReLU, sau đó 1 lớp Linear để ra output_size, cuối cùng là hàm
    activation tùy chọn.

    Input:
        (B, input_size, T)

    Output:
        (B, output_size, T)
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_channel=512,
        output_activate_function=None,
    ):
        super().__init__()

        dilations = (1, 2, 5, 9, 1, 2, 5, 9)

        self.sequence_model = nn.Sequential(
            *[
                TCNBlockPlus(
                    in_channels=input_size,
                    hidden_channel=hidden_channel,
                    out_channels=input_size,
                    dilation=d,
                )
                for d in dilations
            ],
            nn.ReLU(),
        )

        self.fc_output_layer = nn.Linear(input_size, output_size)

        self.output_activate_function = get_activation(
            output_activate_function
        )

    def forward(self, x):

        x = self.sequence_model(x)

        o = self.fc_output_layer(
            x.permute(0, 2, 1)
        )

        o = self.output_activate_function(o)

        o = o.permute(0, 2, 1)

        return o


# ------------------------------------------------------------
# NORMALIZE / UNFOLD / DROP-BAND RIÊNG CHO FULLSUBNET+
# ------------------------------------------------------------

class OfflineLaplaceNormPlus(nn.Module):
    """
    Chuẩn hóa kiểu "laplace" dùng trong FullSubNet+.

    Input:
        (B, C, F, T) (4 chiều bất kỳ)

    mu = trung bình (KHÔNG lấy trị tuyệt đối) trên toàn bộ (C, F, T)
    của từng mẫu trong batch.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):

        mu = torch.mean(
            x,
            dim=(1, 2, 3),
            keepdim=True,
        )

        return x / (mu + self.eps)


def subband_unfold_plus(input_tensor, num_neighbor):
    """
    Unfold theo chiều tần số (dùng pad "reflect" + F.unfold),
    giống hệt BaseModel.unfold() trong repo gốc.

    Input:
        (B, C, F, T)

    Output:
        (B, F, C, 2*num_neighbor + 1, T)
    """

    assert input_tensor.dim() == 4

    (
        batch_size,
        num_channels,
        num_freqs,
        num_frames,
    ) = input_tensor.size()

    if num_neighbor < 1:
        return input_tensor.permute(0, 2, 1, 3).reshape(
            batch_size, num_freqs, num_channels, 1, num_frames,
        )

    output = input_tensor.reshape(
        batch_size * num_channels, 1, num_freqs, num_frames,
    )

    sub_band_unit_size = num_neighbor * 2 + 1

    output = F.pad(
        output,
        [0, 0, num_neighbor, num_neighbor],
        mode="reflect",
    )

    output = F.unfold(
        output,
        (sub_band_unit_size, num_frames),
    )

    output = output.reshape(
        batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs,
    )

    output = output.permute(0, 4, 1, 2, 3).contiguous()

    return output


def subband_drop_band_plus(input_tensor, num_groups=2):
    """
    Drop-band dùng trong FullSubNet+ khi batch_size > num_groups
    (chỉ có tác dụng lúc train nhiều batch; khi infer batch_size=1
    thì hàm này không được gọi tới).
    """

    batch_size, _, num_freqs, _ = input_tensor.shape

    if num_groups <= 1:
        return input_tensor

    if num_freqs % num_groups != 0:
        input_tensor = input_tensor[
            ..., :(num_freqs - (num_freqs % num_groups)), :
        ]
        num_freqs = input_tensor.shape[2]

    output = []

    for group_idx in range(num_groups):

        samples_indices = torch.arange(
            group_idx, batch_size, num_groups,
            device=input_tensor.device,
        )

        freqs_indices = torch.arange(
            group_idx, num_freqs, num_groups,
            device=input_tensor.device,
        )

        selected_samples = torch.index_select(
            input_tensor, dim=0, index=samples_indices,
        )

        selected = torch.index_select(
            selected_samples, dim=2, index=freqs_indices,
        )

        output.append(selected)

    return torch.cat(output, dim=0)


# ============================================================
# FULLSUBNET+ CORE
# ============================================================

class FullSubNetPlusCore(nn.Module):
    """
    FullSubNet+ (khớp state_dict với checkpoint gốc của
    RookieJunChen/FullSubNet-plus).

    Input:
        noisy_mag, noisy_real, noisy_imag: (B, 1, F, T)

    Output:
        compressed cIRM: (B, output_size, F, T_valid)
    """

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
        channel_attention_model="TSSE",
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=2,
        output_size=2,
        subband_num=1,
        kersize=(3, 5, 10),
        weight_init=True,
        **_ignored_extra_kwargs,
    ):

        super().__init__()

        assert sequence_model in ("GRU", "LSTM", "TCN")

        if subband_num == 1:
            self.num_channels = num_freqs
        else:
            self.num_channels = num_freqs // subband_num + 1

        # ====================================================
        # CHANNEL ATTENTION (1 CÁI CHO MỖI LUỒNG MAG/REAL/IMAG)
        # ====================================================

        self.channel_attention = get_plus_channel_attention(
            channel_attention_model, self.num_channels, kersize,
        )

        self.channel_attention_real = get_plus_channel_attention(
            channel_attention_model, self.num_channels, kersize,
        )

        self.channel_attention_imag = get_plus_channel_attention(
            channel_attention_model, self.num_channels, kersize,
        )

        # ====================================================
        # 3 FULL-BAND MODEL (TCN) CHO MAG / REAL / IMAG
        # ====================================================

        self.fb_model = TCNSequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_channel=512,
            output_activate_function=fb_output_activate_function,
        )

        self.fb_model_real = TCNSequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_channel=512,
            output_activate_function=fb_output_activate_function,
        )

        self.fb_model_imag = TCNSequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_channel=512,
            output_activate_function=fb_output_activate_function,
        )

        # ====================================================
        # SUB-BAND MODEL (LSTM/GRU như FullSubNet gốc)
        # ====================================================

        self.sb_model = SequenceModel(
            input_size=(
                (sb_num_neighbors * 2 + 1)
                + 3 * (fb_num_neighbors * 2 + 1)
            ),
            output_size=output_size,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function,
        )

        self.subband_num = subband_num

        self.sb_num_neighbors = sb_num_neighbors

        self.fb_num_neighbors = fb_num_neighbors

        self.look_ahead = look_ahead

        self.num_groups_in_drop_band = num_groups_in_drop_band

        self.output_size = output_size

        if norm_type == "offline_laplace_norm":
            self.norm = OfflineLaplaceNormPlus()

        else:
            self.norm = OfflineLaplaceNormPlus()

        if weight_init:
            generic_weight_init(self)

    def forward(self, noisy_mag, noisy_real, noisy_imag):

        noisy_mag = F.pad(noisy_mag, [0, self.look_ahead])

        noisy_real = F.pad(noisy_real, [0, self.look_ahead])

        noisy_imag = F.pad(noisy_imag, [0, self.look_ahead])

        (
            batch_size,
            num_channels,
            num_freqs,
            num_frames,
        ) = noisy_mag.size()

        assert num_channels == 1

        # ====================================================
        # FULL-BAND MAGNITUDE
        # ====================================================

        fb_input = self.norm(noisy_mag).reshape(
            batch_size, num_channels * num_freqs, num_frames,
        )

        fb_input = self.channel_attention(fb_input)

        fb_output = self.fb_model(fb_input).reshape(
            batch_size, 1, num_freqs, num_frames,
        )

        # ====================================================
        # FULL-BAND REAL
        # ====================================================

        fbr_input = self.norm(noisy_real).reshape(
            batch_size, num_channels * num_freqs, num_frames,
        )

        fbr_input = self.channel_attention_real(fbr_input)

        fbr_output = self.fb_model_real(fbr_input).reshape(
            batch_size, 1, num_freqs, num_frames,
        )

        # ====================================================
        # FULL-BAND IMAGINARY
        # ====================================================

        fbi_input = self.norm(noisy_imag).reshape(
            batch_size, num_channels * num_freqs, num_frames,
        )

        fbi_input = self.channel_attention_imag(fbi_input)

        fbi_output = self.fb_model_imag(fbi_input).reshape(
            batch_size, 1, num_freqs, num_frames,
        )

        # ====================================================
        # UNFOLD 3 FULL-BAND OUTPUT + NOISY MAG NEIGHBORS
        # ====================================================

        fb_output_unfolded = subband_unfold_plus(
            fb_output, self.fb_num_neighbors,
        ).reshape(
            batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames,
        )

        fbr_output_unfolded = subband_unfold_plus(
            fbr_output, self.fb_num_neighbors,
        ).reshape(
            batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames,
        )

        fbi_output_unfolded = subband_unfold_plus(
            fbi_output, self.fb_num_neighbors,
        ).reshape(
            batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames,
        )

        noisy_mag_unfolded = subband_unfold_plus(
            fb_input.reshape(batch_size, 1, num_freqs, num_frames),
            self.sb_num_neighbors,
        ).reshape(
            batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames,
        )

        # ====================================================
        # CONCAT + NORM
        # ====================================================

        sb_input = torch.cat(
            [
                noisy_mag_unfolded,
                fb_output_unfolded,
                fbr_output_unfolded,
                fbi_output_unfolded,
            ],
            dim=2,
        )

        sb_input = self.norm(sb_input)

        # ====================================================
        # DROP BAND (chỉ áp dụng khi batch_size > 1)
        # ====================================================

        if batch_size > 1:

            sb_input = subband_drop_band_plus(
                sb_input.permute(0, 2, 1, 3),
                num_groups=self.num_groups_in_drop_band,
            )

            num_freqs = sb_input.shape[2]

            sb_input = sb_input.permute(0, 2, 1, 3)

        # ====================================================
        # SUB-BAND MODEL
        # ====================================================

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (
                (self.sb_num_neighbors * 2 + 1)
                + 3 * (self.fb_num_neighbors * 2 + 1)
            ),
            num_frames,
        )

        sb_mask = self.sb_model(sb_input)

        sb_mask = sb_mask.reshape(
            batch_size, num_freqs, self.output_size, num_frames,
        ).permute(0, 2, 1, 3).contiguous()

        return sb_mask[:, :, :, self.look_ahead:]


# ============================================================
# FULLSUBNET+ WAVEFORM WRAPPER
# ============================================================

class FullSubNetPlusWrapper(FullSubNetPlusCore):

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        **fsn_kwargs,
    ):

        super().__init__(**fsn_kwargs)

        self.n_fft = n_fft

        self.hop_length = hop_length

        self.win_length = win_length

    def forward(self, noisy_waveform):

        length = noisy_waveform.shape[-1]

        # ====================================================
        # STFT
        # ====================================================

        spec = stft(
            noisy_waveform,
            self.n_fft,
            self.hop_length,
            self.win_length,
        )

        mag = torch.abs(spec).unsqueeze(1)

        real = spec.real.unsqueeze(1)

        imag = spec.imag.unsqueeze(1)

        # ====================================================
        # FULLSUBNET+
        # ====================================================

        compressed_cirm = super().forward(mag, real, imag)

        # ====================================================
        # DECOMPRESS cIRM
        # ====================================================

        cirm = decompress_cIRM(
            compressed_cirm.permute(0, 2, 3, 1)
        )

        cirm = cirm.permute(0, 3, 1, 2)

        mask_real = cirm[:, 0]

        mask_imag = cirm[:, 1]

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

        enhanced_spec = torch.complex(enhanced_real, enhanced_imag)

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


class SubbandInteraction(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.PReLU(),
        )

        self.mean_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
        )

        self.output_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, input_size),
            nn.PReLU(),
        )

        self.norm = nn.LayerNorm(input_size)

    def forward(self, x):
        # x: (B, F, T, C)

        h_local = self.input_linear(x)
        # (B, F, T, H)

        h_mean = h_local.mean(dim=1, keepdim=True)
        # (B, 1, T, H) - thông tin toàn cục theo trục tần số

        h_global = self.mean_linear(h_mean)
        # (B, 1, T, H)

        h_global = h_global.expand(-1, h_local.shape[1], -1, -1)
        # (B, F, T, H)

        combined = torch.cat([h_local, h_global], dim=-1)
        # (B, F, T, 2H)

        out = self.output_linear(combined)
        # (B, F, T, C)

        # Residual + norm
        return self.norm(x + out)


class SubInterLSTMBlock(nn.Module):


    def __init__(
        self,
        input_size,
        subinter_hidden_size,
        rnn_hidden_size,
        sequence_model="LSTM",
    ):
        super().__init__()

        assert sequence_model in ("LSTM", "GRU")

        self.SubInter = SubbandInteraction(
            input_size,
            subinter_hidden_size,
        )

        if sequence_model == "LSTM":
            self.RNN = nn.LSTM(
                input_size,
                rnn_hidden_size,
                num_layers=1,
                batch_first=True,
            )
        else:
            self.RNN = nn.GRU(
                input_size,
                rnn_hidden_size,
                num_layers=1,
                batch_first=True,
            )

        self.norm = nn.LayerNorm(rnn_hidden_size)

    def forward(self, x):
        # x: (B, F, T, C)

        b, f, t, c = x.shape

        x = self.SubInter(x)

        x = x.reshape(b * f, t, c)

        x, _ = self.RNN(x)

        x = self.norm(x)

        x = x.reshape(b, f, t, -1)

        return x


class InterSubbandModel(nn.Module):

    def __init__(
        self,
        sb_num_neighbors,
        sil_hidden_sizes=(93, 307),
        rnn_hidden_size=384,
        output_size=2,
        sequence_model="LSTM",
    ):
        super().__init__()

        input_size = sb_num_neighbors * 2 + 1

        blocks = []
        in_size = input_size

        for hidden_size in sil_hidden_sizes:

            blocks.append(
                SubInterLSTMBlock(
                    in_size,
                    hidden_size,
                    rnn_hidden_size,
                    sequence_model=sequence_model,
                )
            )

            in_size = rnn_hidden_size

        self.sequence_list = nn.ModuleList(blocks)

        self.fc_output_layer = nn.Linear(
            rnn_hidden_size,
            output_size,
        )

    def forward(self, x):
        # x: (B, F, T, C_in)

        for block in self.sequence_list:
            x = block(x)

        x = self.fc_output_layer(x)
        # (B, F, T, output_size)

        return x


# ============================================================
# INTERSUBNET CORE
# ============================================================

class InterSubNetCore(nn.Module):
    """
    InterSubNet không có nhánh full-band (fb_model) như FullSubNet.
    Toàn bộ mô hình chỉ có một sub-band model (self.sb_model),
    nhận trực tiếp noisy magnitude đã unfold theo tần số.
    """

    def __init__(
        self,
        num_freqs,
        look_ahead,
        sequence_model,
        sb_num_neighbors,
        sb_output_activate_function,
        sb_model_hidden_size,
        sil_hidden_sizes=(93, 307),
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=2,
        weight_init=True,
    ):
        super().__init__()

        assert sequence_model in ("GRU", "LSTM")

        self.num_freqs = num_freqs

        self.look_ahead = look_ahead

        self.sb_num_neighbors = sb_num_neighbors

        self.num_groups_in_drop_band = num_groups_in_drop_band

        # ====================================================
        # SUB-BAND MODEL (với subband interaction)
        # ====================================================

        self.sb_model = InterSubbandModel(
            sb_num_neighbors=sb_num_neighbors,
            sil_hidden_sizes=sil_hidden_sizes,
            rnn_hidden_size=sb_model_hidden_size,
            output_size=2,
            sequence_model=sequence_model,
        )

        self.sb_output_activate_function = get_activation(
            sb_output_activate_function
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

        if weight_init:
            generic_weight_init(self)

    def forward(self, noisy_mag):
        """
        Input:
            noisy_mag: (B, 1, F, T)

        Output:
            compressed cIRM: (B, 2, F, T)
        """

        # ====================================================
        # LOOK AHEAD
        # ====================================================

        noisy_mag = F.pad(
            noisy_mag,
            (0, self.look_ahead),
        )

        (
            batch_size,
            num_channels,
            num_freqs,
            num_frames,
        ) = noisy_mag.shape

        # ====================================================
        # NORMALIZE
        # ====================================================

        noisy_mag = self.norm(noisy_mag)

        # ====================================================
        # FREQUENCY NEIGHBORS (subband units)
        # ====================================================

        sb_input = freq_unfold(
            noisy_mag,
            num_neighbors=self.sb_num_neighbors,
        )

        sb_input = sb_input.reshape(
            batch_size,
            self.sb_num_neighbors * 2 + 1,
            num_freqs,
            num_frames,
        )

        # ====================================================
        # DROP BAND (chỉ áp dụng khi train, batch_size > 1)
        # ====================================================

        if batch_size > 1:

            sb_input = drop_band(
                sb_input,
                num_groups=self.num_groups_in_drop_band,
            )

            num_freqs = sb_input.shape[2]

        # ====================================================
        # (B, C, F, T) -> (B, F, T, C) CHO SUBBAND INTERACTION
        # ====================================================

        sb_input = sb_input.permute(0, 2, 3, 1)

        # ====================================================
        # SUB-BAND MODEL (SubInter + RNN x N blocks)
        # ====================================================

        sb_mask = self.sb_model(sb_input)
        # (B, F, T, 2)

        sb_mask = self.sb_output_activate_function(sb_mask)

        sb_mask = sb_mask.permute(0, 3, 1, 2).contiguous()
        # (B, 2, F, T)

        # ====================================================
        # REMOVE LOOK AHEAD
        # ====================================================

        return sb_mask[:, :, :, self.look_ahead:]


# ============================================================
# INTERSUBNET WAVEFORM WRAPPER
# ============================================================

class InterSubNetWrapper(InterSubNetCore):

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        **isn_kwargs,
    ):
        super().__init__(**isn_kwargs)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, noisy_waveform):

        length = noisy_waveform.shape[-1]

        # ====================================================
        # STFT
        # ====================================================

        spec = stft(
            noisy_waveform,
            self.n_fft,
            self.hop_length,
            self.win_length,
        )

        mag = torch.abs(spec).unsqueeze(1)

        # ====================================================
        # INTERSUBNET -> COMPRESSED cIRM
        # ====================================================

        compressed_cirm = super().forward(mag)

        # ====================================================
        # DECOMPRESS cIRM
        # ====================================================

        cirm = decompress_cIRM(
            compressed_cirm.permute(0, 2, 3, 1)
        )

        cirm = cirm.permute(0, 3, 1, 2)

        mask_real = cirm[:, 0]
        mask_imag = cirm[:, 1]

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
# CONV-TASNET HELPER MODULES
# ============================================================

class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError(f"{self.__class__.__name__} accepts 3D tensor as input")
        x = torch.transpose(x, 1, 2)
        x = super().forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError(f"{self.__class__.__name__} accepts 3D tensor as input")
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


def select_norm(norm, dim):
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError(f"Unsupported normalize layer: {norm}")
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D_Block(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, dilation=1, norm_type='gLN'):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm_type, out_channels)
        if norm_type == 'gLN':
            self.padding = (dilation * (kernel_size - 1)) // 2
        else:
            self.padding = dilation * (kernel_size - 1)
        self.dwconv = nn.Conv1d(out_channels, out_channels, kernel_size, 1, dilation=dilation, padding=self.padding, groups=out_channels, bias=True)
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm_type, out_channels)
        self.sconv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.norm_type = norm_type

    def forward(self, x):
        w = self.conv1x1(x)
        w = self.norm1(self.prelu1(w))
        w = self.dwconv(w)
        if self.norm_type == 'cLN':
            w = w[:, :, :-self.padding]
        w = self.norm2(self.prelu2(w))
        w = self.sconv(w)
        x = x + w
        return x


class TCN(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, norm_type='gLN', X=8):
        super().__init__()
        seq = [Conv1D_Block(in_channels, out_channels, kernel_size, 2**i, norm_type) for i in range(X)]
        self.tcn = nn.Sequential(*seq)

    def forward(self, x):
        return self.tcn(x)


class Separation(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, norm_type='gLN', X=8, R=3):
        super().__init__()
        s = [TCN(in_channels, out_channels, kernel_size, norm_type, X) for _ in range(R)]
        self.sep = nn.Sequential(*s)

    def forward(self, x):
        return self.sep(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, bottleneck=128, kernel_size=16, norm_type='gLN'):
        super().__init__()
        self.encoder = nn.Conv1d(in_channels, out_channels, kernel_size, kernel_size // 2, padding=0)
        self.norm = select_norm(norm_type, out_channels)
        self.conv1x1 = nn.Conv1d(out_channels, bottleneck, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        w = self.norm(x)
        w = self.conv1x1(w)
        return x, w


class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, kernel_size=16):
        super().__init__()
        self.decoder = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, kernel_size // 2, padding=0, bias=True)

    def forward(self, x):
        x = self.decoder(x)
        return x.squeeze(1) if x.dim() == 3 and x.size(1) == 1 else x.squeeze()

# ============================================================
# CONV-TASNET CORE
# ============================================================

class ConvTasNetCore(nn.Module):
    """
    Conv-TasNet hoạt động trực tiếp trên Time-Domain (Waveform),
    không qua bước biến đổi STFT (Encoder1D -> Separation -> Decoder1D).
    """

    def __init__(
        self,
        N=512,
        L=16,
        B=128,
        H=512,
        P=3,
        X=8,
        R=3,
        norm="gLN",
        num_spks=1,
        activate="relu",
        causal=False,
    ):
        super().__init__()

        self.num_spks = num_spks

        # ----------------------------------------------------
        # ENCODER / SEPARATION / DECODER
        # ----------------------------------------------------
        self.encoder = Encoder(1, N, B, L, norm)
        self.separation = Separation(B, H, P, norm, X, R)
        self.decoder = Decoder(H, 1, L)
        self.mask = nn.Conv1d(B, H * num_spks, 1, 1)

        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": lambda x: F.softmax(x, dim=0),
        }
        if activate not in supported_nonlinear:
            raise RuntimeError(f"Unsupported non-linear function: {activate}")

        self.non_linear = supported_nonlinear[activate]

    def forward(self, noisy_y):
        """
        Input:
            noisy_y: (B, T) hoặc (B, 1, T) - Waveform thời gian
        Output:
            enhanced_y: (B, T) - Waveform sau khi triệt nhiễu
        """
        if noisy_y.dim() == 1:
            noisy_y = noisy_y.unsqueeze(0)

        x, w = self.encoder(noisy_y)
        w = self.separation(w)
        m = self.mask(w)
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.non_linear(torch.stack(m, dim=0))

        d = [x * m[i] for i in range(self.num_spks)]
        s = [self.decoder(d[i]) for i in range(self.num_spks)]

        # Lấy nguồn tín hiệu đầu ra duy nhất (Single-speaker speech enhancement)
        enhanced_y = s[0]

        # Đảm bảo shape đầu ra là (B, T) khớp với chiều dài input
        if enhanced_y.shape[-1] > noisy_y.shape[-1]:
            enhanced_y = enhanced_y[..., : noisy_y.shape[-1]]
        elif enhanced_y.shape[-1] < noisy_y.shape[-1]:
            enhanced_y = F.pad(enhanced_y, (0, noisy_y.shape[-1] - enhanced_y.shape[-1]))

        return enhanced_y

# ============================================================
# CONV-TASNET WAVEFORM WRAPPER
# ============================================================

class ConvTasNetWrapper(ConvTasNetCore):
    """
    Wrapper đồng bộ với định dạng chung trong dự án (nhận n_fft, hop_length,...
    nhưng Conv-TasNet xử lý trực tiếp tín hiệu dạng sóng Waveform).
    """

    def __init__(
        self,
        n_fft=None,
        hop_length=None,
        win_length=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, noisy_y):
        return super().forward(noisy_y)

# ============================================================
# METRICGAN+ 
# ============================================================

class MetricGANPlusGenerator(nn.Module):
    """
    MetricGAN+ Generator

    Input:
        x: (B, T, 257)

    Output:
        mask: (B, T, 257)
    """

    def __init__(
        self,
        num_freqs=257,
        hidden_size=200,
        num_layers=2,
        dropout=0.1,
        causal=False,
    ):
        super().__init__()

        self.num_freqs = num_freqs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.causal = causal

        # ------------------------------------------------
        # LSTM
        # ------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=num_freqs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=not causal,
            batch_first=True,
        )

        # ------------------------------------------------
        # LSTM Initialization
        # ------------------------------------------------

        for name, param in self.lstm.named_parameters():

            if "bias" in name:
                nn.init.zeros_(param)

            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)

            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        # ------------------------------------------------
        # LSTM output dimension
        # ------------------------------------------------

        lstm_dim = (
            hidden_size * 2
            if not causal
            else hidden_size
        )

        # ------------------------------------------------
        # Fully Connected
        # ------------------------------------------------

        self.fc1 = nn.Linear(
            lstm_dim,
            300
        )

        self.fc2 = nn.Linear(
            300,
            num_freqs
        )

        # ------------------------------------------------
        # Xavier Initialization
        # ------------------------------------------------

        nn.init.xavier_uniform_(
            self.fc1.weight
        )

        nn.init.zeros_(
            self.fc1.bias
        )

        nn.init.xavier_uniform_(
            self.fc2.weight
        )

        nn.init.zeros_(
            self.fc2.bias
        )

        # ------------------------------------------------
        # Activation
        # ------------------------------------------------

        self.LReLU = nn.LeakyReLU(
            negative_slope=0.3
        )

        # ------------------------------------------------
        # Learnable Sigmoid
        # ------------------------------------------------

        self.Learnable_sigmoid = (
            LearnableSigmoid(
                in_features=num_freqs
            )
        )

    def forward(
        self,
        x,
        lengths=None
    ):

        # x:
        # (B, T, F)

        # ------------------------------------------------
        # Pack sequence
        # ------------------------------------------------

        if lengths is not None:

            lengths = lengths.cpu()

            x = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )

        # ------------------------------------------------
        # LSTM
        # ------------------------------------------------

        outputs, _ = self.lstm(
            x
        )

        # ------------------------------------------------
        # Unpack sequence
        # ------------------------------------------------

        if lengths is not None:

            outputs, _ = (
                nn.utils.rnn.pad_packed_sequence(
                    outputs,
                    batch_first=True,
                )
            )

        # ------------------------------------------------
        # FC1
        # ------------------------------------------------

        outputs = self.fc1(
            outputs
        )

        outputs = self.LReLU(
            outputs
        )

        # ------------------------------------------------
        # FC2
        # ------------------------------------------------

        outputs = self.fc2(
            outputs
        )

        # ------------------------------------------------
        # Learnable Sigmoid
        # ------------------------------------------------

        outputs = (
            self.Learnable_sigmoid(
                outputs
            )
        )

        return outputs


class LearnableSigmoid(nn.Module):
    """
    Learnable Sigmoid used in MetricGAN+ Generator.
    """

    def __init__(
        self,
        in_features=257
    ):
        super().__init__()

        self.slope = nn.Parameter(
            torch.ones(
                in_features
            )
        )

    def forward(
        self,
        x
    ):

        return (
            1.2
            * torch.sigmoid(
                self.slope * x
            )
        )


class MetricGANPlusDiscriminator(
    nn.Module
):
    """
    MetricGAN+ Discriminator

    Input:
        x: (B, 2, T, F)

    Output:
        metric score:
        (B, num_target_metrics)
    """

    def __init__(
        self,
        num_target_metrics=1
    ):
        super().__init__()

        # ------------------------------------------------
        # Batch Normalization
        # ------------------------------------------------

        self.BN = nn.BatchNorm2d(
            num_features=2,
            momentum=0.01
        )

        # ------------------------------------------------
        # CNN
        # ------------------------------------------------

        base_channel = 16

        layers = []

        layers.append(
            nn.Conv2d(
                2,
                base_channel,
                kernel_size=(5, 5)
            )
        )

        layers.append(
            nn.Conv2d(
                base_channel,
                base_channel * 2,
                kernel_size=(5, 5)
            )
        )

        layers.append(
            nn.Conv2d(
                base_channel * 2,
                base_channel * 4,
                kernel_size=(5, 5)
            )
        )

        layers.append(
            nn.Conv2d(
                base_channel * 4,
                base_channel * 8,
                kernel_size=(5, 5)
            )
        )

        self.layers = nn.ModuleList(
            layers
        )

        # ------------------------------------------------
        # Initialize CNN
        # ------------------------------------------------

        for layer in self.layers:

            nn.init.xavier_uniform_(
                layer.weight
            )

            nn.init.zeros_(
                layer.bias
            )

        # ------------------------------------------------
        # Activation
        # ------------------------------------------------

        self.LReLU = nn.LeakyReLU(
            negative_slope=0.3
        )

        # ------------------------------------------------
        # Fully Connected
        # ------------------------------------------------

        self.fc1 = nn.Linear(
            base_channel * 8,
            50
        )

        self.fc2 = nn.Linear(
            50,
            10
        )

        self.fc3 = nn.Linear(
            10,
            num_target_metrics
        )

        # ------------------------------------------------
        # Initialize FC
        # ------------------------------------------------

        for layer in [
            self.fc1,
            self.fc2,
            self.fc3
        ]:

            nn.init.xavier_uniform_(
                layer.weight
            )

            nn.init.zeros_(
                layer.bias
            )

    def forward(
        self,
        x
    ):

        # ------------------------------------------------
        # BatchNorm
        # ------------------------------------------------

        x = self.BN(
            x
        )

        # ------------------------------------------------
        # CNN
        # ------------------------------------------------

        for layer in self.layers:

            x = layer(
                x
            )

            x = self.LReLU(
                x
            )

        # ------------------------------------------------
        # Global Average Pooling
        # ------------------------------------------------

        x = torch.mean(
            x,
            dim=(2, 3)
        )

        # ------------------------------------------------
        # Fully Connected
        # ------------------------------------------------

        x = self.fc1(
            x
        )

        x = self.LReLU(
            x
        )

        x = self.fc2(
            x
        )

        x = self.LReLU(
            x
        )

        x = self.fc3(
            x
        )

        return x

# ============================================================
# METRICGAN+ WAVEFORM WRAPPER
# ============================================================

class MetricGANPlusWrapper(
    nn.Module
):

    def __init__(
        self,
        n_fft=512,
        hop_length=128,
        win_length=512,
        num_freqs=257,
        hidden_size=200,
        num_layers=2,
        dropout=0.1,
        causal=False,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_freqs = num_freqs

        # ------------------------------------------------
        # Generator
        # ------------------------------------------------

        self.generator = (
            MetricGANPlusGenerator(
                num_freqs=num_freqs,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                causal=causal,
            )
        )

        # ------------------------------------------------
        # STFT Window
        # ------------------------------------------------

        self.register_buffer(
            "window",
            torch.hann_window(
                win_length
            )
        )

    def forward(
        self,
        wav
    ):

        # ------------------------------------------------
        # Input
        #
        # (B, 1, T)
        # hoặc
        # (B, T)
        # ------------------------------------------------

        if wav.dim() == 3:
            wav = wav.squeeze(1)

        # ------------------------------------------------
        # STFT
        # ------------------------------------------------

        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(
                wav.device
            ),
            return_complex=True,
        )

        # ------------------------------------------------
        # Magnitude
        # ------------------------------------------------

        magnitude = torch.abs(
            spec
        )

        # ------------------------------------------------
        # Phase
        # ------------------------------------------------

        phase = torch.angle(
            spec
        )

        # ------------------------------------------------
        # (B, F, T)
        #
        # ->
        #
        # (B, T, F)
        # ------------------------------------------------

        generator_input = (
            magnitude.transpose(
                1,
                2
            )
        )

        # ------------------------------------------------
        # Generate mask
        # ------------------------------------------------

        mask = self.generator(
            generator_input
        )

        # ------------------------------------------------
        # (B, T, F)
        #
        # ->
        #
        # (B, F, T)
        # ------------------------------------------------

        mask = mask.transpose(
            1,
            2
        )

        # ------------------------------------------------
        # Apply mask
        # ------------------------------------------------

        enhanced_magnitude = (
            magnitude
            * mask
        )

        # ------------------------------------------------
        # Reconstruct complex STFT
        # ------------------------------------------------

        enhanced_spec = (
            enhanced_magnitude
            * torch.exp(
                1j * phase
            )
        )

        # ------------------------------------------------
        # ISTFT
        # ------------------------------------------------

        enhanced_wav = torch.istft(
            enhanced_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(
                wav.device
            ),
            length=wav.shape[-1],
        )

        # ------------------------------------------------
        # Return
        #
        # (B, 1, T)
        # ------------------------------------------------

        return enhanced_wav.unsqueeze(
            1
        )


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

def build_model(cfg: dict) -> nn.Module:

    model_cfg = cfg["model"]
    stft_cfg = cfg["stft"]

    name = model_cfg.get("name", "FullSubNet")

    if name == "FullSubNet":
        model_cfg = model_cfg["FullSubNet"]

    elif name == "FullSubNetPlus" or name == "FullSubNet+":
        model_cfg = model_cfg.get(
            "FullSubNet+",
            model_cfg.get("FullSubNetPlus"),
        )

    elif name == "InterSubNet":
        model_cfg = model_cfg["InterSubNet"]

    elif name == "Conv-TasNet":
        model_cfg = model_cfg["Conv-TasNet"]

    elif name == "MetricGAN+" or name == "MetricGANPlus":
        model_cfg = model_cfg.get(
            "MetricGAN+",
            model_cfg.get("MetricGANPlus"),
        )

    elif name == "CRN":
        model_cfg = model_cfg

    else:
        raise ValueError(f"Unknown model name: {name}")

    # ========================================================
    # CRN
    # ========================================================

    if name == "CRN":

        return CRN(
            n_fft=stft_cfg["n_fft"],
            hop_length=stft_cfg["hop_length"],
            win_length=stft_cfg["win_length"],

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

            n_fft=stft_cfg["n_fft"],

            hop_length=stft_cfg["hop_length"],

            win_length=stft_cfg["win_length"],

            num_freqs=model_cfg["num_freqs"],

            look_ahead=model_cfg["look_ahead"],

            sequence_model=model_cfg["sequence_model"],

            fb_num_neighbors=model_cfg["fb_num_neighbors"],

            sb_num_neighbors=model_cfg["sb_num_neighbors"],

            fb_output_activate_function=(
                model_cfg["fb_output_activate_function"]
            ),

            sb_output_activate_function=(
                model_cfg["sb_output_activate_function"]
            ),

            fb_model_hidden_size=(
                model_cfg["fb_model_hidden_size"]
            ),

            sb_model_hidden_size=(
                model_cfg["sb_model_hidden_size"]
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

            weight_init=False,
        )

    # ========================================================
    # FULLSUBNET+
    # ========================================================

    elif name == "FullSubNetPlus" or name == "FullSubNet+":

        return FullSubNetPlusWrapper(

            n_fft=stft_cfg["n_fft"],

            hop_length=stft_cfg["hop_length"],

            win_length=stft_cfg["win_length"],

            num_freqs=model_cfg["num_freqs"],

            look_ahead=model_cfg["look_ahead"],

            sequence_model=model_cfg["sequence_model"],

            fb_num_neighbors=model_cfg["fb_num_neighbors"],

            sb_num_neighbors=model_cfg["sb_num_neighbors"],

            fb_output_activate_function=(
                model_cfg["fb_output_activate_function"]
            ),

            sb_output_activate_function=(
                model_cfg["sb_output_activate_function"]
            ),

            fb_model_hidden_size=(
                model_cfg["fb_model_hidden_size"]
            ),

            sb_model_hidden_size=(
                model_cfg["sb_model_hidden_size"]
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

            channel_attention_model=model_cfg.get(
                "channel_attention_model",
                "TSSE",
            ),

            output_size=model_cfg.get(
                "output_size",
                2,
            ),

            subband_num=model_cfg.get(
                "subband_num",
                1,
            ),

            kersize=tuple(
                model_cfg.get(
                    "kersize",
                    (3, 5, 10),
                )
            ),

            weight_init=False,
        )
    
    # ========================================================
    # INTERSUBNET
    # ========================================================

    elif name == "InterSubNet":
        return InterSubNetWrapper(

            n_fft=stft_cfg["n_fft"],

            hop_length=stft_cfg["hop_length"],

            win_length=stft_cfg["win_length"],

            num_freqs=model_cfg["num_freqs"],

            look_ahead=model_cfg["look_ahead"],

            sequence_model=model_cfg["sequence_model"],

            sb_num_neighbors=model_cfg["sb_num_neighbors"],

            sb_output_activate_function=(
                model_cfg["sb_output_activate_function"]
            ),

            sb_model_hidden_size=(
                model_cfg["sb_model_hidden_size"]
            ),

            sil_hidden_sizes=tuple(
                model_cfg.get(
                    "sil_hidden_sizes",
                    (93, 307),
                )
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

            weight_init=False,
        )

    # ========================================================
    # CONV-TASNET
    # ========================================================

    elif name in ["ConvTasNet", "Conv-TasNet"]:

        return ConvTasNetWrapper(

            N=model_cfg.get("N", 512),

            L=model_cfg.get("L", 16),

            B=model_cfg.get("B", 128),

            H=model_cfg.get("H", 512),

            P=model_cfg.get("P", 3),

            X=model_cfg.get("X", 8),

            R=model_cfg.get("R", 3),

            norm=model_cfg.get("norm_type", "gLN"),

            num_spks=model_cfg.get("num_spks", 1),

            activate=model_cfg.get("activate", "relu"),

            causal=model_cfg.get("causal", False),
        )

    # ========================================================
    # METRICGAN+
    # ========================================================

    elif name in ["MetricGAN+", "MetricGANPlus"]:
        return MetricGANPlusWrapper(
            n_fft=stft_cfg["n_fft"],
            hop_length=stft_cfg["hop_length"],
            win_length=stft_cfg["win_length"],
            num_freqs=model_cfg.get("num_freqs", 257),
            hidden_size=model_cfg.get("hidden_size", 200),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.0),
            causal=model_cfg.get("causal", False),
        )
        
        return MetricGANPlusWrapper(

            n_fft=stft_cfg["n_fft"],

            hop_length=stft_cfg["hop_length"],

            win_length=stft_cfg["win_length"],

            num_freqs=model_cfg["num_freqs"],

            look_ahead=model_cfg["look_ahead"],

            sequence_model=model_cfg["sequence_model"],

            sb_num_neighbors=model_cfg["sb_num_neighbors"],

            sb_output_activate_function=(
                model_cfg["sb_output_activate_function"]
            ),

            sb_model_hidden_size=(
                model_cfg["sb_model_hidden_size"]
            ),

            sil_hidden_sizes=tuple(
                model_cfg.get(
                    "sil_hidden_sizes",
                    (93, 307),
                )
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

            weight_init=False,
        )

    # ========================================================
    # UNKNOWN MODEL
    # ========================================================

    else:

        raise ValueError(
            f"Unknown model name: {name}"
        )