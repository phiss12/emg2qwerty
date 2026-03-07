# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
import math

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """Applies 2D batch normalization over spectrogram per electrode channel per band.

    Inputs must be of shape (T, N, num_bands, electrode_channels, frequency_bins).
    Stats are computed over (N, freq, time) slices per channel.

    Args:
        channels: Total electrode channels across bands (num_bands * electrode_channels).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)           # (N, bands, C, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)             # (T, N, bands, C, freq)


class RotationInvariantMLP(nn.Module):
    """Applies an MLP over rotated electrode offsets and pools results.

    For a single-band input of shape (T, N, C, ...), shifts electrode channels
    by each offset in ``offsets``, passes all rotations through a shared MLP,
    then pools over the rotation dimension.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features: Flattened input size from the channel dim onwards (C * ...).
        mlp_features: Number of out_features per MLP layer.
        pooling: 'mean' or 'max' pooling over rotations. (default: 'mean')
        offsets: Electrode shift offsets. (default: (-1, 0, 1))
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend([nn.Linear(in_features, out_features), nn.ReLU()])
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling
        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (T, N, C, ...) -> stack rotations -> (T, N, rotation, C, ...)
        x = torch.stack([inputs.roll(o, dims=2) for o in self.offsets], dim=2)
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))
        return x.max(dim=2).values if self.pooling == "max" else x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """Applies a separate `RotationInvariantMLP` per band.

    Input shape: (T, N, num_bands, electrode_channels, ...)
    Output shape: (T, N, num_bands, mlp_features[-1])

    Args:
        in_features: Flattened size from channel dim onwards (C * ...).
        mlp_features: Number of out_features per MLP layer.
        pooling: 'mean' or 'max' pooling over rotations. (default: 'mean')
        offsets: Electrode shift offsets. (default: (-1, 0, 1))
        num_bands: Number of bands. (default: 2)
        stack_dim: Dimension along which bands are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands
        outputs = [mlp(x) for mlp, x in zip(self.mlps, inputs.unbind(self.stack_dim))]
        return torch.stack(outputs, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """2D temporal convolution block from TDS (Hannun et al., arxiv.org/abs/1904.02619).

    Args:
        channels: Number of input/output channels. Invariant: channels * width == num_features.
        width: Input width. Invariant: channels * width == num_features.
        kernel_width: Kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.conv2d = nn.Conv2d(channels, channels, kernel_size=(1, kernel_width))
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.relu(self.conv2d(x))
        x = x.reshape(N, C, -1).movedim(-1, 0)     # (T_out, N, C)
        T_out = x.shape[0]
        return self.layer_norm(x + inputs[-T_out:])  # skip + norm


class TDSFullyConnectedBlock(nn.Module):
    """Fully connected residual block from TDS (Hannun et al., arxiv.org/abs/1904.02619).

    Args:
        num_features: Feature size for input of shape (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.fc_block(inputs) + inputs)


class TDSConvEncoder(nn.Module):
    """TDS convolutional encoder composing TDSConv2dBlock + TDSFullyConnectedBlock pairs.

    Args:
        num_features: Feature size for input of shape (T, N, num_features).
        block_channels: Number of channels per TDSConv2dBlock.
        kernel_width: Kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        assert len(block_channels) > 0
        blocks: list[nn.Module] = []
        for channels in block_channels:
            assert num_features % channels == 0, \
                "block_channels must evenly divide num_features"
            blocks.extend([
                TDSConv2dBlock(channels, num_features // channels, kernel_width),
                TDSFullyConnectedBlock(num_features),
            ])
        self.tds_conv_blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in 'Attention Is All You Need'.

    Args:
        d_model: Model embedding dimension.
        dropout: Dropout probability applied after adding positional encodings.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, d_model)
        x = x + self.pe[: x.size(0)].unsqueeze(1)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder with optional sinusoidal positional encoding.

    Expects (T, N, d_model) and returns (T, N, d_model).

    Args:
        d_model: Model embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of TransformerEncoderLayer stacks.
        dim_feedforward: Inner dimension of the feedforward sublayer.
        dropout: Dropout probability.
        use_positional_encoding: Whether to apply sinusoidal PE. (default: True)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.pos = (
            SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)
            if use_positional_encoding
            else nn.Identity()
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=True,    # pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (T, N, d_model)
        x = self.pos(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.padding = padding
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed entire session at once at test time for realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def _make_dataloader(self, dataset, *, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        # batch_size=1: entire sessions fed at once; avoid padding influence on scores
        return self._make_dataloader(self.test_dataset, batch_size=1, shuffle=False)


# ---------------------------------------------------------------------------
# Shared mixin for step/epoch logic
# ---------------------------------------------------------------------------

class _CTCMixin:
    """Shared _step / _epoch_end logic for CTC-based lightning modules.

    Subclasses must define:
        self.ctc_loss
        self.decoder
        self.metrics
        self.forward(inputs, input_lengths=None) -> log_probs (T, N, C)
    and must implement configure_optimizers().
    """

    def _build_emissions(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass and return (emissions, emission_lengths)."""
        raise NotImplementedError

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions, emission_lengths = self._build_emissions(inputs, input_lengths)

        loss = self.ctc_loss(
            log_probs=emissions,                 # (T, N, num_classes)
            targets=targets.transpose(0, 1),     # (S, N) -> (N, S)
            input_lengths=emission_lengths,      # (N,)
            target_lengths=target_lengths,       # (N,)
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")


# ---------------------------------------------------------------------------
# TDS Conv CTC model
# ---------------------------------------------------------------------------

class TDSConvCTCModule(_CTCMixin, pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None) -> torch.Tensor:
        # input_lengths unused here — TDS uses T_diff from conv receptive field
        return self.model(inputs)

    def _build_emissions(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emissions = self.forward(inputs)
        # Shrink lengths by temporal receptive field consumed by conv blocks
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff
        return emissions, emission_lengths

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# ---------------------------------------------------------------------------
# Transformer CTC model
# ---------------------------------------------------------------------------

class TransformerCTCModule(_CTCMixin, pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        d_model = self.NUM_BANDS * mlp_features[-1]

        # Explicit submodules (not nn.Sequential) so we can pass padding mask
        self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_positional_encoding=True,
        )
        self.classifier = nn.Linear(d_model, charset().num_classes)

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.spec_norm(inputs)   # (T, N, bands, C, freq)
        x = self.mlp(x)              # (T, N, bands, mlp_features[-1])
        x = self.flatten(x)          # (T, N, d_model)

        # Build padding mask: (N, T), True at positions that are padding
        mask = None
        if input_lengths is not None:
            T, N = x.shape[:2]
            mask = (
                torch.arange(T, device=x.device)
                .unsqueeze(0)
                .expand(N, -1)
                >= input_lengths.unsqueeze(1)
            )

        x = self.encoder(x, src_key_padding_mask=mask)  # (T, N, d_model)
        return F.log_softmax(self.classifier(x), dim=-1) # (T, N, num_classes)

    def _build_emissions(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emissions = self.forward(inputs, input_lengths=input_lengths)
        # Transformer preserves T — no temporal downsampling
        return emissions, input_lengths

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
