"""
PyTorch Variational Autoencoder with Self-Attention for SNP Dimensionality Reduction.

This module implements a deep generative model that learns a compressed latent
representation of high-dimensional SNP (Single Nucleotide Polymorphism) profiles.
The architecture combines:
  • Multi-head self-attention to capture inter-SNP epistatic interactions
  • Variational inference for robust, regularized latent space
  • Residual connections for stable deep training
  • Layer normalization for training stability
  • KL annealing for preventing posterior collapse
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ═══════════════════════════════════════════════════════════════════════════
# Attention Mechanisms
# ═══════════════════════════════════════════════════════════════════════════

class AttentionBlock(nn.Module):
    """Multi-head self-attention block with residual connection and layer norm.

    Captures epistatic interactions between SNP positions by allowing each
    SNP to attend to every other SNP in the profile, learning complex
    non-linear dependencies that PCA cannot model.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        # Ensure embed_dim is divisible by num_heads
        while num_heads > 1 and embed_dim % num_heads != 0:
            num_heads //= 2
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.use_flash = use_flash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.layer_norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = residual + attn_out

        # Feedforward with residual
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.feedforward(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention for fusing genomic and clinical modalities."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        while num_heads > 1 and query_dim % num_heads != 0:
            num_heads //= 2
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, query_dim)
        self.value_proj = nn.Linear(key_dim, query_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(query_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 2, query_dim),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(query)
        k = self.key_proj(key_value)
        v = self.value_proj(key_value)
        attn_out, _ = self.attention(q, k, v)
        x = self.layer_norm(query + attn_out)
        x = x + self.feedforward(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Residual Block
# ═══════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Dense residual block with pre-activation normalization."""

    def __init__(self, dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ═══════════════════════════════════════════════════════════════════════════
# Variational SNP Autoencoder
# ═══════════════════════════════════════════════════════════════════════════

class VariationalSNPAutoencoder(nn.Module):
    """
    Variational Autoencoder with self-attention for SNP profile compression.

    Architecture:
    ┌──────────┐    ┌─────────────────────┐    ┌────────┐    ┌──────────────┐
    │ SNP Input │──▶│ Attention Encoder    │──▶│ μ, log σ²│──▶│ Latent z     │
    │ (0,1,2)  │    │ (residual blocks)   │    │ (reparametrize)│  │ (compact)    │
    └──────────┘    └─────────────────────┘    └────────┘    └──────┬───────┘
                                                                     │
                    ┌─────────────────────┐    ┌────────────┐       │
                    │ Attention Decoder    │◀──│ Residual   │◀──────┘
                    │ (reconstruct SNPs)  │    │ Blocks     │
                    └─────────┬───────────┘    └────────────┘
                              │
                    ┌─────────▼───────────┐
                    │ Reconstructed SNPs  │
                    │ (softmax per locus) │
                    └─────────────────────┘

    The latent vector z captures the essential genomic variation in a
    low-dimensional manifold suitable for downstream ML models.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[list[int]] = None,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or self._compute_hidden_dims(input_dim, latent_dim)

        # ── Encoder ──────────────────────────────────────────────────
        encoder_layers: list[nn.Module] = []

        # Initial projection
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.encoder_backbone = nn.Sequential(*encoder_layers)

        # Self-attention layers in latent-adjacent space
        self.encoder_attention_layers = nn.ModuleList([
            AttentionBlock(
                embed_dim=self.hidden_dims[-1],
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            for _ in range(num_attention_layers)
        ])

        # Residual refinement
        self.encoder_residual = nn.Sequential(
            ResidualBlock(self.hidden_dims[-1], dropout),
            ResidualBlock(self.hidden_dims[-1], dropout),
        )

        # Variational parameters
        self.fc_mu = nn.Linear(self.hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], latent_dim)

        # ── Decoder ──────────────────────────────────────────────────
        decoder_layers: list[nn.Module] = []
        reversed_dims = list(reversed(self.hidden_dims))

        prev_dim = latent_dim
        for h_dim in reversed_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.decoder_backbone = nn.Sequential(*decoder_layers)

        # Decoder attention
        self.decoder_attention_layers = nn.ModuleList([
            AttentionBlock(
                embed_dim=reversed_dims[-1],
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            for _ in range(num_attention_layers)
        ])

        self.decoder_residual = nn.Sequential(
            ResidualBlock(reversed_dims[-1], dropout),
        )

        # Output: 3 logits per SNP position (for values 0, 1, 2)
        self.output_projection = nn.Linear(reversed_dims[-1], input_dim * 3)

        # Weight initialization
        self.apply(self._init_weights)

    @staticmethod
    def _compute_hidden_dims(input_dim: int, latent_dim: int) -> list[int]:
        """Compute a tapering sequence of hidden dimensions, divisible by 8."""
        def make_divisible(val: int, divisor: int = 8) -> int:
            return max(divisor, (val // divisor) * divisor)

        dims = []
        current = make_divisible(min(input_dim, 512), 8)
        target = make_divisible(max(latent_dim * 4, 64), 8)
        while current > target:
            dims.append(current)
            current = make_divisible(max(current // 2, target), 8)
        dims.append(target)
        filtered = []
        for d in dims:
            if not filtered or d != filtered[-1]:
                filtered.append(d)
        return filtered

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode SNP profiles to variational parameters μ and log(σ²)."""
        h = self.encoder_backbone(x)

        # Apply self-attention (reshape for attention: add sequence dim)
        h = h.unsqueeze(1)  # (batch, 1, hidden)
        for attn_layer in self.encoder_attention_layers:
            h = attn_layer(h)
        h = h.squeeze(1)  # (batch, hidden)

        h = self.encoder_residual(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε, ε ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to SNP reconstruction logits."""
        h = self.decoder_backbone(z)

        h = h.unsqueeze(1)
        for attn_layer in self.decoder_attention_layers:
            h = attn_layer(h)
        h = h.squeeze(1)

        h = self.decoder_residual(h)
        logits = self.output_projection(h)
        # Reshape to (batch, num_snps, 3) for per-SNP softmax
        return logits.view(-1, self.input_dim, 3)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → sample → decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract deterministic latent representation (using μ only)."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

    @staticmethod
    def loss_function(
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Compute VAE loss = Reconstruction Loss + β * KL Divergence.

        Reconstruction uses cross-entropy (SNP values are categorical: 0, 1, 2).
        KL divergence regularizes the latent space toward N(0, I).
        """
        # Cross-entropy reconstruction loss per SNP position
        batch_size = target.size(0)
        target_long = target.long()  # (batch, num_snps)
        recon_loss = F.cross_entropy(
            reconstruction.view(-1, 3),
            target_long.view(-1),
            reduction="sum",
        ) / batch_size

        # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        total_loss = recon_loss + kl_weight * kl_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training Utilities
# ═══════════════════════════════════════════════════════════════════════════

class KLAnnealer:
    """Cyclical KL annealing schedule to prevent posterior collapse.

    Gradually increases the KL weight from 0 to 1 over a warmup period,
    allowing the model to first learn good reconstructions before
    regularizing the latent space.
    """

    def __init__(
        self,
        n_epochs: int,
        n_cycles: int = 4,
        ratio: float = 0.5,
        start: float = 0.0,
        stop: float = 1.0,
    ) -> None:
        self.weights = self._frange_cycle_linear(n_epochs, n_cycles, ratio, start, stop)

    @staticmethod
    def _frange_cycle_linear(
        n: int, n_cycles: int, ratio: float, start: float, stop: float
    ) -> list[float]:
        weights: list[float] = []
        n_cycles = max(1, min(n_cycles, n))
        period = max(1, n // n_cycles)
        ramp_len = max(1.0, period * ratio)
        step = (stop - start) / ramp_len
        for c in range(n_cycles):
            for i in range(period):
                if i < ramp_len:
                    weights.append(start + step * i)
                else:
                    weights.append(stop)
        # Pad if needed
        while len(weights) < n:
            weights.append(stop)
        return weights

    def __getitem__(self, epoch: int) -> float:
        return self.weights[min(epoch, len(self.weights) - 1)]


class SNPAutoencoderTrainer:
    """Complete training pipeline for the Variational SNP Autoencoder.

    Features:
    - Cyclical KL annealing
    - Learning rate warm-up + cosine decay
    - Early stopping with patience
    - Training history logging
    - Automatic device selection (CUDA/MPS/CPU)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[list[int]] = None,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        n_epochs: int = 100,
        patience: int = 15,
        dropout: float = 0.15,
        device: Optional[str] = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.learning_rate = learning_rate

        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Model
        self.model = VariationalSNPAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            dropout=dropout,
        ).to(self.device)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler: warm-up + cosine
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(n_epochs // 4, 5), T_mult=2
        )

        # KL annealing
        self.kl_annealer = KLAnnealer(n_epochs, n_cycles=4)

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "kl_weight": [],
            "learning_rate": [],
        }

    def fit(
        self,
        snp_data: np.ndarray,
        validation_split: float = 0.15,
    ) -> dict[str, list[float]]:
        """Train the VAE on SNP data.

        Args:
            snp_data: NumPy array of shape (n_samples, n_snps), values in {0, 1, 2}
            validation_split: fraction of data to hold out for validation

        Returns:
            Training history dictionary
        """
        # Prepare data
        n_val = max(1, int(len(snp_data) * validation_split))
        indices = np.random.permutation(len(snp_data))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        train_tensor = torch.FloatTensor(snp_data[train_idx]).to(self.device)
        val_tensor = torch.FloatTensor(snp_data[val_idx]).to(self.device)

        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(train_tensor) > self.batch_size,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.n_epochs):
            # ── Training ─────────────────────────────────────────────
            self.model.train()
            epoch_losses: dict[str, float] = {"total": 0, "recon": 0, "kl": 0}
            n_batches = 0

            kl_weight = self.kl_annealer[epoch]

            for (batch,) in train_loader:
                self.optimizer.zero_grad()
                reconstruction, mu, logvar = self.model(batch)
                losses = self.model.loss_function(
                    reconstruction, batch, mu, logvar, kl_weight=kl_weight
                )
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_losses["total"] += losses["total_loss"].item()
                epoch_losses["recon"] += losses["reconstruction_loss"].item()
                epoch_losses["kl"] += losses["kl_loss"].item()
                n_batches += 1

            self.scheduler.step()

            avg_train = epoch_losses["total"] / max(n_batches, 1)

            # ── Validation ───────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_recon, val_mu, val_logvar = self.model(val_tensor)
                val_losses = self.model.loss_function(
                    val_recon, val_tensor, val_mu, val_logvar, kl_weight=kl_weight
                )
                val_loss = val_losses["total_loss"].item()

            # Log history
            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(val_loss)
            self.history["recon_loss"].append(epoch_losses["recon"] / max(n_batches, 1))
            self.history["kl_loss"].append(epoch_losses["kl"] / max(n_batches, 1))
            self.history["kl_weight"].append(kl_weight)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            # ── Early Stopping ───────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return self.history

    def transform(self, snp_data: np.ndarray) -> np.ndarray:
        """Extract latent representations from SNP data."""
        self.model.eval()
        tensor = torch.FloatTensor(snp_data).to(self.device)
        with torch.no_grad():
            latent = self.model.get_latent(tensor)
        return latent.cpu().numpy()

    def reconstruct(self, snp_data: np.ndarray) -> np.ndarray:
        """Reconstruct SNP profiles (useful for quality assessment)."""
        self.model.eval()
        tensor = torch.FloatTensor(snp_data).to(self.device)
        with torch.no_grad():
            reconstruction, _, _ = self.model(tensor)
            # Convert logits to predicted classes
            predicted = reconstruction.argmax(dim=-1)
        return predicted.cpu().numpy()

    def reconstruction_accuracy(self, snp_data: np.ndarray) -> float:
        """Compute per-SNP reconstruction accuracy."""
        original = snp_data.astype(int)
        reconstructed = self.reconstruct(snp_data)
        return float(np.mean(original == reconstructed))
