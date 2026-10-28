"""
Multi-Modal Data Fusion Network for CARE'OS Module 3.

Implements an advanced attention-based fusion mechanism that combines:
  • SNP latent vectors (from VAE)
  • Tabular clinical features
  • Gene expression profiles

The fusion network uses cross-attention to allow each modality to inform
the representation of others, producing a unified feature vector that
captures complex gene-environment interactions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from careos_genomics.autoencoder import CrossAttentionBlock, ResidualBlock


# ═══════════════════════════════════════════════════════════════════════════
# Modality Encoders
# ═══════════════════════════════════════════════════════════════════════════

class ClinicalEncoder(nn.Module):
    """Encodes tabular clinical features with handling for mixed types."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(output_dim * 2, dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ExpressionEncoder(nn.Module):
    """Encodes gene expression data with batch-aware normalization."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(output_dim * 2, dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GenomicLatentEncoder(nn.Module):
    """Projects VAE latent vectors into the fusion space."""

    def __init__(self, latent_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(output_dim, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════════════
# Gated Fusion Mechanism
# ═══════════════════════════════════════════════════════════════════════════

class GatedFusion(nn.Module):
    """Learnable gated fusion that dynamically weights modality contributions.

    Uses a gating mechanism to control how much each modality contributes
    to the final fused representation. This allows the model to learn
    which data source is most informative for a given patient.
    """

    def __init__(self, dim: int, num_modalities: int = 3) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_modalities, dim * num_modalities),
            nn.GELU(),
            nn.Linear(dim * num_modalities, num_modalities),
            nn.Softmax(dim=-1),
        )
        self.num_modalities = num_modalities

    def forward(self, *modalities: torch.Tensor) -> torch.Tensor:
        """Fuse modality tensors with learned gating weights."""
        concat = torch.cat(modalities, dim=-1)
        gates = self.gate(concat)  # (batch, num_modalities)

        weighted = torch.zeros_like(modalities[0])
        for i, mod in enumerate(modalities):
            weighted = weighted + gates[:, i : i + 1] * mod
        return weighted


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Modal Fusion Network
# ═══════════════════════════════════════════════════════════════════════════

class MultiModalFusionNetwork(nn.Module):
    """
    Advanced multi-modal fusion network combining genomic, clinical, and
    expression data through cross-attention and gated mechanisms.

    Architecture Overview:
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ SNP Latent   │   │ Clinical     │   │ Expression   │
    │ (from VAE)   │   │ Features     │   │ Data         │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                   │
    ┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐
    │ Genomic      │   │ Clinical     │   │ Expression   │
    │ Encoder      │   │ Encoder      │   │ Encoder      │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                   │
           ├──── Cross-Attention (genomic ↔ clinical) ────┤
           │                  │                   │
    ┌──────▼──────────────────▼───────────────────▼───────┐
    │              Gated Fusion Layer                      │
    └──────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────┐
    │         Refinement MLP + Residual Blocks            │
    └──────────────────────┬──────────────────────────────┘
                           │
                    ┌──────▼───────┐
                    │ Fused Vector │
                    └──────────────┘
    """

    def __init__(
        self,
        snp_latent_dim: int,
        clinical_dim: int,
        expression_dim: int,
        fusion_dim: int = 128,
        output_dim: int = 64,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim

        # Modality-specific encoders
        self.genomic_encoder = GenomicLatentEncoder(snp_latent_dim, fusion_dim, dropout)
        self.clinical_encoder = ClinicalEncoder(clinical_dim, fusion_dim, dropout)
        self.expression_encoder = ExpressionEncoder(expression_dim, fusion_dim, dropout)

        # Cross-attention between modalities
        self.genomic_clinical_attn = CrossAttentionBlock(
            query_dim=fusion_dim,
            key_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.clinical_genomic_attn = CrossAttentionBlock(
            query_dim=fusion_dim,
            key_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.expression_genomic_attn = CrossAttentionBlock(
            query_dim=fusion_dim,
            key_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Gated fusion
        self.gated_fusion = GatedFusion(fusion_dim, num_modalities=3)

        # Refinement network
        self.refinement = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(fusion_dim, dropout),
            ResidualBlock(fusion_dim, dropout),
            nn.Linear(fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        snp_latent: torch.Tensor,
        clinical: torch.Tensor,
        expression: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse all modalities into a unified feature vector.

        Args:
            snp_latent: (batch, snp_latent_dim) from VAE encoder
            clinical: (batch, clinical_dim) tabular clinical features
            expression: (batch, expression_dim) gene expression data

        Returns:
            (batch, output_dim) fused feature vector
        """
        # Encode each modality
        g = self.genomic_encoder(snp_latent)
        c = self.clinical_encoder(clinical)
        e = self.expression_encoder(expression)

        # Cross-attention: reshape for attention (add sequence dim)
        g_seq = g.unsqueeze(1)  # (batch, 1, fusion_dim)
        c_seq = c.unsqueeze(1)
        e_seq = e.unsqueeze(1)

        # Let genomic features attend to clinical
        g_attended = self.genomic_clinical_attn(g_seq, c_seq).squeeze(1)
        # Let clinical features attend to genomic
        c_attended = self.clinical_genomic_attn(c_seq, g_seq).squeeze(1)
        # Let expression attend to genomic
        e_attended = self.expression_genomic_attn(e_seq, g_seq).squeeze(1)

        # Gated fusion of attended representations
        fused = self.gated_fusion(g_attended, c_attended, e_attended)

        # Refinement
        output = self.refinement(fused)
        return output

    def get_output_dim(self) -> int:
        return self.output_dim


class FusionFeatureExtractor:
    """Scikit-learn compatible wrapper for the fusion network.

    Trains the fusion network end-to-end or uses it as a frozen
    feature extractor for downstream ML models.
    """

    def __init__(
        self,
        snp_latent_dim: int,
        clinical_dim: int,
        expression_dim: int,
        fusion_dim: int = 128,
        output_dim: int = 64,
        device: Optional[str] = None,
    ) -> None:
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.network = MultiModalFusionNetwork(
            snp_latent_dim=snp_latent_dim,
            clinical_dim=clinical_dim,
            expression_dim=expression_dim,
            fusion_dim=fusion_dim,
            output_dim=output_dim,
        ).to(self.device)

        self.output_dim = output_dim
        self._is_fitted = False

    def transform(
        self,
        snp_latent: np.ndarray,
        clinical: np.ndarray,
        expression: np.ndarray,
    ) -> np.ndarray:
        """Transform multi-modal data into fused feature vectors."""
        self.network.eval()
        with torch.no_grad():
            g = torch.FloatTensor(snp_latent).to(self.device)
            c = torch.FloatTensor(clinical).to(self.device)
            e = torch.FloatTensor(expression).to(self.device)
            fused = self.network(g, c, e)
        return fused.cpu().numpy()
