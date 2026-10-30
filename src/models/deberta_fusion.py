"""
DeBERTa + Stylometric Fusion for Style Change Detection.

Modes:
  1. Pair-level (--finetune): Independent pairwise classification
  2. Document-level (--finetune --bilstm): BiLSTM over boundary sequence
  3. Document-level + adversarial (--finetune --bilstm --adversarial):
     adds a pseudo-topic discriminator with gradient reversal to
     disentangle topic signal from the learned boundary representation.
  4. Document-level + contrastive (--finetune --bilstm --contrastive):
     adds a supervised contrastive (topic-decoupling) loss on the fused
     representation, pre-BiLSTM -- pulls together same-label boundary
     representations and pushes apart differently-labeled ones,
     regardless of topic. A different (and per SCL-DeBERTa's published
     result, more successful) approach to the same goal as the GRL
     adversarial branch above.

Tier B additions (all independent, own-work techniques motivated by
failure modes diagnosed earlier in this project):
  - SelfAdjustingDiceLoss : loss-level fix for Hard-tier class imbalance
  - FiLMFusion            : affine (scale+shift) alternative to the
                            multiplicative-gate-only StyleAttentionFusion
  - TransformerSeqEncoder : self-attention stacked on top of the BiLSTM
                            output, so any boundary can attend to any
                            other boundary in the document directly
  - XBMQueue              : cross-batch memory for the contrastive loss,
                            fixing the too-few-pairs-per-step problem
  - rdrop_kl_loss         : R-Drop consistency regularization

Architecture (document-level):
    For each document:
      - Encode all sentence pairs through DeBERTa → [CLS] hidden states
      - Attention-weighted (or FiLM) fusion with stylometric diffs
      - (optional) Supervised contrastive loss on this fused representation
      - Project to lower dimension
      - BiLSTM over the boundary sequence → document context
      - (optional) Transformer encoder over the BiLSTM output
      - MLP head → per-boundary logits
      - (optional) Gradient-reversed topic classifier on the BiLSTM output
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch.utils.data import Dataset


def load_state_dict_flexible(model, state_dict, verbose=True):
    """
    Load a checkpoint into `model`, skipping any key whose shape doesn't
    match the model's own parameter shape (instead of crashing, which is
    what strict=False alone does on a shape mismatch).

    Returns the list of skipped key names.
    """
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, val in state_dict.items():
        if key in model_state and model_state[key].shape == val.shape:
            compatible[key] = val
        else:
            skipped.append(key)

    model.load_state_dict(compatible, strict=False)

    if verbose:
        print(f"[load_state_dict_flexible] loaded {len(compatible)}/{len(state_dict)} keys")
        if skipped:
            print(f"[load_state_dict_flexible] skipped (shape mismatch or new, "
                 f"will use fresh init): {skipped}")
    return skipped


# --------------------------------------------------------------------------- #
#  Attention-Weighted Stylometric Fusion
# --------------------------------------------------------------------------- #

class StyleAttentionFusion(nn.Module):
    def __init__(self, deberta_dim: int, style_dim: int):
        super().__init__()
        self.style_norm = nn.LayerNorm(style_dim)
        self.gate = nn.Sequential(
            nn.Linear(deberta_dim, style_dim),
            nn.Tanh(),
            nn.Linear(style_dim, style_dim),
            nn.Sigmoid(),
        )

    def forward(self, cls_hidden: torch.Tensor, style_diff: torch.Tensor) -> torch.Tensor:
        style_normed = self.style_norm(style_diff)
        attn_weights = self.gate(cls_hidden)
        weighted_style = style_normed * attn_weights
        return torch.cat([cls_hidden, weighted_style], dim=-1)


# --------------------------------------------------------------------------- #
#  Gradient Reversal Layer (for pseudo-topic adversarial training)
# --------------------------------------------------------------------------- #

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class TopicAdversary(nn.Module):
    """Small MLP that tries to predict pseudo-topic from the boundary
    representation. Gradient reversal forces the encoder to NOT carry
    topic information forward, only style information."""

    def __init__(self, input_dim: int, n_topics: int = 20, hidden: int = 64):
        super().__init__()
        self.n_topics = n_topics
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_topics),
        )

    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        reversed_x = GradientReversal.apply(x, lambda_)
        return self.classifier(reversed_x)


# --------------------------------------------------------------------------- #
#  Supervised Contrastive Loss (topic-decoupling, applied pre-BiLSTM)
# --------------------------------------------------------------------------- #

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020), adapted for the
    binary style-change label instead of multi-class labels.

    Pulls together boundary representations sharing the same style-change
    label (change vs no-change) and pushes apart differently-labeled ones,
    REGARDLESS of topic -- this is the topic-decoupling mechanism
    SCL-DeBERTa uses. Unlike the GRL adversarial branch above (already
    reported as a negative result in this project), this doesn't need
    pseudo-topic labels at all; it directly rewards label-consistent
    clustering.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (N, D) L2-normalized embeddings
        labels:   (N,) binary labels (0/1)
        """
        device = features.device
        n = features.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=device)

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(n, device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = mask.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
        loss = -mean_log_prob_pos.mean()
        return loss


# --------------------------------------------------------------------------- #
#  Tier B additions
# --------------------------------------------------------------------------- #

class SelfAdjustingDiceLoss(nn.Module):
    """Dice loss (Li et al. 2020) for imbalanced binary classification.
    Hard tier is ~17% positive -- this directly targets that imbalance in
    the loss itself, weighting down easy well-classified examples via
    (1-p_t)^alpha, unlike pos_weight in BCE which just reweights uniformly."""
    def __init__(self, alpha: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        probs_pos = probs * labels + (1 - probs) * (1 - labels)
        weight = (1 - probs_pos) ** self.alpha
        p = weight * probs
        numerator = 2 * (p * labels).sum() + self.smooth
        denominator = (p ** 2).sum() + (labels ** 2).sum() + self.smooth
        return 1 - numerator / denominator


class FiLMFusion(nn.Module):
    """FiLM (Perez et al. 2018) affine conditioning: the DeBERTa CLS vector
    predicts BOTH a scale (gamma) and shift (beta) applied to the
    stylometric diff vector, instead of StyleAttentionFusion's
    multiplicative-gate-only. Strictly more expressive -- addresses the
    limitation this project found where a gate alone couldn't represent
    inverted small-value-means-signal relationships."""
    def __init__(self, deberta_dim: int, style_dim: int):
        super().__init__()
        self.style_norm = nn.LayerNorm(style_dim)
        self.film_generator = nn.Sequential(
            nn.Linear(deberta_dim, style_dim * 2),
            nn.Tanh(),
            nn.Linear(style_dim * 2, style_dim * 2),
        )
        self.style_dim = style_dim

    def forward(self, cls_hidden: torch.Tensor, style_diff: torch.Tensor) -> torch.Tensor:
        style_normed = self.style_norm(style_diff)
        gamma_beta = self.film_generator(cls_hidden)
        gamma, beta = gamma_beta[:, :self.style_dim], gamma_beta[:, self.style_dim:]
        modulated = gamma * style_normed + beta
        return torch.cat([cls_hidden, modulated], dim=-1)


class TransformerSeqEncoder(nn.Module):
    """Self-attention layer stacked ON TOP of the BiLSTM output. Lets any
    boundary attend directly to any other boundary in the document, not
    only through the recurrent BiLSTM state -- useful since an author's
    block can span many boundaries, not just adjacent ones."""
    def __init__(self, dim: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, dim) * 0.02)
        self.pre_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs in fp32 regardless of the surrounding autocast context --
        the softmax inside self-attention is numerically unstable under
        fp16 when stacked this deep (same class of issue that broke
        R-Drop's KL term)."""
        n = x.shape[1]
        x = x.float()
        x = self.pre_norm(x + self.pos_embed[:, :n, :])
        return self.encoder(x)


class XBMQueue:
    """Cross-Batch Memory (Wang et al. 2020) for the contrastive loss.
    Your own diagnosis was that SupCon's signal was flat/noisy -- each
    document-level "batch" (1 document) gives far too few positive/negative
    pairs per contrastive step. XBM keeps a FIFO queue of recent
    (embedding, label) pairs across many batches so every contrastive step
    sees a much larger, more stable comparison pool without needing a
    bigger GPU batch."""
    def __init__(self, size: int = 2048, dim: int = 128, device="cuda"):
        self.size = size
        self.feats = torch.zeros(size, dim, device=device)
        self.labels = torch.zeros(size, device=device)
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def enqueue(self, feats, labels):
        n = feats.shape[0]
        if n == 0:
            return
        if n >= self.size:
            self.feats[:] = feats[-self.size:]
            self.labels[:] = labels[-self.size:]
            self.ptr = 0
            self.full = True
            return
        end = self.ptr + n
        if end <= self.size:
            self.feats[self.ptr:end] = feats
            self.labels[self.ptr:end] = labels
        else:
            first = self.size - self.ptr
            self.feats[self.ptr:] = feats[:first]
            self.labels[self.ptr:] = labels[:first]
            self.feats[:end - self.size] = feats[first:]
            self.labels[:end - self.size] = labels[first:]
            self.full = True
        self.ptr = end % self.size
        if end >= self.size:
            self.full = True

    def get(self):
        if self.full:
            return self.feats, self.labels
        return self.feats[:self.ptr], self.labels[:self.ptr]


def rdrop_kl_loss(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """R-Drop (Liang et al. 2021): two forward passes of the SAME input
    through the model (different dropout masks each time), penalized by
    symmetric KL between the two predicted distributions. Targets
    overfitting risk of fine-tuning only 2 DeBERTa layers on a few
    thousand Hard-tier documents -- forces predictions to be robust to the
    model's own dropout noise instead of memorizing it.

    Computed in fp32 regardless of the surrounding autocast context --
    under fp16, sigmoid outputs near 0/1 make log() blow up to NaN."""
    logits1 = logits1.float()
    logits2 = logits2.float()
    p1 = torch.sigmoid(logits1).clamp(1e-6, 1 - 1e-6)
    p2 = torch.sigmoid(logits2).clamp(1e-6, 1 - 1e-6)
    kl1 = p1 * (torch.log(p1) - torch.log(p2)) + (1 - p1) * (torch.log(1 - p1) - torch.log(1 - p2))
    kl2 = p2 * (torch.log(p2) - torch.log(p1)) + (1 - p2) * (torch.log(1 - p2) - torch.log(1 - p1))
    return 0.5 * (kl1 + kl2).mean()


# --------------------------------------------------------------------------- #
#  End-to-end fine-tuning model
# --------------------------------------------------------------------------- #

class DeBERTaFusionE2E(nn.Module):
    DEBERTA_MODEL = "microsoft/deberta-v3-small"
    DEBERTA_DIM = 768

    def __init__(
        self,
        style_dim: int = 193,
        unfreeze_layers: int = 2,
        use_style: bool = True,
        use_attention_fusion: bool = False,
        use_bilstm: bool = False,
        bilstm_hidden: int = 128,
        dropout: float = 0.3,
        use_adversarial: bool = False,
        n_topics: int = 20,
        use_contrastive: bool = False,
        use_film: bool = False,
        use_transformer_seq: bool = False,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ):
        super().__init__()
        self.use_style = use_style
        self.use_attention_fusion = use_attention_fusion
        self.use_bilstm = use_bilstm
        self.unfreeze_layers = unfreeze_layers
        self.use_adversarial = use_adversarial
        self.n_topics = n_topics
        self.use_film = use_film

        from transformers import DebertaV2Model
        self.deberta = DebertaV2Model.from_pretrained(self.DEBERTA_MODEL)

        for param in self.deberta.parameters():
            param.requires_grad = False

        n_layers = len(self.deberta.encoder.layer)
        for layer in self.deberta.encoder.layer[n_layers - unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        if hasattr(self.deberta, 'pooler') and self.deberta.pooler is not None:
            for param in self.deberta.pooler.parameters():
                param.requires_grad = True

        # Fusion
        if use_style and use_attention_fusion:
            if use_film:
                self.style_fusion = FiLMFusion(self.DEBERTA_DIM, style_dim)
            else:
                self.style_fusion = StyleAttentionFusion(self.DEBERTA_DIM, style_dim)
            fused_dim = self.DEBERTA_DIM + style_dim
        elif use_style:
            fused_dim = self.DEBERTA_DIM + style_dim
        else:
            fused_dim = self.DEBERTA_DIM

        self.fused_dim = fused_dim

        # Contrastive projection head -- only used during training when
        # use_contrastive=True. Operates on the fused per-boundary
        # representation, BEFORE the BiLSTM (see module docstring for why:
        # this shapes the fusion output itself to be label-discriminative
        # and topic-invariant, matching where SCL-DeBERTa applies its
        # contrastive+topic-decoupling objective "atop DeBERTa").
        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )

        # BiLSTM path
        if use_bilstm:
            self.proj = nn.Sequential(
                nn.Linear(fused_dim, bilstm_hidden * 2),
                nn.LayerNorm(bilstm_hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.bilstm = nn.LSTM(
                bilstm_hidden * 2, bilstm_hidden,
                num_layers=1, batch_first=True,
                bidirectional=True, dropout=0,
            )
            head_input = bilstm_hidden * 2  # 256
            self.use_transformer_seq = use_transformer_seq
            if use_transformer_seq:
                self.seq_transformer = TransformerSeqEncoder(
                    head_input, n_heads=transformer_heads, n_layers=transformer_layers,
                )
        else:
            head_input = fused_dim
            self.use_transformer_seq = False

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(head_input, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Topic adversarial branch — operates on the same representation
        # that feeds the classification head (post-BiLSTM if BiLSTM is on,
        # otherwise the fused pair representation).
        if use_adversarial:
            self.topic_adversary = TopicAdversary(head_input, n_topics)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"DeBERTa Fusion E2E: {trainable:,} trainable / {total:,} total")
        print(f"Fine-tuning last {unfreeze_layers} DeBERTa layers")
        print(f"Stylometric features: {'ON' if use_style else 'OFF'}")
        print(f"Attention fusion: {'ON (FiLM)' if (use_attention_fusion and use_film) else ('ON (gate)' if use_attention_fusion else 'OFF (concat)')}")
        print(f"BiLSTM: {'ON' if use_bilstm else 'OFF'}")
        print(f"Transformer-over-BiLSTM: {'ON' if self.use_transformer_seq else 'OFF'}")
        print(f"Topic adversarial: {'ON (' + str(n_topics) + ' pseudo-topics)' if use_adversarial else 'OFF'}")
        print(f"Contrastive (topic-decoupling): {'ON' if use_contrastive else 'OFF'}")

    def encode_pairs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_diff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode sentence pairs through DeBERTa + fusion. Returns fused representations."""
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]

        if self.use_style and style_diff is not None:
            if self.use_attention_fusion:
                fused = self.style_fusion(cls_hidden, style_diff)
            else:
                fused = torch.cat([cls_hidden, style_diff], dim=-1)
        else:
            fused = cls_hidden

        return fused

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_diff: Optional[torch.Tensor] = None,
        topic_lambda: Optional[float] = None,
    ):
        """Pair-level forward (no BiLSTM). Returns logits (B,), or
        (logits, topic_logits) if adversarial mode is on and topic_lambda
        is provided."""
        fused = self.encode_pairs(input_ids, attention_mask, style_diff)
        logits = self.head(fused).squeeze(-1)

        if self.use_adversarial and topic_lambda is not None:
            topic_logits = self.topic_adversary(fused, lambda_=topic_lambda)
            return logits, topic_logits

        return logits

    def forward_document(
        self,
        pair_reprs: torch.Tensor,
        topic_lambda: Optional[float] = None,
    ):
        """
        Document-level forward with BiLSTM.
        pair_reprs: (1, n_boundaries, fused_dim) — pre-encoded pair representations
        Returns: logits (n_boundaries,), or (logits, topic_logits) if
        adversarial mode is on and topic_lambda is provided.
        """
        x = self.proj(pair_reprs)          # (1, n_boundaries, hidden*2)
        lstm_out, _ = self.bilstm(x)       # (1, n_boundaries, hidden*2)
        if self.use_transformer_seq:
            lstm_out = self.seq_transformer(lstm_out).to(lstm_out.dtype)
        logits = self.head(lstm_out)       # (1, n_boundaries, 1)
        logits = logits.squeeze(-1).squeeze(0)  # (n_boundaries,)

        if self.use_adversarial and topic_lambda is not None:
            topic_logits = self.topic_adversary(lstm_out.squeeze(0), lambda_=topic_lambda)
            return logits, topic_logits

        return logits

    def get_deberta_params(self):
        params = []
        for name, param in self.deberta.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def get_head_params(self):
        head_params = list(self.head.parameters())
        if self.use_style and self.use_attention_fusion:
            head_params += list(self.style_fusion.parameters())
        if self.use_bilstm:
            head_params += list(self.proj.parameters())
            head_params += list(self.bilstm.parameters())
        if self.use_transformer_seq:
            head_params += list(self.seq_transformer.parameters())
        if self.use_contrastive:
            head_params += list(self.contrastive_proj.parameters())
        return head_params

    def get_adversary_params(self):
        if self.use_adversarial:
            return list(self.topic_adversary.parameters())
        return []


# --------------------------------------------------------------------------- #
#  Dataset for E2E fine-tuning (pair-level)
# --------------------------------------------------------------------------- #

class E2EDataset(Dataset):
    def __init__(
        self,
        sentences_list: List[List[str]],
        labels_list: List[List[int]],
        style_diffs_list: Optional[List[np.ndarray]] = None,
    ):
        self.pairs = []
        self.labels = []
        self.styles = []

        for doc_idx, (sents, labels) in enumerate(zip(sentences_list, labels_list)):
            n_boundaries = len(sents) - 1
            for i in range(min(n_boundaries, len(labels))):
                self.pairs.append((sents[i], sents[i + 1]))
                self.labels.append(labels[i])
                if style_diffs_list is not None:
                    self.styles.append(style_diffs_list[doc_idx][i])
                else:
                    self.styles.append(None)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return {
            "sent_a": self.pairs[idx][0],
            "sent_b": self.pairs[idx][1],
            "label": self.labels[idx],
            "style_diff": self.styles[idx],
        }


def e2e_collate_fn(batch, tokenizer, max_length=256):
    sent_a = [item["sent_a"] for item in batch]
    sent_b = [item["sent_b"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    encoded = tokenizer(
        sent_a, sent_b,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    result = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }

    if batch[0]["style_diff"] is not None:
        style_diffs = torch.tensor(
            np.stack([item["style_diff"] for item in batch]),
            dtype=torch.float32,
        )
        result["style_diff"] = style_diffs

    return result


# --------------------------------------------------------------------------- #
#  Dataset for document-level training (BiLSTM)
# --------------------------------------------------------------------------- #

class DocLevelDataset(Dataset):
    """Each item is one document: list of sentence pairs + labels + style diffs.

    doc_id is carried through so the adversarial branch can look up the
    precomputed pseudo-topic labels for each boundary of this document.
    """

    def __init__(
        self,
        sentences_list: List[List[str]],
        labels_list: List[List[int]],
        style_diffs_list: Optional[List[np.ndarray]] = None,
        doc_ids: Optional[List[str]] = None,
    ):
        self.docs = []
        for doc_idx, (sents, labels) in enumerate(zip(sentences_list, labels_list)):
            n = min(len(sents) - 1, len(labels))
            if n <= 0:
                continue
            pairs = [(sents[i], sents[i + 1]) for i in range(n)]
            doc_labels = labels[:n]
            if style_diffs_list is not None:
                doc_styles = style_diffs_list[doc_idx][:n]
            else:
                doc_styles = None
            doc_id = doc_ids[doc_idx] if doc_ids is not None else str(doc_idx)
            self.docs.append({
                "doc_id": doc_id,
                "pairs": pairs,
                "labels": doc_labels,
                "style_diffs": doc_styles,
            })

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]


# --------------------------------------------------------------------------- #
#  Frozen DeBERTa encoder (for precompute mode)
# --------------------------------------------------------------------------- #

class DeBERTaEncoder:
    DEBERTA_MODEL = "microsoft/deberta-v3-small"

    def __init__(self, device: str = "cpu", batch_size: int = 16):
        from transformers import DebertaV2Model, DebertaV2Tokenizer
        self.device = device
        self.batch_size = batch_size

        print(f"[DeBERTaEncoder] Loading {self.DEBERTA_MODEL} on {device}...")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.DEBERTA_MODEL)
        self.model = DebertaV2Model.from_pretrained(self.DEBERTA_MODEL).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print("[DeBERTaEncoder] Ready.")

    @torch.no_grad()
    def encode_pairs(self, sentences: List[str]) -> torch.Tensor:
        if len(sentences) < 2:
            return torch.zeros(0, 768)

        all_cls = []
        pairs_a = sentences[:-1]
        pairs_b = sentences[1:]

        for start in range(0, len(pairs_a), self.batch_size):
            end = min(start + self.batch_size, len(pairs_a))
            batch_a = pairs_a[start:end]
            batch_b = pairs_b[start:end]

            encoded = self.tokenizer(
                batch_a, batch_b,
                padding=True, truncation=True, max_length=256,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**encoded)
            cls_vec = out.last_hidden_state[:, 0, :]
            all_cls.append(cls_vec.cpu())

        return torch.cat(all_cls, dim=0)


# --------------------------------------------------------------------------- #
#  Frozen mode model + dataset
# --------------------------------------------------------------------------- #

class DeBERTaFusionModel(nn.Module):
    def __init__(self, deberta_dim=768, style_dim=193, use_style=True, dropout=0.3):
        super().__init__()
        self.use_style = use_style
        input_dim = deberta_dim + (style_dim if use_style else 0)

        self.bilstm = nn.LSTM(
            input_dim, 128, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.bilstm(x)
        logits = self.head(lstm_out).squeeze(-1)
        return logits


class DeBERTaFusionDataset(Dataset):
    def __init__(self, deberta_embs, style_diffs, labels):
        self.items = []
        for emb, sd, lbl in zip(deberta_embs, style_diffs, labels):
            if emb.shape[0] == 0:
                continue
            self.items.append((emb, sd, torch.tensor(lbl, dtype=torch.float32)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def deberta_fusion_collate_fn(batch):
    embs, styles, labels = zip(*batch)

    max_len = max(e.shape[0] for e in embs)
    B = len(batch)
    feat_dim = embs[0].shape[1] + styles[0].shape[1]

    x_pad = torch.zeros(B, max_len, feat_dim)
    y_pad = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (e, s, l) in enumerate(batch):
        n = e.shape[0]
        x_pad[i, :n] = torch.cat([e, s], dim=-1)
        y_pad[i, :n] = l
        mask[i, :n] = True

    return x_pad, y_pad, mask