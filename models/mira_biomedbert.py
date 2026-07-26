"""
MIRA + BiomedBERT multimodal classifier for the MedTsLLM framework.
=================================================================

This module drops a new model, ``mira_biomedbert``, into the med-ts-llm
(kkianamm/medtsllm5) codebase. It keeps the *spirit* of MedTsLLM -- align a
raw physiological time series with textual clinical context and read out a
task head -- but replaces the single frozen text-LLM backbone with TWO
domain-specific encoders:

  * TIME-SERIES branch  : MIRA  (microsoft/MIRA), a medical time-series
                          foundation model. We use only its transformer
                          *backbone* (``MIRAModel``) as a frozen feature
                          extractor -- NOT its Neural-ODE forecasting head.
  * TEXT branch         : BiomedBERT
                          (microsoft/BiomedNLP-BiomedBERT-base-uncased-...),
                          a biomedical BERT used to encode the patient/context
                          prompt into token + [CLS] embeddings.

Why this is a *fusion* model and not a literal MedTsLLM reprogramming:
--------------------------------------------------------------------
MedTsLLM's reprogramming layer maps time-series patches INTO a text LLM's
*word-embedding space*, then concatenates real text-token embeddings and runs
the whole sequence through the frozen LLM. That trick requires the backbone to
be a language model with a shared token embedding space.

MIRA is NOT a language model. It is a decoder-only *time-series* model
(Time-MoE lineage: CT-RoPE + frequency MoE + Neural ODE). It has no word
embeddings and cannot ingest text tokens. So we cannot literally feed
"reprogrammed patches + text tokens" through MIRA.

Instead we keep the cross-modal *alignment* idea via a reprogramming-style
cross-attention block: the per-lead MIRA features act as queries and attend
over BiomedBERT's text token embeddings (keys/values). The text-conditioned
time-series representation is then pooled and fused with the BiomedBERT [CLS]
vector and sent to a linear classification head.

Interface contract (matches tasks/classification.py + tasks/base.py):
  * __init__(self, config, dataset)
  * supported_tasks / supported_modes class attrs
  * forward(inputs) -> logits [B, n_classes]  (raw during training,
    softmax-normalised during eval, mirroring models/medtsllm.py)
  * self.aux_loss attribute (None here; the task adds it if present)
  * state_dict() strips the frozen backbones so checkpoints stay small
  * load_pretrained(saved_state)

Register it by adding to models/__init__.py:
    from .mira_biomedbert import MiraBiomedBERT
    model_lookup["mira_biomedbert"] = MiraBiomedBERT
"""

import os
import sys
import math
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Reprogramming-style cross-attention (time-series queries attend to text).
# This is the same multi-head cross-attention shape used by MedTsLLM's
# ReprogrammingLayer, generalised so queries and keys can have different dims.
# ---------------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_query, d_context, d_head, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.q_proj = nn.Linear(d_query, d_head * n_heads)
        self.k_proj = nn.Linear(d_context, d_head * n_heads)
        self.v_proj = nn.Linear(d_context, d_head * n_heads)
        self.out_proj = nn.Linear(d_head * n_heads, d_query)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, context_mask=None):
        # query:   [B, Lq, d_query]
        # context: [B, Lk, d_context]
        # mask:    [B, Lk] (1 = keep, 0 = pad) or None
        B, Lq, _ = query.shape
        Lk = context.shape[1]
        H, E = self.n_heads, self.d_head

        q = self.q_proj(query).view(B, Lq, H, E)
        k = self.k_proj(context).view(B, Lk, H, E)
        v = self.v_proj(context).view(B, Lk, H, E)

        scale = 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", q, k) * scale  # [B, H, Lq, Lk]

        if context_mask is not None:
            m = context_mask[:, None, None, :].to(torch.bool)   # [B,1,1,Lk]
            scores = scores.masked_fill(~m, float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshe->blhe", attn, v).reshape(B, Lq, H * E)
        return self.out_proj(out)                               # [B, Lq, d_query]


class MiraBiomedBERT(nn.Module):

    supported_tasks = ["classification"]
    supported_modes = ["univariate", "multivariate"]

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.model_config = config.models.mira_biomedbert
        self.device = None
        self.aux_loss = None                      # read by ClassificationTask

        self.task = config.task
        assert self.task == "classification", "This model only implements classification."

        self.seq_len = config.history_len
        self.n_features = dataset.n_features      # 12 for PTB-XL
        self.n_classes = dataset.n_classes        # 5 for PTB-XL superclasses
        self.dataset_description = getattr(dataset, "description", "")
        self.task_description = getattr(dataset, "task_description", "")

        self.dropout = config.training.dropout
        self.sampling_rate = float(self.model_config.get("sampling_rate", 100.0))

        # --- build the two frozen backbones -------------------------------
        self.d_mira = self._build_ts_backbone()   # sets self.mira
        self.d_text = self._build_text_backbone() # sets self.bert / self.tokenizer

        # --- projections + fusion + head ----------------------------------
        d_fuse = self.model_config.get("d_fuse", 256)
        n_heads = self.model_config.get("n_heads", 8)
        d_head = self.model_config.get("d_head", 32)

        # bring both modalities to a common width before fusion
        self.ts_in_proj = nn.Linear(self.d_mira, d_fuse)
        self.text_kv_proj = nn.Linear(self.d_text, d_fuse)

        self.fusion = CrossAttentionFusion(
            d_query=d_fuse, d_context=d_fuse,
            d_head=d_head, n_heads=n_heads, dropout=self.dropout,
        )
        self.fuse_norm = nn.LayerNorm(d_fuse)

        # classification head reads [pooled text-conditioned TS ; text CLS]
        self.text_cls_proj = nn.Linear(self.d_text, d_fuse)
        self.head = nn.Sequential(
            nn.Linear(2 * d_fuse, d_fuse),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(d_fuse, self.n_classes),
        )

        self._report_params()

    # ------------------------------------------------------------------ #
    # Backbone construction (isolated so they can be mocked in tests)
    # ------------------------------------------------------------------ #
    def _build_ts_backbone(self):
        """Load MIRA's transformer backbone (frozen). Returns hidden width."""
        ts_cfg = self.model_config.ts
        repo_path = ts_cfg.get("repo_path", "")
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        module_name = ts_cfg.get("import_module", "MIRA.models.modeling_mira")
        cfg_module = ts_cfg.get("config_module", "MIRA.models.configuration_mira")
        try:
            modeling = importlib.import_module(module_name)
        except Exception as e:                       # pragma: no cover
            raise ImportError(
                "Could not import MIRA. Clone https://github.com/microsoft/MIRA "
                "and set models.mira_biomedbert.ts.repo_path to the repo root "
                f"(so `import {module_name}` works). Original error: {e}"
            )
        MIRAModel = getattr(modeling, "MIRAModel")

        ckpt = ts_cfg.get("checkpoint", "")
        allow_random = ts_cfg.get("allow_random_init", False)

        if ckpt and ckpt.lower() not in ("", "none", "random"):
            # Load pretrained weights (HF id or local dir). We instantiate the
            # backbone via from_pretrained so CT-RoPE/MoE weights are populated.
            self.mira = MIRAModel.from_pretrained(
                ckpt, trust_remote_code=True,
                cache_dir=self.config.get("paths", {}).get("llm_path") or None,
            )
        elif allow_random:
            MIRAConfig = getattr(importlib.import_module(cfg_module), "MIRAConfig")
            mira_cfg = MIRAConfig(**ts_cfg.get("random_config", {}))
            self.mira = MIRAModel(mira_cfg)
        else:
            raise ValueError(
                "No MIRA checkpoint provided. Set models.mira_biomedbert.ts."
                "checkpoint to a HF id / local path, or set allow_random_init "
                "= true to train the architecture from scratch (NOT the "
                "pretrained foundation model)."
            )

        self.mira.config.output_hidden_states = False
        if ts_cfg.get("freeze", True):
            for p in self.mira.parameters():
                p.requires_grad = False
            self.mira.eval()
        return self.mira.config.hidden_size

    def _build_text_backbone(self):
        """Load BiomedBERT (frozen by default). Returns hidden width."""
        from transformers import AutoModel, AutoTokenizer

        txt_cfg = self.model_config.text
        name = txt_cfg.get(
            "model",
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        )
        cache_dir = self.config.get("paths", {}).get("llm_path") or None
        self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        self.bert = AutoModel.from_pretrained(name, cache_dir=cache_dir)
        self.text_max_len = txt_cfg.get("max_len", 64)

        if txt_cfg.get("lora", False):
            from peft import LoraConfig, get_peft_model, TaskType
            peft_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=txt_cfg.get("lora_rank", 8),
                lora_alpha=txt_cfg.get("lora_alpha", 16),
                lora_dropout=txt_cfg.get("lora_dropout", 0.05),
                target_modules=txt_cfg.get("lora_targets", ["query", "value"]),
            )
            self.bert = get_peft_model(self.bert, peft_cfg)
        elif txt_cfg.get("freeze", True):
            for p in self.bert.parameters():
                p.requires_grad = False
            self.bert.eval()

        return self.bert.config.hidden_size

    # ------------------------------------------------------------------ #
    # Encoders
    # ------------------------------------------------------------------ #
    def encode_ts(self, x_enc):
        """x_enc: [B, T, F] -> per-lead MIRA features [B, F, d_mira]."""
        if x_enc.ndim == 2:
            x_enc = x_enc.unsqueeze(-1)
        B, T, Fdim = x_enc.shape

        # MIRA is univariate (input_size=1): encode each lead independently.
        # [B, T, F] -> [B*F, T, 1]
        x = x_enc.permute(0, 2, 1).reshape(B * Fdim, T, 1).contiguous()

        # per-sequence instance normalisation (MIRA expects standardised input)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x = (x - mean) / std

        # regular timestamps for CT-RoPE: 0, 1/fs, 2/fs, ...
        t = torch.arange(T, device=x.device, dtype=torch.float32) / self.sampling_rate
        time_values = t[None, :].expand(B * Fdim, T).contiguous()

        ts_ctx = torch.no_grad() if self.model_config.ts.get("freeze", True) \
            else torch.enable_grad()
        with ts_ctx:
            out = self.mira(input_ids=x, time_values=time_values,
                            return_dict=True)
        h = out.last_hidden_state                # [B*F, T, d_mira]
        h = h.mean(dim=1)                        # pool time -> [B*F, d_mira]
        return h.view(B, Fdim, self.d_mira)      # [B, F, d_mira]

    def encode_text(self, descriptions, batch_size, device):
        """descriptions: list[str] (len B) or "" -> token embs + cls + mask."""
        if isinstance(descriptions, str):
            descriptions = [descriptions] * batch_size
        if descriptions is None or len(descriptions) == 0:
            descriptions = [""] * batch_size

        # Prepend a compact task/dataset framing so the text branch always has
        # clinical context even if per-sample descriptions are empty.
        framed = [
            f"{self.task_description} {d}".strip() for d in descriptions
        ]

        tok = self.tokenizer(
            framed, return_tensors="pt", padding=True, truncation=True,
            max_length=self.text_max_len,
        )
        input_ids = tok.input_ids.to(device)
        attn = tok.attention_mask.to(device)

        txt_ctx = torch.no_grad() if self.model_config.text.get("freeze", True) \
            and not self.model_config.text.get("lora", False) else torch.enable_grad()
        with txt_ctx:
            out = self.bert(input_ids=input_ids, attention_mask=attn)
        tokens = out.last_hidden_state           # [B, L, d_text]
        cls = tokens[:, 0]                       # [CLS] -> [B, d_text]
        return tokens, cls, attn

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def predict(self, inputs):
        x_enc = inputs["x_enc"]
        if self.device is None:
            self.device = x_enc.device
        B = x_enc.size(0)

        ts_feats = self.encode_ts(x_enc)                       # [B, F, d_mira]
        text_tokens, text_cls, text_mask = self.encode_text(
            inputs.get("descriptions"), B, x_enc.device
        )

        # project to common width
        ts_q = self.ts_in_proj(ts_feats)                       # [B, F, d_fuse]
        text_kv = self.text_kv_proj(text_tokens)               # [B, L, d_fuse]

        # reprogramming-style cross-attention: leads attend over text
        fused = self.fusion(ts_q, text_kv, context_mask=text_mask)
        fused = self.fuse_norm(ts_q + fused)                   # residual
        ts_pooled = fused.mean(dim=1)                          # [B, d_fuse]

        text_vec = self.text_cls_proj(text_cls)                # [B, d_fuse]
        joint = torch.cat([ts_pooled, text_vec], dim=-1)       # [B, 2*d_fuse]
        logits = self.head(joint)                              # [B, n_classes]
        return logits

    def forward(self, inputs):
        logits = self.predict(inputs)
        if not self.training:
            logits = F.softmax(logits, dim=-1)
        return logits

    # ------------------------------------------------------------------ #
    # Checkpoint hygiene: don't save the frozen giant backbones
    # ------------------------------------------------------------------ #
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        drop_frozen_ts = self.model_config.ts.get("freeze", True)
        drop_frozen_text = (
            self.model_config.text.get("freeze", True)
            and not self.model_config.text.get("lora", False)
        )
        for k in list(sd.keys()):
            if drop_frozen_ts and k.startswith("mira."):
                del sd[k]
            elif drop_frozen_text and k.startswith("bert."):
                del sd[k]
        return sd

    def load_pretrained(self, saved_state):
        for k in ("head.3.weight", "head.3.bias"):
            saved_state.pop(k, None)               # allow re-init of final layer
        incompat = self.load_state_dict(saved_state, strict=False)
        assert len(incompat.unexpected_keys) == 0, \
            f"Unexpected keys: {incompat.unexpected_keys}"
        return list(saved_state.keys())

    # ------------------------------------------------------------------ #
    def _report_params(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[mira_biomedbert] total params: {total:,} | trainable: {train:,}")
        print(f"[mira_biomedbert] d_mira={self.d_mira} d_text={self.d_text} "
              f"n_features={self.n_features} n_classes={self.n_classes}")
