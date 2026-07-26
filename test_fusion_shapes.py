"""
Standalone shape + gradient test for MiraBiomedBERT's fusion plumbing.

It replaces the two heavy backbones (MIRA, BiomedBERT) with tiny fakes so we
can verify -- with NO downloads and NO GPU -- that:
  * [B, T, F] ECG in  ->  [B, n_classes] logits out
  * per-lead time encoding + text cross-attention shapes line up
  * gradients reach the trainable head/fusion (and NOT the frozen backbones)
  * state_dict() drops the frozen backbone weights

Run:  python test_fusion_shapes.py
"""
import types
import importlib.util
import pathlib

import torch
import torch.nn as nn

# import mira_biomedbert.py directly, no package layout required
_spec = importlib.util.spec_from_file_location(
    "mira_biomedbert", pathlib.Path(__file__).parent / "mira_biomedbert.py"
)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


# --- tiny fakes -----------------------------------------------------------
class FakeMIRAOut:
    def __init__(self, h):
        self.last_hidden_state = h


class FakeMIRA(nn.Module):
    """Stand-in for MIRAModel: [B,T,1] + time -> [B,T,d]."""
    def __init__(self, d=48):
        super().__init__()
        self.proj = nn.Linear(1, d)
        self.config = types.SimpleNamespace(hidden_size=d, output_hidden_states=False)

    def forward(self, input_ids=None, time_values=None, return_dict=True):
        return FakeMIRAOut(self.proj(input_ids))


class FakeBertOut:
    def __init__(self, h):
        self.last_hidden_state = h


class FakeBert(nn.Module):
    """Stand-in for BiomedBERT: token ids -> [B,L,768]."""
    def __init__(self, d=768, vocab=30522):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.config = types.SimpleNamespace(hidden_size=d)

    def forward(self, input_ids=None, attention_mask=None):
        return FakeBertOut(self.emb(input_ids))


class FakeTokenizer:
    def __call__(self, texts, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        L = 12
        ids = torch.randint(1, 100, (len(texts), L))
        mask = torch.ones(len(texts), L, dtype=torch.long)
        return types.SimpleNamespace(input_ids=ids, attention_mask=mask)


# --- config / dataset stubs ----------------------------------------------
def ns(**kw):
    o = types.SimpleNamespace(**kw)
    o.get = lambda k, d=None: getattr(o, k, d)   # dict-ish .get on a namespace
    return o


def make_config():
    ts = ns(freeze=True, checkpoint="random", allow_random_init=True,
            repo_path="", import_module="x", config_module="x",
            random_config={}, sampling_rate=100.0)
    text = ns(model="fake", freeze=True, lora=False, max_len=64)
    mbb = ns(ts=ts, text=text, d_fuse=64, n_heads=4, d_head=16,
             sampling_rate=100.0)
    models = ns(mira_biomedbert=mbb)
    training = ns(dropout=0.1)
    cfg = ns(models=models, task="classification", history_len=512,
             training=training, paths={})
    return cfg


def make_dataset():
    d = ns()
    d.n_features = 12
    d.n_classes = 5
    d.description = "PTB-XL 12-lead ECG."
    d.task_description = "Classify the ECG into one of five superclasses."
    return d


# --- patch the backbone builders to use the fakes -------------------------
def build_ts(self):
    self.mira = FakeMIRA(d=48)
    for p in self.mira.parameters():
        p.requires_grad = False
    return 48


def build_text(self):
    self.bert = FakeBert(d=768)
    self.tokenizer = FakeTokenizer()
    self.text_max_len = 64
    for p in self.bert.parameters():
        p.requires_grad = False
    return 768


def main():
    M.MiraBiomedBERT._build_ts_backbone = build_ts
    M.MiraBiomedBERT._build_text_backbone = build_text

    model = M.MiraBiomedBERT(make_config(), make_dataset())

    B, T, Fdim = 4, 512, 12
    inputs = {
        "x_enc": torch.randn(B, T, Fdim),
        "labels": torch.randint(0, 5, (B,)),
        "descriptions": [f'Patient {{"age": {30+i}, "sex": "male"}}' for i in range(B)],
    }

    # train mode: raw logits
    model.train()
    logits = model(inputs)
    assert logits.shape == (B, 5), logits.shape
    print("train logits:", tuple(logits.shape))

    loss = nn.CrossEntropyLoss()(logits, inputs["labels"])
    loss.backward()

    # gradients reach head + fusion, NOT frozen backbones
    assert model.head[0].weight.grad is not None
    assert model.fusion.q_proj.weight.grad is not None
    assert all(p.grad is None for p in model.mira.parameters())
    assert all(p.grad is None for p in model.bert.parameters())
    print("grad check: head/fusion updated, backbones frozen  OK")

    # eval mode: probabilities that sum to 1
    model.eval()
    with torch.no_grad():
        probs = model(inputs)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
    print("eval probs sum to 1  OK")

    # state_dict drops frozen backbones
    sd = model.state_dict()
    assert not any(k.startswith("mira.") for k in sd)
    assert not any(k.startswith("bert.") for k in sd)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"state_dict keys: {len(sd)} (no mira./bert.)  OK")
    print(f"trainable params: {n_trainable:,}")
    print("\nALL SHAPE/GRAD CHECKS PASSED")


if __name__ == "__main__":
    main()
