# MIRA + BiomedBERT on PTB-XL — integration & honest assessment

This bundle adds a new model, `mira_biomedbert`, to your fork
(`kkianamm/medtsllm5`). It fuses **MIRA** (medical time-series foundation
model, used as a frozen backbone) with **BiomedBERT** (biomedical text encoder)
in a MedTsLLM-style classifier, and evaluates on **PTB-XL**.

## Files
- `mira_biomedbert.py` → copy into `models/`
- `ptbxl_mira_biomedbert.toml` → copy into `configs/datasets/`
- `test_fusion_shapes.py` → standalone shape/gradient test (no downloads/GPU)

## Wiring (3 steps)

1. **Register the model.** In `models/__init__.py` add:
   ```python
   from .mira_biomedbert import MiraBiomedBERT
   model_lookup["mira_biomedbert"] = MiraBiomedBERT
   ```

2. **Get the backbones.**
   - Clone MIRA and set `ts.repo_path` in the config to its root:
     `git clone https://github.com/microsoft/MIRA`
     Check the actual package folder casing and adjust `ts.import_module`
     (the repo ships `MIRA/MIRA/models/modeling_mira.py`).
   - `pip install transformers peft wfdb torchdiffeq`
     (`torchdiffeq` is a MIRA import dependency even though we don't run the ODE.)

3. **Get PTB-XL** from PhysioNet into `data/ptbxl/` (your `datasets/ptbxl.py`
   already reads `ptbxl_database.csv`, `scp_statements.csv`, `records100/`).

Run:
```bash
python3 train.py configs/datasets/ptbxl_mira_biomedbert.toml
```

Sanity-check the plumbing first (recommended, runs in seconds):
```bash
python3 test_fusion_shapes.py
```

## What the model actually does
- **TS branch:** each of the 12 ECG leads is encoded independently by MIRA's
  transformer backbone (`MIRAModel`, univariate, `input_size=1`) with a regular
  CT-RoPE timestamp grid; time-pooled to one vector per lead → `[B, 12, d_mira]`.
- **Text branch:** BiomedBERT encodes the task description + per-record patient
  JSON (`{"age":.., "sex":..}`, already produced by your `ptbxl.py`) →
  token embeddings + `[CLS]`.
- **Fusion:** a reprogramming-style cross-attention where the 12 lead vectors
  (queries) attend over BiomedBERT token embeddings (keys/values); pooled and
  concatenated with the `[CLS]` vector → 2-layer MLP head → 5 logits.
- Both backbones are **frozen** by default; only projections + fusion + head
  train (~0.1–2M params depending on `d_fuse`). `state_dict()` omits the frozen
  backbones so checkpoints stay small.

---

## Will this work? — honest assessment

**Short answer:** it will *run and train* a working PTB-XL classifier, and the
plumbing is verified (`test_fusion_shapes.py` passes). But several of your
premises don't line up with what these components actually are, and a few will
materially affect results. Read these before trusting any number it produces.

### 1. Your fork already contains something different from what you described
The `kkianamm/medtsllm5` fork does **not** use BiomedBERT or MIRA. It already
implements PTB-XL classification with:
- a general text LLM backbone (FLAN-T5-XL by default), and
- an optional **BiomedCoOp** head (a CLIP-style prompt-learning classifier,
  Koleilat et al., CVPR 2025).

**BiomedCoOp ≠ BiomedBERT ≠ MIRA.** So "combine BiomedBERT and MIRA" is a new
build, not a tweak of the existing files. This bundle is that new build,
implemented as a separate model so it doesn't disturb your existing setup.

### 2. MedTsLLM's core trick doesn't transfer to MIRA
MedTsLLM works by *reprogramming* time-series patches into a **text LLM's word-
embedding space**, then concatenating real text tokens and running everything
through the frozen LLM. That only works because the backbone is a language
model. **MIRA is a time-series model** (Time-MoE lineage: CT-RoPE + frequency
MoE + Neural ODE) with no word embeddings and no ability to ingest text. You
therefore *cannot* do literal MedTsLLM reprogramming with MIRA as the backbone.
This model keeps the cross-modal *alignment* idea (via cross-attention) but is
honestly a **dual-encoder fusion**, not the MedTsLLM mechanism.

### 3. MIRA is a *forecasting* model, not a classifier
MIRA was built and pretrained for zero-shot **forecasting**. Using it for
classification means using its backbone as a frozen feature extractor — a
reasonable, common move, but "off-label." Its features were never optimized to
be linearly separable by diagnosis, so don't expect foundation-model magic on
classification out of the box. Fine-tuning (unfreeze, or LoRA on the backbone)
will likely matter more here than it would for its native forecasting task.

### 4. The full-scale MIRA weights don't appear to be publicly released
This is the biggest practical risk. The official `microsoft/MIRA` repo ships
code that defaults to initializing from `Maple728/TimeMoE-50M` and loads a local
`/checkpoint` placeholder — no full 454B-token weights. There *is* a community
**demo reproduction** on Hugging Face (`MIRA-Mode/MIRA`), explicitly described
as *"not the full-scale MIRA model,"* trained only on public data. So your
realistic options are:
  - use the **demo** checkpoint (weak stand-in for the paper's model), or
  - init from **TimeMoE-50M** and accept it's not really "MIRA," or
  - **train MIRA yourself** (454B time points — not feasible for most).

Please re-verify current availability; if Microsoft has since released weights,
point `ts.checkpoint` at them.

### 5. PTB-XL is *in-distribution* for MIRA (a caveat, not a bug)
MIRA's pretraining corpus **includes PTB-XL**. So evaluating a MIRA-based model
on PTB-XL is not a clean out-of-distribution transfer test, and there's a mild
leakage flavor (waveforms were seen during pretraining, though not the
diagnostic labels). Fine for a course/engineering project; state it explicitly
if you make any "foundation model generalization" claim.

### 6. PTB-XL is really multi-label; the fork does single-label
`datasets/ptbxl.py` keeps only records with exactly one superclass and trains
single-label cross-entropy. The standard PTB-XL benchmark is **multi-label**
(records can carry several superclasses) and is scored with **macro-AUROC**, not
accuracy. Your accuracy numbers therefore won't be comparable to published
PTB-XL results. To match the literature: keep multi-label targets, switch the
head to `BCEWithLogitsLoss`, and report macro-AUROC.

### 7. Compute & practical notes
- MIRA-large (455M) + BiomedBERT (110M), both frozen, + a small trainable head
  fits on a single mid-range GPU. Full fine-tuning is much heavier — start
  frozen, then try LoRA on BiomedBERT (`text.lora = true`) and/or unfreezing
  MIRA's top layers.
- Encoding 12 leads independently multiplies the TS batch by 12; drop
  `batch_size` if you hit OOM.
- `torchdiffeq` is imported by MIRA even though we bypass the ODE; install it.

### Bottom line
- **Feasible and will train?** Yes — verified plumbing, standard training loop.
- **A faithful "MedTsLLM with BiomedBERT + MIRA"?** Not literally — it's a
  dual-encoder fusion, because MIRA can't host the reprogramming mechanism.
- **Will it beat the fork's existing FLAN-T5 / BiomedCoOp setup?** Unknown and
  not guaranteed, mainly because (a) real MIRA weights may be unavailable and
  (b) MIRA features aren't tuned for classification. Treat it as a research
  experiment, fix the multi-label/AUROC issue before comparing to published
  numbers, and be explicit about the checkpoint you actually used.
