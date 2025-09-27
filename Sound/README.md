## Prereqs
1. `ffmpeg` installed on the system

## `process_data/preprocess.py`
```bash
python process_data/preprocess.py --config configs/initial_config.yaml --in data/MP3/a00001.mp3 --out outputs/sample_prep.wav --save-aux
```

## Source Code

### `src/dataio/` – Data IO and preprocessing

* **`preprocess.py`**
  Implements digital signal processing (DSP) steps: DC removal, resampling, band-limiting, EQ, HPSS, compression, de-essing, envelope extraction.
  *Why*: Removes nuisance variation and enhances swallowing cues before ML sees the data.

* **`augment.py`**
  Implements on-the-fly waveform augmentations: additive noise, IR convolution, dynamic range changes, EQ tilt, HPSS mixing, pitch/time shifts, dry-food simulation.
  *Why*: Improves generalization by simulating realistic acoustic variability and edge cases.

* **`features.py`**
  Extracts frame-level features (STFT, log-mel, deltas, auxiliary scalars). Supports SpecAugment.
  *Why*: Converts waveform into a time–frequency representation that CRNNs can efficiently learn from.

* **`dataset.py`**
  PyTorch Dataset that ties together loading, preprocessing, augmentation, feature extraction, and target slicing into windows.
  *Why*: Provides ready-to-train minibatches with aligned features and labels.

---

### `src/models/` – Models and loss functions

* **`sed_crnn.py`**
  Implements the baseline CRNN (Convolutional + RNN/Transformer encoder). Includes optional multitask heads (duration regression, global count).
  *Why*: Learns temporal structure of swallow sounds in spectrograms; main backbone of the system.

* **`losses.py`**
  Loss functions: BCE, focal loss, Dice loss, count consistency.
  *Why*: Handles class imbalance, stabilizes training on rare events, and enforces consistency between framewise probabilities and counts.

---

### `src/train/` – Training pipeline

* **`main.py`**
  Entry point script for training. Loads config, datasets, model, optimizer, and runs training/validation loops.
  *Why*: Orchestrates the entire training process.

* **`train_step.py`**
  Implements one optimization step: forward pass, loss computation, backward pass, gradient clipping, optimizer/scheduler update.
  *Why*: Encapsulates training logic cleanly, ensuring reproducibility and modularity.

---

### `src/infer/` – Inference and evaluation

* **`inference.py`**
  Runs trained model on full recordings:

  * Window into chunks → forward pass
  * Stitch overlapping outputs → smooth posteriors
  * Apply hysteresis decoding, minima constraints, optional HMM
    Produces onset/offset times and swallow counts.
    *Why*: Turns raw posterior probabilities into usable swallow events.

* **`eval_metrics.py`**
  Evaluation metrics:

  * Event-based F1 with tolerance (via `sed_eval`)
  * Count MAE/MAPE
  * Onset MAE
  * Subject-wise macro F1
    *Why*: Quantifies performance in both detection accuracy and counting reliability.

---

### Other modules

* **`targets.py`**
  Utilities to convert annotated onsets/offsets into frame-level labels, slice them to windows, and count events.
  *Why*: Bridges human labels with frame-based model outputs.

* **`utils/verify_cuda.py`**
  Quick script to verify CUDA/torch installation and GPU availability.
  *Why*: Prevents silent failures from running training on CPU by mistake.

---

### Importance Summary

* **DSP preprocessing** (`preprocess.py`) reduces irrelevant noise and emphasizes swallow signatures.
* **Augmentations** (`augment.py`) improve robustness to recording conditions and subject variability.
* **Features** (`features.py`) supply the CRNN with spectrogram-like inputs.
* **Dataset** (`dataset.py`) unifies data → augmentation → features → labels.
* **Model** (`sed_crnn.py`) is the core detector, capturing temporal structure.
* **Losses** (`losses.py`) handle class imbalance and thin events.
* **Training pipeline** (`train_step.py`, `main.py`) enables efficient, reproducible optimization.
* **Inference + metrics** (`inference.py`, `eval_metrics.py`) transform predictions into counts and evaluate them meaningfully.

---

### Read order
To understand the code and repository, read in this order. Stop after each file and answer the check-questions.

1. **`configs/initial_config.yaml`**
   What toggles exist? What SR, windowing, thresholds, and augment probs are set?

2. **`src/dataio/preprocess.py`**
   How does raw audio become the model waveform? Note: dc\_block → resample → HP/LP → optional HPSS/gate/EQ/comp/de-esser → envelope.
   Check: which steps are enabled by your config?

3. **`src/dataio/features.py`**
   How does waveform → (C,F,T)? Note STFT params, mel, standardization, deltas, aux, windowing.
   Check: shapes for one file, and how `window_into_chunks` computes indices.

4. **`src/targets.py`**
   How do onsets/offsets become frame targets and counts?
   Check: weak-label duration, dilation, min event frames mapping.

5. **`src/dataio/dataset.py`**
   Where everything is wired: load → preprocess → (train? augment) → features → window → slice targets.
   Check: one `__getitem__` pass; confirm outputs `(C,F,Tw)`, `(Tw,)`, and count.

6. **`src/models/sed_crnn.py`**
   What is the backbone and head? Note DSConv blocks, freq pooling, LSTM/Transformer switch, and outputs dict.
   Check: expected input channel count matches features.

7. **`src/models/losses.py`**
   Which loss is active (BCE vs focal + Dice)? How count consistency is added.
   Check: class-imbalance settings from config.

8. **`src/train/train_step.py`** and **`src/train/main.py`**
   How one step is run (AMP, grad clip, AdamW, cosine warmup) and how loaders/model/optimizer are created.
   Check: batch shapes, scheduler warmup, early stop.

9. **`src/infer/inference.py`**
   Window → forward → stitch → smooth → hysteresis → constraints → events.
   Check: smoothing mode and thresholds; confirm count equals number of segments.

10. **`src/infer/eval_metrics.py`**
    How F1 (sed\_eval), count MAE/MAPE, onset MAE, and subject macro are computed.
    Check: onset tolerance and whether offsets are scored.

11. **`src/dataio/augment.py`** (last)
    What randomness is applied at train time only; how each aug maps to your spec.


## References
Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, “Metrics for polyphonic sound event detection”, Applied Sciences, 6(6):162, 2016
