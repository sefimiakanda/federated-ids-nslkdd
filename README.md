# Federated Intrusion Detection on NSL-KDD

Five banks, one shared enemy: detect network intrusions **without ever pooling customer traffic data**.

This project trains a network-intrusion detector with **Federated Learning (FedAvg)** on the
[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) dataset. Each of five simulated banks trains a model
locally on its own data; only model weights are shared and averaged into a global model. Raw records never
leave a client.

It was built for the *Secure AI Competition – Day 1 (Beginner Track)* on Kaggle, starting from the organisers'
weak baseline (single 8-unit hidden layer, F1 ≈ 0.75–0.78) and improving it.

> **Result:** the submission scored **0.991** on the Kaggle leaderboard (self-reported by the author).
> See [Interpreting the score](#interpreting-the-score) for important context.

---

## Why federated learning for security?

Banks, hospitals and telecoms often hold the data needed to detect attacks but are legally or contractually
forbidden from sharing it. Federated learning addresses this:

```
        ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
        │ Bank 1 │   │ Bank 2 │   │ Bank 3 │   │ Bank 4 │   │ Bank 5 │
        │ (data) │   │ (data) │   │ (data) │   │ (data) │   │ (data) │
        └───┬────┘   └───┬────┘   └───┬────┘   └───┬────┘   └───┬────┘
            │ local training (data stays here)                 │
            └──────────────┬──────────────┬────────────────────┘
                     model weights only  ▼
                        ┌─────────────────────┐
                        │  FedAvg aggregation │  weighted by client size
                        └──────────┬──────────┘
                                   ▼
                          new global model ──► sent back to all banks
                                   (repeat for N rounds)
```

## What was improved over the baseline

| Area | Baseline | This project |
|---|---|---|
| Features | 41 raw features | Dropped near-constant `num_outbound_cmds`; added 6 derived signals (below) |
| Model | 1 hidden layer, 8 units | 2 hidden layers (64 → 32), LayerNorm, Dropout |
| Optimiser | SGD, 1 local epoch | Adam + weight decay, 3 local epochs, 10 rounds |
| Class imbalance | none | Per-client `pos_weight` in the loss, computed from *local* labels only |
| Decision rule | fixed 0.5 threshold | Threshold swept over 0.30–0.70 to maximise F1 |

**Engineered features** (all simple functions of existing standardised columns, so they apply identically at
inference time):

- `error_asymmetry` = `serror_rate − rerror_rate`
- `srv_error_asymmetry` = `srv_serror_rate − srv_rerror_rate`
- `bytes_diff` = `src_bytes − dst_bytes`
- `host_diff_ratio` = `dst_host_diff_srv_rate − dst_host_same_srv_rate`
- `conn_density` = `count × srv_count`
- `login_risk` = `num_failed_logins + is_guest_login`

**Design note – why LayerNorm and not BatchNorm?** FedAvg averages only *learnable parameters*. BatchNorm also
keeps running mean/variance *buffers* that are not parameters and would silently not be averaged, leaving the
global model with inconsistent statistics. LayerNorm normalises per sample and has no such buffers.

**Design note – privacy of the class weighting.** Each client computes `pos_weight = n_normal / n_attack` from
its own labels. No global label statistics are exchanged.

## Repository structure

```
federated-ids-nslkdd/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── 01_Beginner_Track_Day1_improved.ipynb   # the competition notebook, improved
├── src/
│   └── fl_ids.py                               # same pipeline as a standalone script
└── data/
    └── .gitkeep                                # put test_public.csv here (not committed)
```

## Getting started

### Option A – Kaggle / Colab (easiest)

1. Upload `notebooks/01_Beginner_Track_Day1_improved.ipynb`.
2. Add the competition's `test_public.csv` (on Kaggle: attach the competition data, then set
   `KAGGLE_TEST_PATH = "/kaggle/input/<competition-slug>/test_public.csv"`).
3. Run all cells top to bottom **once**. The notebook downloads NSL-KDD itself (internet must be on).
4. The last cell writes `submission.csv`. Upload that file to the leaderboard.

### Option B – local script

```bash
git clone https://github.com/sefimiakanda/federated-ids-nslkdd.git
cd federated-ids-nslkdd
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# place the competition file at data/test_public.csv, then:
python src/fl_ids.py --test-path data/test_public.csv --out submission.csv
```

Options: `--rounds` (default 10), `--epochs` local epochs per round (default 3).

## How it works, step by step

1. **Load & clean** – download `KDDTrain+` / `KDDTest+`; label `normal → 0`, everything else `→ 1`.
2. **Encode & scale** – label-encode `protocol_type`, `service`, `flag`; standardise all features
   (zero mean, unit variance).
3. **Reserve a secret holdout** – with `SEED = 42`, 15,000 rows are set aside *before* partitioning. No client
   ever trains on them; the organisers use the same rows for the leaderboard. A printed MD5 fingerprint lets you
   check you reproduced the same reservation.
4. **Partition** – the remaining rows are split into 5 IID clients ("banks").
5. **Federated training** – every round, each client copies the global model, trains locally, and the server
   computes a size-weighted average of the parameters (FedAvg).
6. **Threshold tuning** – sweep the decision threshold and keep the one with the best F1 on `KDDTest+`.
7. **Predict** – apply the same feature engineering to `test_public.csv` and write `Id,Expected`.

## Interpreting the score

A score of 0.991 is much higher than the ~0.75–0.78 baseline, so it deserves a sanity check rather than
blind celebration. Things worth knowing:

- **Different test distributions.** The Kaggle test set is a random slice of `KDDTrain+` (same distribution as
  the training pool). The notebook's local evaluation uses `KDDTest+`, which is known to be *harder* because it
  contains attack types that never appear in training. Expect a noticeably lower local F1 than leaderboard F1;
  that gap is normal here and is not by itself evidence of leakage.
- **Leakage check.** The holdout fingerprint must match the organisers'. If you change `SEED` or
  `HOLDOUT_SIZE`, your clients may overlap the hidden test rows and inflate the score.
- **Threshold tuning.** The threshold is tuned on `KDDTest+`, not on the Kaggle holdout, so it does not leak
  the leaderboard labels — but it is still a choice made on one evaluation set.
- **Scope.** NSL-KDD is a 2009-era benchmark. High scores here do not imply the model would catch modern
  attacks on real traffic.

## Limitations & ideas for future work

- Partitions are **IID**; real banks have heterogeneous traffic. Try non-IID splits (e.g. Dirichlet by attack type).
- FedAvg has no protection against **malicious clients** (model poisoning). Robust aggregation such as
  Krum, trimmed mean or median is the natural next step.
- Model updates can still leak information about training data. Consider **differential privacy** (clipping +
  Gaussian noise) and **secure aggregation**.
- Try gradient-boosted trees or per-attack-family metrics; report per-class recall (DoS, Probe, R2L, U2R).

## Data & attribution

- **NSL-KDD**: Tavallaee, Bagheri, Lu & Ghorbani, *A Detailed Analysis of the KDD CUP 99 Data Set*, 2009.
  Mirror used for download: [jmnwong/NSL-KDD-Dataset](https://github.com/jmnwong/NSL-KDD-Dataset).
- **Competition & starter notebook**: Nani Desmond, *Secure AI Competition – Day 1*, Kaggle, 2026.
- Competition data files (`test_public.csv`, `sample_submission.csv`) are **not** included in this repository.

## License

Released under the MIT License – see [LICENSE](LICENSE).
