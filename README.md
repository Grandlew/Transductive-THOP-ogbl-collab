# Transductive T-HOP Fusion for ogbl-collab

This repository contains the official implementation of **Transductive T-HOP Fusion** for the OGB `ogbl-collab` link prediction benchmark.

Transductive T-HOP Fusion is a non-neural temporal high-order path fusion method. It combines weighted structural closure, temporal two-hop and three-hop path signals, direct collaboration memory, and semantic similarity.

---

### Installation requirements

```bash
numpy>=1.22
pandas>=1.3
torch>=2.0
ogb>=1.3.6
tqdm>=4.64
scipy>=1.8
```
### Install dependencies:
```
pip install -r requirements.txt.
```

### Dataset
The code automatically downloads and processes ogbl-collab through OGB:

```
from ogb.linkproppred import PygLinkPropPredDataset
dataset = PygLinkPropPredDataset(name="ogbl-collab", root="data")
```
### Evaluation protocol
For validation:

```
input graph = training positive edges only
target = validation positive/negative edges
```
For final test inference:

```
input graph = training positive edges + validation positive edges
target = test positive/negative edges
```

No validation negative edges or test labels are used for graph construction or feature computation.

External data: No

### Basic command-line arguments

```
--config: path to the configuration file.
--root: directory where the OGB dataset is stored/downloaded.
--cache_dir: directory used to cache deterministic feature matrices.
--result_dir: directory used to save seed-level reports and predictions.
--trials: number of validation-search trials per seed.
--seed: random seed for one run.
--force_rebuild: rebuild cached feature matrices.
```
### Reproduce one seed

```
python run_thop_collab.py \
  --config configs/base.json \
  --root data \
  --cache_dir cache \
  --result_dir results \
  --seed 0 \
  --trials 10000
```
  ### This saves:
  results/tt_seed0_report.json
results/tt_seed0_predictions.pkl
results/tt_seed0_records.json

### Reproduce official 10-seed result
I use seeds 0–9.

```
python run_10_seeds.py \
  --config configs/base.json \
  --root data \
  --cache_dir cache \
  --result_dir results \
  --trials 10000
  ```

  This saves:

```
results/tt_seed0_report.json
...
results/tt_seed9_report.json
results/ten_seed_results.csv
results/ten_seed_summary.json
```

### Create test submission files

After running at least one seed:
```
python submit_collab.py \
  --pred_path results/tt_seed0_predictions.pkl \
  --submission_dir submission
```

This saves the prediction tensors and metadata into:

submission/

### Performance

Official 10-seed result over seeds 0–9:
| Dataset     |  Metric |      Validation |            Test |
| ----------- | ------: | --------------: | --------------: |
| ogbl-collab | Hits@50 | 0.6962 ± 0.0003 | 0.6925 ± 0.0010 |

The standard deviation is the unbiased standard deviation over 10 seeds.

Model

For a candidate pair (u, v), Transductive T-HOP Fusion computes:

```
weighted Adamic-Adar
weighted Resource Allocation
weighted common-neighbor closure
recency-weighted two-hop temporal path features
capped three-hop temporal path features
direct collaboration memory
semantic cosine similarity from OGB node features
```
The final score is a validation-selected linear fusion:

```
S(u, v) = Σ_i α_i z_i(u, v)
```
where z_i is a standardized feature score.

### Tuned hyperparameters

```
recency_lambda: [0.20*]
coherence_mu: [0.35*]
top_neighbors_3hop: [64*]
feature_chunk_size: [50000*]
blend_trials: [10000*]
random seeds: [0,1,2,3,4,5,6,7,8,9]
```

Fusion weight search ranges:
```
w_aa_product, w_aa_sqrt, aa, ra, w_ra_product: Uniform(0.0, 1.5)
w_cn_product: Uniform(0.0, 1.0)
temp2_sum, temp2_coh_sum, log1p_temp2, log1p_temp2_coh: Uniform(0.0, 1.5)
temp3_capped_sum, temp3_coh_capped_sum, log1p_temp3: Uniform(-0.2, 1.2)
direct_weight, direct_recency, direct_weight_x_recency, x_cosine: Uniform(0.0, 0.8)
```

Asterisks denote selected values.

### Parameters
This method has no neural trainable parameters.

```
#Params: 17 scalar fusion weights, 0 neural parameters
``` 
### Hardware

The method is CPU-compatible. No GPU is required.

 Hardware used:

```
Google Colab CPU
```

### Officiality

This is the official implementation by the author of the submitted method.

### Technical report

See:
```
TECHNICAL_REPORT.md

for the method description, protocol, feature definitions, hyperparameters, and 10-seed result. 
---


