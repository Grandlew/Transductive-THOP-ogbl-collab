
# Technical Report: Transductive T-HOP Fusion for ogbl-collab

## 1. Method Name

**Transductive T-HOP Fusion**

T-HOP stands for **Temporal High-Order Path Fusion**.

---

## 2. Dataset

We evaluate the method on the OGB link prediction benchmark:

```text
ogbl-collab

3. External Data

External data: No

The method only uses:

official OGB training edges
official OGB validation edges
official OGB test candidate edges
official OGB edge weights
official OGB edge years
official OGB node features

No external pretrained model, raw text, external graph, or additional labeled/unlabeled data is used.

Summary of the Method

Transductive T-HOP Fusion is a non-neural link prediction method for temporal collaboration graphs.

The method combines:

weighted Adamic-Adar
weighted Resource Allocation
weighted common-neighbor closure
recency-weighted two-hop temporal paths
recency-weighted capped three-hop temporal paths
direct collaboration memory
semantic cosine similarity

For each candidate pair (u, v), the method computes a set of scalar graph features and combines them using a validation-selected linear fusion model.

The final score is:

S(u, v) = Σ_i α_i z_i(u, v)

where:

z_i(u, v)

is the standardized value of feature i, and:

α_i

is the fusion weight selected using validation Hits@50.

Evaluation Protocol

The method uses a transductive inference protocol for final test prediction.

Validation

For validation, the input graph contains only training positive edges:

G_valid_input = train positive edges

Validation candidate edges are scored using this train-only graph.

Test

For final test inference, validation positive edges are added to the observed graph:

G_test_input = train positive edges + validation positive edges

Test candidate edges are scored using this train+validation graph.

No validation negative edges are used in graph construction.
No test labels are used in graph construction, model selection, feature computation, or hyperparameter tuning.

Model selection is performed only using validation Hits@50.

Feature Definitions

For a candidate pair (u, v), let N(u) denote the neighbor set of node u in the corresponding input graph.

Common Neighbors

CN(u, v) = |N(u) ∩ N(v)|

This measures local collaboration closure.

Resource Allocation

RA(u, v) = Σ_{k ∈ N(u) ∩ N(v)} 1 / deg(k)

This downweights high-degree common neighbors

Adamic-Adar

AA(u, v) = Σ_{k ∈ N(u) ∩ N(v)} 1 / log(deg(k) + 2)

The +2 term is used for numerical stability.

Weighted Adamic-Adar

Let w(a, b) be the observed collaboration weight between nodes a and b.

The product-weighted Adamic-Adar score is:

w_aa_product(u, v)
=
Σ_{k ∈ N(u) ∩ N(v)}
w(u, k) w(k, v) / log(deg(k) + 2)

The square-root weighted Adamic-Adar score is:

w_aa_sqrt(u, v)
=
Σ_{k ∈ N(u) ∩ N(v)}
sqrt(w(u, k) w(k, v)) / log(deg(k) + 2)

Weighted Resource Allocation

w_ra_product(u, v)
=
Σ_{k ∈ N(u) ∩ N(v)}
w(u, k) w(k, v) / deg(k)

Weighted Common Neighbor Closure

w_cn_product(u, v)
=
Σ_{k ∈ N(u) ∩ N(v)}
w(u, k) w(k, v)

Recency-Weighted Two-Hop Temporal Paths

Each historical edge has a year t_e. We define edge recency as:

r_e = exp(-λ(t_max - t_e))

where:

λ = 0.20

For a two-hop path:

u → k → v

the temporal path strength is:

sqrt(w(u, k) w(k, v)) r(u, k) r(k, v)

The feature temp2_sum sums this value over all common neighbors.

Time-Coherent Two-Hop Temporal Paths

We also score whether the two edges in a two-hop path happened in a similar time period.

The coherence term is:

coh(u, k, v) = exp(-μ |t(u, k) - t(k, v)|)

where:

μ = 0.35

The coherent two-hop temporal path score is:

temp2_coh_sum(u, v)
=
Σ_k sqrt(w(u, k) w(k, v)) r(u, k) r(k, v)
exp(-μ |t(u, k) - t(k, v)|)

Capped Three-Hop Temporal Paths

We also compute capped high-order temporal paths of the form:

u → a → b → v

To keep computation feasible, we use the top 64 temporal-strength neighbors of each endpoint. Edge temporal strength is:

w(e) r(e)

For a three-hop path, the path score is:

s_3(u, a, b, v)
=
[w(u, a) r(u, a)]
[w(a, b) r(a, b)]
[w(b, v) r(b, v)]

The feature temp3_capped_sum sums this score over capped three-hop paths.

A time-coherent variant uses the year variance of the three edges:

temp3_coh(u, a, b, v)
=
s_3(u, a, b, v)
exp(-μ Var(t(u, a), t(a, b), t(b, v)))

Direct Collaboration Memory

The method includes direct historical collaboration features:

direct_weight
direct_recency
direct_weight_x_recency

These capture whether two authors have previously collaborated in the input graph, and how strong/recent that collaboration was.

Semantic Similarity

The method uses the official OGB node features. For a pair (u, v), semantic similarity is computed as cosine similarity:

x_cosine(u, v)
=
x_u^T x_v / (||x_u|| ||x_v||)

No external embeddings or pretrained models are used.

Final Fusion Features

The final feature set is:
w_aa_product
w_aa_sqrt
aa
ra
w_ra_product
w_cn_product
temp2_sum
temp2_coh_sum
temp3_capped_sum
temp3_coh_capped_sum
log1p_temp2
log1p_temp2_coh
log1p_temp3
direct_weight
direct_recency
direct_weight_x_recency
x_cosine

Each feature is standardized before fusion

Model Selection

For each random seed, fusion weights are selected using random search over validation Hits@50.

The selected weights are then used to score the test split.

The test split is never used for selecting fusion weights.

Tuned Hyperparameters

The following hyperparameters were used:

recency_lambda: [0.20*]
coherence_mu: [0.35*]
top_neighbors_3hop: [64*]
feature_chunk_size: [50000*]
blend_trials: [10000*]
random seeds: [0,1,2,3,4,5,6,7,8,9]

The asterisk indicates the selected value.

Fusion Weight Search Space

For each seed, the fusion weights are sampled from the following ranges:

w_aa_product, w_aa_sqrt, aa, ra, w_ra_product:
    Uniform(0.0, 1.5)

w_cn_product:
    Uniform(0.0, 1.0)

temp2_sum, temp2_coh_sum, log1p_temp2, log1p_temp2_coh:
    Uniform(0.0, 1.5)

temp3_capped_sum, temp3_coh_capped_sum, log1p_temp3:
    Uniform(-0.2, 1.2)

direct_weight, direct_recency, direct_weight_x_recency, x_cosine:
    Uniform(0.0, 0.8)

Each seed uses:

10,000 validation-search trials

Official 10-Seed Results

The official result is reported over seeds:

0, 1, 2, 3, 4, 5, 6, 7, 8, 9

The mean and unbiased standard deviation are:

Validation Hits@50 = 0.696165 ± 0.000264
Test Hits@50       = 0.692506 ± 0.000975

Rounded leaderboard format:

Validation Hits@50 = 0.6962 ± 0.0003
Test Hits@50       = 0.6925 ± 0.0010

Parameters

The method has no neural trainable parameters.

The final fusion model uses:

17 scalar fusion weights

Therefore:

#Params: 17 scalar fusion weights, 0 neural parameters

Hardware

The method is CPU-compatible. No GPU is required.

Example hardware:

Google Colab CPU / local CPU

If a specific machine is used for final reproduction, the exact CPU and RAM can be reported in the leaderboard submission.

OGB Version

The OGB version should be reported from:

import ogb
print(ogb.__version__)

The code was designed for:

ogb >= 1.3.6

Officiality

This is the official implementation by the author of Transductive T-HOP Fusion.

Reproducibility

To reproduce the 10-seed result:

python run_10_seeds.py \
  --config configs/base.json \
  --root data \
  --cache_dir cache \
  --result_dir results \
  --trials 10000

This produces:

results/tt_seed0_report.json
...
results/tt_seed9_report.json
results/ten_seed_results.csv
results/ten_seed_summary.json

The leaderboard values should be taken from:

results/ten_seed_summary.json

Notes

This method is not a GNN. It is a feature-fusion method based on temporal high-order structural signals.

The main contribution is the combination of:

weighted structural closure
temporal two-hop path strength
capped temporal three-hop path strength
direct collaboration memory
semantic similarity
transductive ogbl-collab test graph construction

under a reproducible 10-seed OGB evaluation protocol.
