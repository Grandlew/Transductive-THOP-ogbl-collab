# Transductive T-HOP Fusion for ogbl-collab

This repository contains the code for Transductive T-HOP Fusion, a non-neural temporal high-order path fusion method for the OGB ogbl-collab link prediction benchmark.

## Result

Official 10-seed result over seeds `0–9`:

| Dataset | Metric | Validation | Test |
|---|---:|---:|---:|
| ogbl-collab | Hits@50 | 0.6962 ± 0.0003 | 0.6925 ± 0.0010 |

The reported standard deviation is the unbiased standard deviation over 10 seeds.

## Method summary

Transductive T-HOP Fusion ranks candidate collaboration links using:

- weighted Adamic-Adar features
- resource allocation features
- weighted common-neighbor closure
- recency-weighted two-hop temporal paths
- recency-weighted capped three-hop temporal paths
- direct collaboration memory
- semantic cosine similarity from OGB node features

The final score is a validation-selected linear fusion of these features.

## Evaluation protocol

For validation:

input graph = training positive edges only

target = validation positive/negative edges.

For final test inference:

input graph = training positive edges + validation positive edges

target = test positive/negative edges
