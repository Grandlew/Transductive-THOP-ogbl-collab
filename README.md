# Transductive T-HOP Fusion for ogbl-collab

This repository contains the code used for an OGB `ogbl-collab` leaderboard submission.

## Method

Transductive T-HOP Fusion is a non-neural temporal high order path fusion method for collaboration link prediction.

The model ranks candidate links using:

- weighted Adamic-Adar / Resource Allocation features
- weighted common-neighbor closure
- recency-weighted two-hop temporal path features
- recency-weighted capped three-hop temporal path features
- direct collaboration memory
- semantic cosine similarity from OGB node features

Feature weights are selected on the official validation split.

## Evaluation protocol

For validation, the input graph contains only training edges.

For final test inference, validation positive edges are added to the observed graph:

- validation graph: train only
- test graph: train + validation positives

This follows the common `ogbl-collab` transductive inference protocol where validation edges may be used as input for final test prediction after model selection.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt