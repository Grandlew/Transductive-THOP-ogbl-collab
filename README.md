# Transductive T-HOP Fusion for ogbl-collab

This repository contains the official implementation of **Transductive T-HOP Fusion** for the OGB `ogbl-collab` link prediction benchmark.

Transductive T-HOP Fusion is a non-neural temporal high-order path fusion method. It combines weighted structural closure, temporal two-hop and three-hop path signals, direct collaboration memory, and semantic similarity.

---

## Installation requirements

```bash
numpy>=1.22
pandas>=1.3
torch>=2.0
ogb>=1.3.6
tqdm>=4.64
scipy>=1.8
