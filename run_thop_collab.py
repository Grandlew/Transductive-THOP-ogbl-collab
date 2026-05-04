
import argparse
import json
import math
import os
import pickle
import random

import numpy as np
import torch
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


def edge_key(u: int, v: int):
    u, v = int(u), int(v)
    return (u, v) if u < v else (v, u)


def recency_from_year(t, t_max, lam):
    if t_max <= 0 or float(t) <= 0:
        return 0.0
    return math.exp(-lam * (t_max - float(t)))


def build_graph_object(edge_arr, weight_arr, year_arr, num_nodes, t_max_train, name):
    t_max = float(np.max(year_arr)) if len(year_arr) else t_max_train

    adj = [set() for _ in range(num_nodes)]
    edge_w = {}
    edge_t = {}

    for i in range(edge_arr.shape[0]):
        u, v = int(edge_arr[i, 0]), int(edge_arr[i, 1])
        if u == v:
            continue

        w = float(weight_arr[i])
        y = float(year_arr[i])

        adj[u].add(v)
        adj[v].add(u)

        k = edge_key(u, v)
        edge_w[k] = edge_w.get(k, 0.0) + w
        edge_t[k] = max(edge_t.get(k, 0.0), y)

    deg = np.array([len(a) for a in adj], dtype=np.float32)

    print(f"{name}: unique_edges={len(edge_w):,}, mean_deg={deg.mean():.2f}, max_deg={deg.max():.0f}")

    return {
        "name": name,
        "adj": adj,
        "edge_w": edge_w,
        "edge_t": edge_t,
        "deg": deg,
        "t_max": t_max,
    }


TT_FEATURE_NAMES = [
    "cn", "ra", "aa", "jaccard",
    "w_cn_product", "w_cn_sqrt", "w_ra_product", "w_ra_sqrt", "w_aa_product", "w_aa_sqrt",
    "temp2_sum", "temp2_coh_sum", "temp2_max",
    "temp3_capped_sum", "temp3_coh_capped_sum", "temp3_capped_max",
    "direct_edge", "direct_weight", "direct_recency", "direct_weight_x_recency",
    "degree_u", "degree_v", "degree_product", "degree_min", "degree_max", "degree_absdiff",
    "x_cosine", "x_l1_mean", "x_l2",
    "log1p_cn", "log1p_ra", "log1p_aa",
    "log1p_w_cn_product", "log1p_w_cn_sqrt",
    "log1p_w_aa_product", "log1p_w_aa_sqrt",
    "log1p_temp2", "log1p_temp2_coh", "log1p_temp3",
    "log1p_direct_weight", "log1p_direct_weight_x_recency",
]


def graph_w(G, u, v):
    return G["edge_w"].get(edge_key(u, v), 0.0)


def graph_year(G, u, v):
    return G["edge_t"].get(edge_key(u, v), 0.0)


def graph_recency(G, u, v, lam):
    return recency_from_year(graph_year(G, u, v), G["t_max"], lam)


def graph_has_edge(G, u, v):
    return edge_key(u, v) in G["edge_w"]


def common_neighbors_graph(G, u, v):
    Nu = G["adj"][u]
    Nv = G["adj"][v]
    if len(Nu) < len(Nv):
        return [k for k in Nu if k in Nv]
    return [k for k in Nv if k in Nu]


def top_neighbors_by_temporal_strength(G, u, topk, lam):
    neigh = list(G["adj"][u])
    if len(neigh) == 0:
        return []

    vals = np.array([graph_w(G, u, v) * graph_recency(G, u, v, lam)
                    for v in neigh], dtype=np.float32)
    neigh = np.asarray(neigh, dtype=np.int64)

    if len(neigh) <= topk:
        order = np.argsort(-vals)
    else:
        part = np.argpartition(-vals, topk - 1)[:topk]
        order = part[np.argsort(-vals[part])]

    return [(int(neigh[i]), float(vals[i])) for i in order]


def compute_tt_features_np(edge_pairs, G, x_np, x_norm, cfg):
    pairs = edge_pairs.cpu().numpy() if isinstance(
        edge_pairs, torch.Tensor) else edge_pairs
    X = np.zeros((pairs.shape[0], len(TT_FEATURE_NAMES)), dtype=np.float32)

    lam = cfg["recency_lambda"]
    mu = cfg["coherence_mu"]
    topk = cfg["top_neighbors_3hop"]
    eps = 1e-9

    adj = G["adj"]
    deg = G["deg"]

    for i, (u_raw, v_raw) in enumerate(pairs):
        u, v = int(u_raw), int(v_raw)

        Nu, Nv = adj[u], adj[v]
        C = common_neighbors_graph(G, u, v)

        cn = float(len(C))
        du = float(deg[u])
        dv = float(deg[v])

        ra = 0.0
        aa = 0.0
        w_cn_product = 0.0
        w_cn_sqrt = 0.0
        w_ra_product = 0.0
        w_ra_sqrt = 0.0
        w_aa_product = 0.0
        w_aa_sqrt = 0.0

        temp2_vals = []
        temp2_coh_vals = []

        for k in C:
            dk = max(float(deg[k]), 1.0)
            log_dk = math.log(dk + 2.0)

            w_uk = graph_w(G, u, k)
            w_kv = graph_w(G, k, v)

            r_uk = graph_recency(G, u, k, lam)
            r_kv = graph_recency(G, k, v, lam)

            t_uk = graph_year(G, u, k)
            t_kv = graph_year(G, k, v)

            w_prod = w_uk * w_kv
            w_sqrt = math.sqrt(max(w_uk, 0.0) * max(w_kv, 0.0))

            ra += 1.0 / dk
            aa += 1.0 / log_dk

            w_cn_product += w_prod
            w_cn_sqrt += w_sqrt
            w_ra_product += w_prod / dk
            w_ra_sqrt += w_sqrt / dk
            w_aa_product += w_prod / log_dk
            w_aa_sqrt += w_sqrt / log_dk

            path_strength = w_sqrt * r_uk * r_kv
            coh = math.exp(-mu * abs(t_uk - t_kv)
                           ) if t_uk > 0 and t_kv > 0 else 0.0

            temp2_vals.append(path_strength)
            temp2_coh_vals.append(path_strength * coh)

        union = float(len(Nu) + len(Nv) - len(C))
        jaccard = cn / (union + eps)

        temp2_sum = float(np.sum(temp2_vals)) if temp2_vals else 0.0
        temp2_coh_sum = float(np.sum(temp2_coh_vals)
                              ) if temp2_coh_vals else 0.0
        temp2_max = float(np.max(temp2_vals)) if temp2_vals else 0.0

        u_top = top_neighbors_by_temporal_strength(G, u, topk, lam)
        v_top = top_neighbors_by_temporal_strength(G, v, topk, lam)

        temp3_vals = []
        temp3_coh_vals = []

        if len(u_top) > 0 and len(v_top) > 0:
            for a, ua_val in u_top:
                for b, bv_val in v_top:
                    if a == b:
                        continue
                    if not graph_has_edge(G, a, b):
                        continue

                    ab_val = graph_w(G, a, b) * graph_recency(G, a, b, lam)
                    s3 = ua_val * ab_val * bv_val

                    t1 = graph_year(G, u, a)
                    t2 = graph_year(G, a, b)
                    t3 = graph_year(G, b, v)

                    coh3 = math.exp(-mu * np.var([t1, t2, t3])
                                    ) if t1 > 0 and t2 > 0 and t3 > 0 else 0.0

                    temp3_vals.append(s3)
                    temp3_coh_vals.append(s3 * coh3)

        temp3_sum = float(np.sum(temp3_vals)) if temp3_vals else 0.0
        temp3_coh_sum = float(np.sum(temp3_coh_vals)
                              ) if temp3_coh_vals else 0.0
        temp3_max = float(np.max(temp3_vals)) if temp3_vals else 0.0

        direct = 1.0 if graph_has_edge(G, u, v) else 0.0
        direct_w = graph_w(G, u, v)
        direct_r = graph_recency(G, u, v, lam)
        direct_wr = direct_w * direct_r

        xu = x_np[u]
        xv = x_np[v]
        xun = x_norm[u]
        xvn = x_norm[v]

        x_cos = float(np.dot(xun, xvn))
        diff = xu - xv

        vals = [
            cn, ra, aa, jaccard,
            w_cn_product, w_cn_sqrt, w_ra_product, w_ra_sqrt, w_aa_product, w_aa_sqrt,
            temp2_sum, temp2_coh_sum, temp2_max,
            temp3_sum, temp3_coh_sum, temp3_max,
            direct, direct_w, direct_r, direct_wr,
            du, dv, du * dv, min(du, dv), max(du, dv), abs(du - dv),
            x_cos, float(np.mean(np.abs(diff))), float(
                np.sqrt(np.sum(diff * diff))),
            math.log1p(cn), math.log1p(ra), math.log1p(aa),
            math.log1p(max(w_cn_product, 0.0)),
            math.log1p(max(w_cn_sqrt, 0.0)),
            math.log1p(max(w_aa_product, 0.0)),
            math.log1p(max(w_aa_sqrt, 0.0)),
            math.log1p(max(temp2_sum, 0.0)),
            math.log1p(max(temp2_coh_sum, 0.0)),
            math.log1p(max(temp3_sum, 0.0)),
            math.log1p(max(direct_w, 0.0)),
            math.log1p(max(direct_wr, 0.0)),
        ]

        X[i] = np.asarray(vals, dtype=np.float32)

    return X


def compute_tt_features_chunked(edge_pairs, G, x_np, x_norm, cfg, desc):
    pairs = edge_pairs.cpu().numpy() if isinstance(
        edge_pairs, torch.Tensor) else edge_pairs
    chunks = []
    n = pairs.shape[0]
    chunk_size = cfg["feature_chunk_size"]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        print(f"{desc}: {end:,}/{n:,}")
        chunks.append(compute_tt_features_np(
            pairs[start:end], G, x_np, x_norm, cfg))

    return np.vstack(chunks).astype(np.float32)


def reshape_neg(cache, neg_flat):
    if cache["neg_shape"][0] == "flat":
        return neg_flat
    _, n_rows, n_neg = cache["neg_shape"]
    return neg_flat.reshape(n_rows, n_neg)


def standardize_joint(pos, neg):
    both = np.concatenate(
        [pos.reshape(-1), neg.reshape(-1)]).astype(np.float32)
    mu = both.mean()
    sd = both.std() + 1e-9
    return (pos - mu) / sd, (neg - mu) / sd


def build_bank(cache, features):
    pos_bank = {}
    neg_bank = {}

    for feat in features:
        idx = TT_FEATURE_NAMES.index(feat)
        p = cache["pos_X"][:, idx]
        n = reshape_neg(cache, cache["neg_X"][:, idx])
        pz, nz = standardize_joint(p, n)

        pos_bank[feat] = pz.astype(np.float32)
        neg_bank[feat] = nz.astype(np.float32)

    return pos_bank, neg_bank


def score_from_weights(pos_bank, neg_bank, weights):
    pos_score = np.zeros_like(next(iter(pos_bank.values())), dtype=np.float32)
    neg_score = np.zeros_like(next(iter(neg_bank.values())), dtype=np.float32)

    for feat, w in weights.items():
        pos_score += float(w) * pos_bank[feat]
        neg_score += float(w) * neg_bank[feat]

    return pos_score.astype(np.float32), neg_score.astype(np.float32)


def hits_from_scores(evaluator, pos_score, neg_score):
    return float(evaluator.eval({
        "y_pred_pos": torch.from_numpy(pos_score.astype(np.float32)),
        "y_pred_neg": torch.from_numpy(neg_score.astype(np.float32)),
    })["hits@50"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/seed909.json")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg_file = json.load(f)

    cfg = cfg_file["config"]
    seed = int(cfg_file["seed"])
    weights = cfg_file["weights"]
    fusion_features = cfg_file["features"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = PygLinkPropPredDataset(name="ogbl-collab", root=args.root)
    data = dataset[0]
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(name="ogbl-collab")

    num_nodes = data.num_nodes
    x_np = data.x.float().cpu().numpy()
    x_norm = x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-9)

    train_edge = split_edge["train"]["edge"].long()
    valid_edge = split_edge["valid"]["edge"].long()
    valid_edge_neg = split_edge["valid"]["edge_neg"].long()
    test_edge = split_edge["test"]["edge"].long()
    test_edge_neg = split_edge["test"]["edge_neg"].long()

    train_weight = split_edge["train"].get(
        "weight", torch.ones(train_edge.shape[0])).view(-1).float()
    train_year = split_edge["train"].get(
        "year", torch.zeros(train_edge.shape[0])).view(-1).float()

    train_np = train_edge.cpu().numpy()
    valid_np = valid_edge.cpu().numpy()
    train_w_np = train_weight.cpu().numpy()
    train_y_np = train_year.cpu().numpy()
    t_max_train = float(train_y_np.max()) if len(train_y_np) else 0.0

    G_valid = build_graph_object(
        train_np, train_w_np, train_y_np, num_nodes, t_max_train, "train_only_for_valid")

    valid_w = np.ones(valid_np.shape[0], dtype=np.float32)
    valid_y = np.full(valid_np.shape[0], t_max_train, dtype=np.float32)

    train_valid_edges = np.vstack([train_np, valid_np])
    train_valid_weights = np.concatenate([train_w_np, valid_w])
    train_valid_years = np.concatenate([train_y_np, valid_y])

    G_test = build_graph_object(train_valid_edges, train_valid_weights,
                                train_valid_years, num_nodes, t_max_train, "train_plus_valid_for_test")

    def build_or_load_features(split_name):
        path = os.path.join(
            args.cache_dir, f"{split_name}_tt_features_seed909.pkl")
        if os.path.exists(path) and not args.force_rebuild:
            print("Loading:", path)
            with open(path, "rb") as f:
                return pickle.load(f)

        if split_name == "valid":
            G = G_valid
            pos_edge = valid_edge.cpu().numpy()
            neg_edge = valid_edge_neg.cpu().numpy()
        elif split_name == "test":
            G = G_test
            pos_edge = test_edge.cpu().numpy()
            neg_edge = test_edge_neg.cpu().numpy()
        else:
            raise ValueError(split_name)

        if neg_edge.ndim == 2:
            neg_shape = ("flat", neg_edge.shape[0])
            neg_flat = neg_edge
        else:
            neg_shape = ("matrix", neg_edge.shape[0], neg_edge.shape[1])
            neg_flat = neg_edge.reshape(-1, 2)

        print(f"Building {split_name} features using {G['name']}")
        pos_X = compute_tt_features_chunked(
            pos_edge, G, x_np, x_norm, cfg, f"{split_name} pos")
        neg_X = compute_tt_features_chunked(
            neg_flat, G, x_np, x_norm, cfg, f"{split_name} neg")

        obj = {
            "split": split_name,
            "graph_name": G["name"],
            "pos_edge": pos_edge.astype(np.int64),
            "neg_edge": neg_edge.astype(np.int64),
            "neg_shape": neg_shape,
            "pos_X": pos_X.astype(np.float32),
            "neg_X": neg_X.astype(np.float32),
            "feature_names": TT_FEATURE_NAMES,
        }

        with open(path, "wb") as f:
            pickle.dump(obj, f)

        return obj

    valid_cache = build_or_load_features("valid")
    test_cache = build_or_load_features("test")

    valid_pos_bank, valid_neg_bank = build_bank(valid_cache, fusion_features)
    test_pos_bank, test_neg_bank = build_bank(test_cache, fusion_features)

    valid_pos, valid_neg = score_from_weights(
        valid_pos_bank, valid_neg_bank, weights)
    test_pos, test_neg = score_from_weights(
        test_pos_bank, test_neg_bank, weights)

    valid_hits = hits_from_scores(evaluator, valid_pos, valid_neg)
    test_hits = hits_from_scores(evaluator, test_pos, test_neg)

    report = {
        "model": "Transductive T-HOP Fusion seed 909",
        "dataset": "ogbl-collab",
        "metric": "Hits@50",
        "valid_hits50": valid_hits,
        "test_hits50_local": test_hits,
        "protocol": cfg_file["protocol"],
        "seed": seed,
        "weights": weights,
        "features": fusion_features,
    }

    report_path = os.path.join(args.result_dir, "tt_seed909_report.json")
    pred_path = os.path.join(args.result_dir, "tt_seed909_predictions.pkl")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(pred_path, "wb") as f:
        pickle.dump({
            "valid_pos": valid_pos,
            "valid_neg": valid_neg,
            "test_pos": test_pos,
            "test_neg": test_neg,
            "report": report,
        }, f)

    print(json.dumps(report, indent=2))
    print("Saved report:", report_path)
    print("Saved predictions:", pred_path)


if __name__ == "__main__":
    main()
