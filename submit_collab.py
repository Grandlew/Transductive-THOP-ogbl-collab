import argparse
import json
import os
import pickle

import numpy as np
import torch
from ogb.linkproppred import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--submission_dir", type=str, default="submission")
    args = parser.parse_args()

    os.makedirs(args.submission_dir, exist_ok=True)

    with open(args.pred_path, "rb") as f:
        obj = pickle.load(f)

    test_pos = obj["test_pos"]
    test_neg = obj["test_neg"]

    evaluator = Evaluator(name="ogbl-collab")

    input_dict = {
        "y_pred_pos": torch.from_numpy(test_pos.astype(np.float32)),
        "y_pred_neg": torch.from_numpy(test_neg.astype(np.float32)),
    }

    local_eval = evaluator.eval(input_dict)
    print("Local test eval:", local_eval)

    try:
        evaluator.save_test_submission(input_dict, args.submission_dir)
        print("Saved official OGB submission to:", args.submission_dir)
    except Exception as e:
        print("save_test_submission failed or unavailable:", type(e).__name__, e)

    fallback_path = os.path.join(
        args.submission_dir, "test_predictions_seed909.pt")
    torch.save(input_dict, fallback_path)
    print("Saved fallback predictions:", fallback_path)

    meta = {
        "model": "Transductive T-HOP Fusion seed 909",
        "pred_path": args.pred_path,
        "local_test_eval": local_eval,
    }

    meta_path = os.path.join(args.submission_dir, "submission_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved metadata:", meta_path)


if __name__ == "__main__":
    main()
