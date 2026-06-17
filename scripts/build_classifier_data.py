#!/usr/bin/env python3
"""build_classifier_data.py — build HIL-SERL reward-classifier data from converted demos.

train_reward_classifier.py (upstream/hil-serl/examples) globs:
    classifier_data/*success*.pkl   -> positive (seated/inserted) transitions
    classifier_data/*failure*.pkl   -> negative (not-seated) transitions
each a LIST of SERL transition dicts; it reads obs[key] for key in config.classifier_keys.

This selects frames from the already-converted SERL19 demos
(/home/robot/hilserl-fr3/demos/serl19/*.pkl) so the classifier sees the SAME
preprocessed images as online RL (Option A: full-frame). Selection is the ONLY
task-specific knob and is exposed as CLI params so it can be tuned from the
calibration/scope findings without touching the structure.

Positives  = last  --seated-last-k frames of each SUCCESS demo (plug seated).
Negatives  = (a) the first --neg-frac fraction of each SUCCESS demo (clearly
                 not seated), optionally restricted to the approach phase, PLUS
             (b) ALL frames of every FAIL demo,
             subsampled to --neg-per-pos x (#positives) for class balance.

Outputs two pkls into --out-dir (default classifier_data/), names containing
'success' / 'failure' so the trainer's glob picks them up. Pure selection +
copy: no image reprocessing (the SERL19 demos are already in deploy format).
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import random

import numpy as np


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def seated_frames(demo, last_k, z_thresh):
    """Return indices of seated frames in one success demo.

    Default rule: the last `last_k` transitions (terminal reward=1 is the last).
    If z_thresh is not None, instead take every frame whose RelativeFrame z
    (state[0][2], i.e. descent below the reset/approach pose) is <= z_thresh.
    """
    n = len(demo)
    if z_thresh is not None:
        idx = [i for i, tr in enumerate(demo)
               if float(np.asarray(tr["observations"]["state"]).reshape(-1)[2]) <= z_thresh]
        # always include the terminal frame
        if (n - 1) not in idx:
            idx.append(n - 1)
        return sorted(set(idx))
    return list(range(max(0, n - last_k), n))


def approach_negatives(demo, neg_frac):
    """Early (clearly not-seated) frames of a success demo: first neg_frac fraction."""
    n = len(demo)
    cut = int(round(neg_frac * n))
    return list(range(0, max(1, cut)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serl19-dir", default="/home/robot/hilserl-fr3/demos/serl19")
    ap.add_argument("--out-dir", default=None,
                    help="default: <repo>/classifier_data (trainer uses os.getcwd()/classifier_data)")
    ap.add_argument("--seated-last-k", type=int, default=5)
    ap.add_argument("--seated-z-thresh", type=float, default=None,
                    help="if set, seated = RelativeFrame z (state[2]) <= this; overrides last-k")
    ap.add_argument("--neg-frac", type=float, default=0.5,
                    help="fraction of each success demo (from the start) used as clear negatives")
    ap.add_argument("--neg-per-pos", type=float, default=3.0,
                    help="target #negatives = this x #positives (class balance via subsample)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = args.out_dir or os.path.join(os.getcwd(), "classifier_data")
    os.makedirs(out_dir, exist_ok=True)

    succ = sorted(glob.glob(os.path.join(args.serl19_dir, "*success*.pkl")))
    fail = sorted(glob.glob(os.path.join(args.serl19_dir, "*fail*.pkl")))
    assert succ, f"no success demos in {args.serl19_dir}"

    positives, negatives = [], []
    for p in succ:
        d = load(p)
        si = seated_frames(d, args.seated_last_k, args.seated_z_thresh)
        ni = approach_negatives(d, args.neg_frac)
        positives.extend(d[i] for i in si)
        negatives.extend(d[i] for i in ni)
    for p in fail:
        d = load(p)
        negatives.extend(d)  # every frame of a failed insertion is a negative

    # class balance: subsample negatives to neg_per_pos x positives
    target_neg = int(round(args.neg_per_pos * len(positives)))
    if len(negatives) > target_neg:
        negatives = rng.sample(negatives, target_neg)

    pos_path = os.path.join(out_dir, "plug_insertion_success.pkl")
    neg_path = os.path.join(out_dir, "plug_insertion_failure.pkl")
    with open(pos_path, "wb") as f:
        pickle.dump(positives, f)
    with open(neg_path, "wb") as f:
        pickle.dump(negatives, f)

    print(f"positives (seated): {len(positives)}  -> {pos_path}")
    print(f"negatives (not-seated, balanced): {len(negatives)}  -> {neg_path}")
    print(f"  selection: seated={'z<=%.3f' % args.seated_z_thresh if args.seated_z_thresh is not None else 'last %d' % args.seated_last_k}"
          f"  neg_frac={args.neg_frac}  neg_per_pos={args.neg_per_pos}")


if __name__ == "__main__":
    main()
