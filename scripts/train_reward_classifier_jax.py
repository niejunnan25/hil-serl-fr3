#!/usr/bin/env python3
"""Train the SERL reward classifier natively in JAX/Flax.

Why not `scripts/train_reward_classifier.py` + `convert_classifier_pt_to_jax.py`:
the torch model there is a ResNet-18-style network with BatchNorm, while the
deployed classifier (`serl_launcher.networks.reward_classifier.create_classifier`)
is a ResNet-10 with GroupNorm and learned spatial embeddings. No weight mapping
between those two architectures preserves the function, and the converter also
imports a `make_classifier_func` that does not exist in this checkout. Training
the deployed architecture directly removes both problems: the saved checkpoint
is exactly what `load_classifier_func` restores at deploy time.

Two choices that matter more than the optimizer:

  * the validation split is by episode, not by image. Consecutive frames of one
    demo are near-duplicates, so an image-wise split reports a score that never
    reflects an unseen trajectory.
  * the pretrained encoder is frozen by default (stop_gradient), so only the
    spatial embeddings and the head adapt. With a few hundred positives that is
    the difference between transfer and memorisation.

Usage:
  python scripts/train_reward_classifier_jax.py \
      --data-dir data/classifier_front --output-dir classifier_ckpt_front \
      --epochs 60 --batch-size 64 --val-fraction 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_KEYS = ["side_classifier"]
IMAGE_SIZE = 128
SELECTION_RULE = "maximize validation (F1 at fixed deploy threshold, -FPR, AUC)"


def load_manifest(data_dir: Path):
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path} not found; build the dataset first")
    return json.loads(manifest_path.read_text())


def input_contract(manifest, image_key: str, image_size: int):
    """Validate the declared camera input before constructing the model."""
    if not image_key or type(image_size) is not int or image_size <= 0:
        raise SystemExit("image-key must be nonempty and image-size must be a positive integer")
    if manifest.get("classifier_key") != image_key:
        raise SystemExit(
            f"manifest classifier_key={manifest.get('classifier_key')!r} "
            f"does not match --image-key={image_key!r}")
    shape = [image_size, image_size, 3]
    declared_shape = manifest.get("image_shape")
    if declared_shape is not None and declared_shape != shape:
        raise SystemExit(f"manifest image_shape={declared_shape} does not match expected {shape}")
    profile = manifest.get("image_profile")
    if isinstance(profile, list) and len(profile) == 1:
        profile = profile[0]
    if not isinstance(profile, str) or not profile:
        raise SystemExit("manifest image_profile must identify exactly one image profile")
    return {"image_key": image_key, "image_shape": shape, "image_dtype": "uint8",
            "image_profile": profile}


def load_images(data_dir: Path, manifest, *, image_key=IMAGE_KEYS[0], image_size=IMAGE_SIZE):
    """Return exact RGB pixels X (N,H,W,3), labels and episode ids; never resize."""
    contract = input_contract(manifest, image_key, image_size)
    expected_shape = tuple(contract["image_shape"])
    images, labels, episodes = [], [], []
    for label, key in ((1.0, "positive"), (0.0, "negative")):
        for item in manifest["files"][key]:
            path = data_dir / key / item["file"]
            with Image.open(path) as handle:
                array = np.asarray(handle)
            if array.shape != expected_shape or array.dtype != np.uint8:
                raise SystemExit(
                    f"{path}: unexpected shape/dtype {array.shape} {array.dtype}; "
                    f"expected {expected_shape} uint8")
            images.append(array)
            labels.append(label)
            episodes.append(item["episode"])
    if not images:
        raise SystemExit(f"{data_dir}: manifest contains no classifier images")
    return (np.stack(images), np.asarray(labels, dtype=np.float32), np.asarray(episodes))


def encoder_mask(params):
    """Mark only leaves inside a named pretrained_encoder subtree."""
    import jax

    def marked(path, _leaf):
        return any(getattr(part, "key", None) == "pretrained_encoder" for part in path)

    return jax.tree_util.tree_map_with_path(marked, params)


def make_optimizer(learning_rate, weight_decay, *, frozen_mask=None):
    """Keep frozen parameters exact, including AdamW's decoupled weight decay."""
    import optax

    transforms = [optax.clip_by_global_norm(5.0),
                  optax.adamw(learning_rate, weight_decay=weight_decay)]
    if frozen_mask is not None:
        transforms.append(optax.masked(optax.set_to_zero(), frozen_mask))
    return optax.chain(*transforms)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC with tie correction."""
    labels = np.asarray(labels)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    index = 0
    while index < len(scores):
        end = index
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        ranks[order[index:end + 1]] = 0.5 * (index + end) + 1.0
        index = end + 1
    positives = labels == 1
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion(labels: np.ndarray, probabilities: np.ndarray, threshold: float):
    predicted = probabilities >= threshold
    actual = labels == 1
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    tn = int(np.sum(~predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0}


def checkpoint_rank(at_deploy, auc):
    """Prefer fixed-threshold deployment quality before threshold-free ranking."""
    return (at_deploy["f1"], -at_deploy["false_positive_rate"], auc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/classifier_front"))
    parser.add_argument("--output-dir", type=Path, default=Path("classifier_ckpt_front"))
    parser.add_argument("--image-key", default=IMAGE_KEYS[0],
                        help="Observation key recorded in manifest.classifier_key (default side_classifier)")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                        help="Exact square RGB input size; images are never resized (default 128)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deploy-threshold", type=float, default=0.78)
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="Also update the pretrained encoder (more capacity, more overfit risk)")
    parser.add_argument("--smoke", action="store_true", help="Two epochs on a small subset; prints shapes only")
    args = parser.parse_args()

    import flax
    import flax.training.checkpoints as flax_checkpoints
    import jax
    import jax.numpy as jnp
    import optax
    from serl_launcher.networks.reward_classifier import create_classifier, load_classifier_func

    manifest = load_manifest(args.data_dir)
    print(f"[data] profile={manifest.get('image_profile')} rule={manifest['label_rule']['positive']}")
    contract = input_contract(manifest, args.image_key, args.image_size)
    X, y, episodes = load_images(args.data_dir, manifest, image_key=args.image_key,
                                image_size=args.image_size)
    image_keys = [args.image_key]
    print(f"[input] {json.dumps(contract, ensure_ascii=False)}")
    print(f"[data] images={len(X)} positives={int(y.sum())} negatives={int((1 - y).sum())} "
          f"episodes={len(set(episodes.tolist()))}")

    # Episode-level split: no frame of a validation episode is ever trained on.
    unique_episodes = np.array(sorted(set(episodes.tolist())))
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(unique_episodes)
    n_val = max(1, int(round(len(shuffled) * args.val_fraction)))
    val_episodes = set(shuffled[:n_val].tolist())
    val_mask = np.isin(episodes, list(val_episodes))
    print(f"[split] train episodes={len(unique_episodes) - n_val} val episodes={n_val} "
          f"| train images={int((~val_mask).sum())} val images={int(val_mask.sum())}")

    if args.smoke:
        args.epochs = 2
        train_index = np.nonzero(~val_mask)[0][:128]
        val_index = np.nonzero(val_mask)[0][:64]
    else:
        train_index = np.nonzero(~val_mask)[0]
        val_index = np.nonzero(val_mask)[0]

    sample = {key: np.zeros((1, args.image_size, args.image_size, 3), dtype=np.uint8)
              for key in image_keys}
    key = jax.random.PRNGKey(args.seed)
    classifier = create_classifier(key, sample, image_keys, n_way=2)
    apply_fn = classifier.apply_fn
    freeze = not args.unfreeze_encoder
    trainable = sum(x.size for x in jax.tree_util.tree_leaves(classifier.params))
    print(f"[model] parameters={trainable:,} freeze_encoder={freeze}")

    schedule = optax.cosine_decay_schedule(args.lr, decay_steps=max(1, args.epochs * 20), alpha=0.05)
    frozen_mask = encoder_mask(classifier.params) if freeze else None
    optimizer = make_optimizer(schedule, args.weight_decay, frozen_mask=frozen_mask)

    # stop_gradient skips the encoder backward pass; the optimizer mask also
    # removes weight decay and any optimizer-state contribution to its updates.

    def forward(params, batch, train, rng=None):
        effective = params
        if frozen_mask is not None:
            effective = jax.tree_util.tree_map(
                lambda mask, leaf: jax.lax.stop_gradient(leaf) if mask else leaf,
                frozen_mask, params)
        rngs = {"dropout": rng} if train else None
        return apply_fn({"params": effective}, {image_keys[0]: batch[:, None]},
                        train=train, rngs=rngs)

    pos_weight = float((1 - y[train_index]).sum() / max(1.0, y[train_index].sum()))

    def loss_fn(params, batch, targets, rng):
        logits = jnp.reshape(forward(params, batch, True, rng), (-1,))
        weights = jnp.where(targets > 0.5, pos_weight, 1.0)
        losses = optax.sigmoid_binary_cross_entropy(logits, targets)
        return jnp.mean(losses * weights)

    train_step = jax.jit(jax.value_and_grad(loss_fn))
    eval_step = jax.jit(lambda params, batch: jnp.reshape(forward(params, batch, False), (-1,)))

    state = {"params": classifier.params, "opt_state": optimizer.init(classifier.params), "step": 0}
    batch_size = min(args.batch_size, len(train_index))
    history, best = [], {"rank": None, "auc": -1.0, "epoch": -1, "params": None}
    print(f"[selection] {SELECTION_RULE}; threshold={args.deploy_threshold}")

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        order = np.random.default_rng(args.seed + epoch).permutation(train_index)
        epoch_loss, batches = 0.0, 0
        for begin in range(0, len(order), batch_size):
            index = order[begin:begin + batch_size]
            if len(index) < 2:
                continue
            loss, grads = train_step(state["params"], jnp.asarray(X[index]), jnp.asarray(y[index]),
                                     jax.random.PRNGKey(state["step"]))
            updates, state["opt_state"] = optimizer.update(grads, state["opt_state"], state["params"])
            state["params"] = optax.apply_updates(state["params"], updates)
            state["step"] += 1
            epoch_loss += float(loss)
            batches += 1

        val_logits = np.concatenate([np.asarray(eval_step(state["params"], jnp.asarray(X[val_index[i:i + 256]])))
                                     for i in range(0, len(val_index), 256)])
        val_probabilities = 1.0 / (1.0 + np.exp(-val_logits))
        val_labels = y[val_index]
        auc = roc_auc(val_labels, val_probabilities)
        at_deploy = confusion(val_labels, val_probabilities, args.deploy_threshold)
        entry = {"epoch": epoch, "train_loss": epoch_loss / max(1, batches), "val_auc": auc,
                 "val_f1_at_deploy_threshold": at_deploy["f1"],
                 "val_fpr_at_deploy_threshold": at_deploy["false_positive_rate"],
                 "seconds": time.time() - started}
        history.append(entry)
        marker = ""
        rank = checkpoint_rank(at_deploy, auc)
        if best["rank"] is None or rank > best["rank"]:
            best = {"rank": rank, "auc": auc, "epoch": epoch,
                    "params": jax.tree_util.tree_map(lambda v: v.copy(), state["params"])}
            marker = "  <- best"
        print(f"[epoch {epoch:3d}] loss={entry['train_loss']:.4f} val_auc={auc:.4f} "
              f"val_f1@{args.deploy_threshold}={at_deploy['f1']:.4f} ({entry['seconds']:.1f}s){marker}")

        if epoch - best["epoch"] >= args.patience:
            print(f"[stop] no improvement for {args.patience} epochs")
            break

    # Save with a freshly built TrainState so the checkpoint structure matches
    # exactly what create_classifier produces at deploy time.
    deployed = create_classifier(jax.random.PRNGKey(0), sample, image_keys, n_way=2)
    deployed = deployed.replace(params=best["params"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Orbax refuses relative checkpoint paths.
    flax_checkpoints.save_checkpoint(str(args.output_dir.resolve()), deployed,
                                     step=best["epoch"], keep=1, overwrite=True)

    # Verify through the deployment loader, not through our own training path.
    restored = load_classifier_func(key=jax.random.PRNGKey(0), sample=sample,
                                    image_keys=image_keys,
                                    checkpoint_path=str(args.output_dir.resolve()))
    logits = np.concatenate([np.asarray(jnp.reshape(restored({image_keys[0]: X[val_index[i:i + 256]][:, None]}), (-1,)))
                             for i in range(0, len(val_index), 256)])
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    val_labels = y[val_index]
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sweep = [confusion(val_labels, probabilities, float(t)) for t in thresholds]
    best_f1 = max(sweep, key=lambda item: item["f1"])
    metrics = {
        "input_contract": contract,
        "selection_rule": SELECTION_RULE,
        "images": {"total": int(len(X)), "positives": int(y.sum()), "negatives": int((1 - y).sum())},
        "episodes": {"total": int(len(unique_episodes)), "train": int(len(unique_episodes) - n_val),
                     "val": int(n_val), "val_ids": sorted(val_episodes)},
        "data_manifest_rule": manifest["label_rule"],
        "freeze_encoder": freeze, "epochs_run": len(history), "best_epoch": best["epoch"],
        "val_auc": best["auc"],
        "confusion_at_deploy_threshold": confusion(val_labels, probabilities, args.deploy_threshold),
        "best_threshold_by_f1": best_f1,
        "checkpoint": str(args.output_dir.resolve()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
