"""RoboMeter native inference adapter, adapted from the verified SEAL-0 service.

Reference: goal-rm50-20260919-r1 at 8c5985ed02a5e73eebe43cdf19465ccb25049f42.
Torch and the separately installed RoboMeter package load only on construction.
"""
from __future__ import annotations
import logging
from pathlib import Path
import sys
import threading
from typing import Any, Sequence
LOGGER = logging.getLogger(__name__)

class RoboMeterNativeBackend:
    """In-process RoboMeter inference backend.

    This follows RoboMeter's policy-learning relabel path: each query gets its
    own prefix sample, then those samples are batched for one model forward. It
    avoids the extra official eval-server process and repeated multipart .npy
    transfer while preserving the prefix-per-query reward semantics.
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        forward_batch_size: int,
        robometer_root: str | None = None,
    ) -> None:
        if robometer_root:
            root = Path(robometer_root).expanduser().resolve()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))

        import torch
        from robometer.models.utils import convert_bins_to_continuous
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        self.torch = torch
        self.convert_bins_to_continuous = convert_bins_to_continuous
        self.device = torch.device(device)
        self.forward_batch_size = max(1, int(forward_batch_size))

        LOGGER.info("loading RoboMeter native backend from %s on %s", model_path, self.device)
        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=str(model_path),
            device=self.device,
        )
        model_dtype = getattr(model, "model_dtype", next(model.parameters()).dtype)
        # Checkpoints can contain a mixed set of fp32 vision parameters and
        # bf16 language/head parameters.  The native eval path does not run
        # under Accelerate autocast, so normalize every floating parameter and
        # buffer to the checkpoint's declared compute dtype before inference.
        model = model.to(device=self.device, dtype=model_dtype)
        model.eval()

        self.exp_config = exp_config
        self.tokenizer = tokenizer
        self.model = model
        self.model_dtype = model_dtype
        self.batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
        progress_loss_type = getattr(exp_config.loss, "progress_loss_type", "l2")
        self.is_discrete_mode = str(progress_loss_type).lower() == "discrete"
        self.num_bins = int(
            getattr(
                exp_config.loss,
                "progress_discrete_bins",
                getattr(exp_config.model, "progress_discrete_bins", 10),
            )
        )
        self.model_type = str(getattr(exp_config.model, "model_type", ""))
        self.lock = threading.Lock()
        LOGGER.info("RoboMeter native backend compute dtype: %s", self.model_dtype)

    def predict_progress_samples(self, samples: Sequence[dict[str, Any]]) -> list[list[float]]:
        if not samples:
            return []
        buckets: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for sample_idx, sample in enumerate(samples):
            buckets.setdefault(self._sample_frame_count(sample), []).append((sample_idx, sample))

        outputs: list[list[float] | None] = [None] * len(samples)
        with self.lock:
            with self.torch.inference_mode():
                for indexed_samples in buckets.values():
                    for start in range(0, len(indexed_samples), self.forward_batch_size):
                        indexed_chunk = indexed_samples[start : start + self.forward_batch_size]
                        chunk_indices = [sample_idx for sample_idx, _sample in indexed_chunk]
                        chunk = [sample for _sample_idx, sample in indexed_chunk]
                        chunk_outputs = self._predict_same_length_chunk(chunk)
                        if len(chunk_outputs) != len(chunk_indices):
                            raise RuntimeError(
                                f"expected {len(chunk_indices)} progress outputs, got {len(chunk_outputs)}"
                            )
                        for sample_idx, sample_output in zip(chunk_indices, chunk_outputs):
                            outputs[sample_idx] = sample_output
        if any(output is None for output in outputs):
            raise RuntimeError("RoboMeter native backend did not produce all requested outputs")
        return [list(output) for output in outputs if output is not None]

    @staticmethod
    def _sample_frame_count(sample: dict[str, Any]) -> int:
        trajectory = sample.get("trajectory", {})
        frames = trajectory.get("frames") if isinstance(trajectory, dict) else None
        if hasattr(frames, "shape"):
            return int(frames.shape[0])
        return len(frames) if frames is not None else 0

    def _predict_same_length_chunk(self, chunk: Sequence[dict[str, Any]]) -> list[list[float]]:
        frame_lengths = {self._sample_frame_count(sample) for sample in chunk}
        if len(frame_lengths) > 1:
            raise RuntimeError(f"native RoboMeter batch has mixed frame lengths: {sorted(frame_lengths)}")
        batch_inputs = self.batch_collator(chunk)["progress_inputs"]
        batch_inputs = {
            key: (
                value.to(
                    device=self.device,
                    dtype=self.model_dtype if value.is_floating_point() else value.dtype,
                )
                if isinstance(value, self.torch.Tensor)
                else value
            )
            for key, value in batch_inputs.items()
        }
        return self._compute_progress_outputs(batch_inputs)

    def _compute_progress_outputs(self, batch_inputs: dict[str, Any]) -> list[list[float]]:
        if "rewind" in self.model.__class__.__name__.lower():
            model_output, _extra = self.model(
                video_embeddings=batch_inputs.get("video_embeddings"),
                text_embeddings=batch_inputs.get("text_embeddings"),
                sample_type="progress",
                timing_raw=None,
            )
        else:
            model_output, _extra = self.model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                pixel_values=batch_inputs.get("pixel_values", None),
                pixel_values_videos=batch_inputs.get("pixel_values_videos", None),
                image_grid_thw=batch_inputs.get("image_grid_thw", None),
                video_grid_thw=batch_inputs.get("video_grid_thw", None),
                second_per_grid_ts=batch_inputs.get("second_per_grid_ts", None),
                sample_type="progress",
                timing_raw=None,
            )

        progress_logits = getattr(model_output, "progress_logits", None)
        if not isinstance(progress_logits, dict):
            return [[] for _ in range(int(batch_inputs["input_ids"].shape[0]))]
        seq_a = progress_logits.get("A")
        if seq_a is None:
            return [[] for _ in range(int(batch_inputs["input_ids"].shape[0]))]

        progress_pred: list[list[float]] = []
        for item in [seq_a[idx] for idx in range(seq_a.shape[0])]:
            if self.is_discrete_mode:
                continuous = self.convert_bins_to_continuous(item.detach().cpu().float())
                progress_pred.append(continuous.numpy().flatten().tolist())
            else:
                progress_pred.append(item.detach().cpu().flatten().tolist())
        return progress_pred
