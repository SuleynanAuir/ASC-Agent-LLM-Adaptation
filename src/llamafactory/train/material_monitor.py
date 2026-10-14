"""Opt-in local/W&B diagnostics and fixed held-out LoRA comparisons."""

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import TrainerCallback


def write_json(path, data):
    """Atomically publish readable, strict JSON including non-finite markers."""
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class MaterialMonitorCallback(TrainerCallback):
    """Generate comparisons in the training process without a second GPU model."""

    def __init__(self, cases_path):
        self.cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
        self.rows = []
        self.baseline = None
        self.started = time.time()
        self.max_wall_seconds = int(os.environ.get("MATERIAL_MAX_WALL_SECONDS", "0"))

    def _root(self, args):
        root = Path(args.output_dir) / "diagnostics"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _wandb(self, payload):
        import wandb

        if wandb.run is not None:
            try:
                wandb.log(payload)
            except Exception as error:
                print(f"[material-monitor] W&B upload failed; local outputs retained: {error}", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.max_wall_seconds and time.time() - self.started >= self.max_wall_seconds:
            # Finish an optimizer step, evaluate and save before a graceful stop.
            control.should_evaluate = True
            control.should_save = True
            control.should_training_stop = True
            print(f"[material-monitor] Wall-time limit reached at step {state.global_step}; saving.", flush=True)
        return control

    def _generate(self, args, model, tokenizer, baseline=False):
        was_training = model.training
        responses = []
        device = next(model.parameters()).device
        dtype = torch.float16 if args.fp16 else torch.bfloat16
        adapter_context = model.disable_adapter() if baseline else nullcontext()
        try:
            model.eval()
            with adapter_context, torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                for case in self.cases:
                    messages = []
                    if case.get("system"):
                        messages.append({"role": "system", "content": case["system"]})
                    prompt = case["instruction"]
                    if case.get("input"):
                        prompt += "\n" + case["input"]
                    messages.append({"role": "user", "content": prompt})
                    ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        enable_thinking=False, return_tensors="pt",
                    ).to(device)
                    # Preserve exactly the same full prompt for both models.
                    if ids.shape[1] > 1536:
                        raise ValueError(f"Comparison prompt too long: {case['case_id']}")
                    start = time.time()
                    generated = model.generate(
                        input_ids=ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=384, do_sample=False, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )[0, ids.shape[1]:]
                    responses.append({
                        "case_id": case["case_id"],
                        "question": prompt,
                        "reference": case["output"],
                        "answer": tokenizer.decode(generated, skip_special_tokens=True),
                        "new_tokens": len(generated),
                        "hit_token_limit": len(generated) >= 384,
                        "seconds": round(time.time() - start, 3),
                    })
                    print(f"[material-monitor] {'base' if baseline else 'adapter'} {case['case_id']} saved", flush=True)
        finally:
            model.train(was_training)
            torch.cuda.empty_cache()
        return responses

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        import wandb

        root = self._root(args)
        if wandb.run:
            write_json(root / "wandb_run.json", {"url": wandb.run.url, "id": wandb.run.id})
        baseline_path = root / "baseline.json"
        if baseline_path.exists():
            self.baseline = json.loads(baseline_path.read_text())
        else:
            tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
            self.baseline = self._generate(args, kwargs["model"], tokenizer, baseline=True)
            write_json(baseline_path, self.baseline)
        self.rows = list(state.log_history)
        write_json(root / "comparison_protocol.json", {
            "cases": [c["case_id"] for c in self.cases],
            "split": "explicit holdout; identical normalized questions excluded from train",
            "decoding": "greedy, enable_thinking=false, max_new_tokens=384",
            "baseline": "same Qwen3-8B with LoRA adapter disabled",
            "warning": "Qualitative samples only; generated answers are not factual accuracy scores.",
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        row = {"step": state.global_step, "wall_seconds": time.time() - self.started, **logs}
        self.rows.append(row)
        root = self._root(args)
        write_json(root / "metrics.json", self.rows)
        print("[material-monitor] " + json.dumps(row, ensure_ascii=False), flush=True)
        nonfinite = {k: v for k, v in logs.items() if isinstance(v, (float, int)) and not math.isfinite(v)}
        if nonfinite:
            print(f"[material-monitor] NONFINITE step={state.global_step}: {nonfinite}", flush=True)
        if state.global_step > 0 and (state.global_step in (1, 5) or state.global_step % 25 == 0) and "loss" in logs:
            self._plot(args, state)

    def _plot(self, args, state):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        root = self._root(args)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, key, title in zip(axes.flat, ["loss", "grad_norm", "eval_loss", "learning_rate"],
                                 ["Training loss", "Gradient norm (before clipping)", "Held-out loss", "Learning rate"]):
            rows = [r for r in self.rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
            if rows:
                ax.plot([r["step"] for r in rows], [r[key] for r in rows], linewidth=1,
                        marker="." if key == "eval_loss" or len(rows) == 1 else None)
                if key == "loss" and len(rows) >= 10:
                    smooth = [sum(x[key] for x in rows[max(0, i-9):i+1]) / len(rows[max(0, i-9):i+1]) for i in range(len(rows))]
                    ax.plot([r["step"] for r in rows], smooth, label="10-record mean")
                    ax.legend()
            ax.set(title=title, xlabel="Optimizer step")
            ax.grid(alpha=.25)
        fig.suptitle(f"Qwen3-8B material QA / step {state.global_step}")
        path = root / f"curves-step-{state.global_step:04d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self._wandb({"diagnostics/curves": wandb.Image(str(path)), "train/global_step": state.global_step})

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        import wandb

        root = self._root(args)
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        tuned = self._generate(args, kwargs["model"], tokenizer)
        base = {r["case_id"]: r for r in self.baseline}
        comparison = [{**r, "base_answer": base[r["case_id"]]["answer"],
                       "base_hit_token_limit": base[r["case_id"]]["hit_token_limit"],
                       "step": state.global_step} for r in tuned]
        write_json(root / f"comparison-step-{state.global_step:04d}.json", comparison)
        table = wandb.Table(columns=["case_id", "question", "reference", "base_answer", "tuned_answer", "tuned_token_limit"])
        for r in comparison:
            table.add_data(r["case_id"], r["question"], r["reference"], r["base_answer"], r["answer"], r["hit_token_limit"])
        self._wandb({f"comparisons/step_{state.global_step}": table, "train/global_step": state.global_step})
        self._plot(args, state)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._plot(args, state)
        nonfinite = [{"step": r.get("step"), "metric": k, "value": str(v)} for r in self.rows
                     for k, v in r.items() if isinstance(v, (int, float)) and not math.isfinite(v)]
        write_json(self._root(args) / "summary.json", {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "eval_history": [r for r in self.rows if "eval_loss" in r],
            "nonfinite": nonfinite,
            "note": "Assess trends jointly; no universal gradient or loss-drop threshold proves stability or accuracy.",
        })
