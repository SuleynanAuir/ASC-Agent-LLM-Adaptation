"""Prepare a reproducible holdout and a measured 3-4 hour WebUI experiment."""

import hashlib
import json
import random
import re
from pathlib import Path

import yaml


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo / "test/wpFuh_5cOk4Y/alpaca.json"
    data = json.loads(source.read_text())
    destination = repo / "data/material_diagnostic_20261014"
    destination.mkdir(parents=True, exist_ok=True)
    groups = {}
    for i, row in enumerate(data):
        key = re.sub(r"\s+", "", row["instruction"] + row.get("input", "")).casefold()
        groups.setdefault(key, []).append(i)
    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    holdout_ids = set()
    for key in keys:
        holdout_ids.update(groups[key])
        if len(holdout_ids) >= 96:
            break
    train = [r for i, r in enumerate(data) if i not in holdout_ids]
    validation = [r for i, r in enumerate(data) if i in holdout_ids]
    candidates = [(i, r) for i, r in enumerate(data) if i in holdout_ids and len(r["output"]) <= 2500]
    random.Random(123).shuffle(candidates)
    cases = [{"case_id": f"qa-{i:05d}", **row} for i, row in candidates[:6]]
    if len(cases) != 6:
        raise ValueError("Need six comparison cases")
    for name, value in [("train.json", train), ("validation.json", validation), ("comparison_cases.json", cases)]:
        path = destination / name
        text = json.dumps(value, ensure_ascii=False)
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Existing split differs; refusing to overwrite {path}")
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    info = {}
    for key, name in [("material_train", "train.json"), ("material_validation", "validation.json")]:
        info[key] = {"file_name": name, "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}}
    (destination / "dataset_info.json").write_text(json.dumps(info, indent=2))
    manifest = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_count": len(data),
                "train_count": len(train), "validation_count": len(validation), "seed": 42,
                "validation_source_indices": sorted(holdout_ids), "group_key": "whitespace-normalized instruction+input",
                "limitations": "Question-disjoint, not document-disjoint; near-duplicate topics may remain."}
    (destination / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    config = yaml.safe_load((repo / "examples/train_lora/qwen3_8b_material_qa_autodl.yaml").read_text())
    config.update(dataset="material_train", eval_dataset="material_validation", dataset_dir=str(destination),
                  val_size=0., max_steps=800, logging_steps=1, logging_nan_inf_filter=False,
                  eval_strategy="steps", eval_steps=200, eval_on_start=True, do_eval=True,
                  save_strategy="steps", save_steps=200, save_total_limit=4,
                  load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                  prediction_loss_only=True, fp16_full_eval=True, bf16=False, fp16=True, tf32=False,
                  output_dir="/root/autodl-tmp/saves/qwen3-8b/lora/material-diagnostic-3h4h-20261014",
                  run_name="qwen3-8b-material-diagnostic-3h4h-20261014", report_to="wandb",
                  optim="adamw_torch", overwrite_cache=False)
    config.pop("num_train_epochs", None)
    config_path = repo / "examples/train_lora/qwen3_8b_material_diagnostic_autodl.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    preset = yaml.safe_load((repo / "llamaboard_config/qwen3_8b_material_qa.yaml").read_text())
    preset.update({"train.dataset_dir": str(destination), "train.dataset": ["material_train"],
                   "train.max_samples": str(len(train)), "train.val_size": 0., "train.logging_steps": 1,
                   "train.compute_type": "fp16", "train.save_steps": 200, "train.output_dir": config["output_dir"],
                   "train.config_path": "qwen3_8b_material_diagnostic.yaml"})
    extras = json.loads(preset["train.extra_args"])
    for key in ["eval_dataset", "do_eval", "max_steps", "logging_nan_inf_filter", "eval_strategy", "eval_steps", "eval_on_start",
                "save_strategy", "save_steps", "save_total_limit", "load_best_model_at_end", "metric_for_best_model",
                "greater_is_better", "prediction_loss_only", "fp16_full_eval", "tf32", "run_name", "overwrite_cache"]:
        extras[key] = config[key]
    preset["train.extra_args"] = json.dumps(extras)
    (repo / "llamaboard_config/qwen3_8b_material_diagnostic.yaml").write_text(yaml.safe_dump(preset, sort_keys=False, allow_unicode=True))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Config:", config_path)


if __name__ == "__main__":
    main()
