"""Serve LlamaBoard with fixed experiment settings and a live report tab."""

import html
import json
import os
from pathlib import Path

import gradio as gr
import yaml

from llamafactory.webui.interface import create_ui
from llamafactory.webui.common import save_config


REPO = Path(__file__).resolve().parents[1]
PRESET = REPO / "llamaboard_config/qwen3_8b_material_diagnostic.yaml"
os.environ["LLAMABOARD_PRESET"] = str(PRESET)
preset = yaml.safe_load(PRESET.read_text())
save_config(lang="zh", hub_name="huggingface", model_name=preset["top.model_name"], model_path=preset["top.model_path"])
output_dir = Path(preset["train.output_dir"])
reports = output_dir / "diagnostics"


def read_reports():
    if not reports.exists():
        return "等待训练启动；基模回答会在第一个优化 step 前生成。", [], ""
    metrics_path = reports / "metrics.json"
    rows = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    url_path = reports / "wandb_run.json"
    url = json.loads(url_path.read_text()).get("url") if url_path.exists() else None
    latest = rows[-1] if rows else {}
    text = f"最新指标：`{json.dumps(latest, ensure_ascii=False)}`\n\n"
    if url:
        text += f"[打开 W&B 运行页面]({url})\n\n"
    text += "梯度图为裁剪前范数。回答采用同一组留出题、贪心解码及 384 新 token 上限；达到上限的回答会注明。"
    images = [(str(p), p.stem) for p in sorted(reports.glob("curves-step-*.png"))[-6:]]
    comparisons = sorted(reports.glob("comparison-step-*.json"))
    body = []
    for path in comparisons:
        body.append(f"<h3>{html.escape(path.stem)}</h3>")
        for row in json.loads(path.read_text()):
            body.append(f"<details><summary>{html.escape(row['case_id'] + ': ' + row['question'])}</summary>")
            for label, key in [("未微调基模", "base_answer"), ("当前 LoRA", "answer"), ("数据集参考答案（需领域核验）", "reference")]:
                body.append(f"<h4>{label}</h4><pre style='white-space:pre-wrap'>{html.escape(row[key])}</pre>")
            body.append(f"<p>基模达到长度上限：{row['base_hit_token_limit']}；当前 LoRA 达到长度上限：{row['hit_token_limit']}</p></details>")
    if not comparisons and (reports / "baseline.json").exists():
        body.append("<p>基模回答已保存，等待 step 200 的首轮对比。</p>")
    return text, images, "".join(body)


demo = create_ui()
train_fn = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "run_train")
runner = train_fn.fn.__self__
for elem in runner.manager.get_elem_list():
    key = runner.manager.get_id_by_elem(elem)
    if key not in preset:
        continue
    elem.value = preset[key]
    if isinstance(elem, gr.Dropdown):
        values = preset[key] if isinstance(preset[key], list) else [preset[key]]
        existing = [v for _, v in elem.choices]
        for value in values:
            if value not in existing:
                elem.choices.append((str(value), value))

payload = [preset.get(runner.manager.get_id_by_elem(elem), elem.value) for elem in train_fn.inputs]
reviewed = runner._parse_train_args(dict(zip(train_fn.inputs, payload)))
assert reviewed["max_steps"] == 800 and reviewed["fp16"] and not reviewed["bf16"]
assert reviewed["eval_dataset"] == "material_validation" and reviewed["report_to"] == "wandb"
assert reviewed["gradient_checkpointing"] and reviewed["gradient_accumulation_steps"] == 8
assert reviewed["load_best_model_at_end"] and reviewed["eval_steps"] == reviewed["save_steps"] == 200
logs = Path("/root/autodl-tmp/logs")
logs.mkdir(parents=True, exist_ok=True)
(logs / "material_webui_api_inputs.json").write_text(json.dumps(payload, ensure_ascii=False))
(logs / "material_webui_resolved.yaml").write_text(yaml.safe_dump(reviewed, sort_keys=False, allow_unicode=True))

with demo:
    with gr.Tab("诊断图与基模对比"):
        gr.Markdown("## 材料 QA · 800 step 诊断（预计约 3–4 小时）\n"
                    "所有训练参数已自动设置；每 25 step 留存曲线，每 200 step 保存同题回答对比。\n\n"
                    "训练 9,348 条、固定验证 96 条。界面验证比例为 0，因为已经使用独立验证文件；"
                    "总步数由高级参数 max_steps=800 控制。")
        refresh = gr.Button("刷新监测结果")
        status = gr.Markdown()
        gallery = gr.Gallery(label="中途训练曲线", columns=2, height=650)
        comparison = gr.HTML()
        refresh.click(read_reports, outputs=[status, gallery, comparison], concurrency_limit=None)
        gr.Timer(30).tick(read_reports, outputs=[status, gallery, comparison], concurrency_limit=None)
demo.config = demo.get_config_file()
demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False,
                    inbrowser=False, allowed_paths=[str(output_dir)])
