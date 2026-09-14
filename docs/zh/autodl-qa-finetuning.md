# AutoDL 上进行材料 QA LoRA 微调

本配置采用 `Qwen/Qwen3-8B`，在 AutoDL 上用 LoRA 训练 `test/wpFuh_5cOk4Y/alpaca.json`。Qwen3-8B 是 8.2B 参数文本模型，支持思考/非思考切换、工具调用和多语言指令跟随；材料领域知识由本项目 QA 数据注入。

## 数据与配置

- 数据集：由 1526 份材料资料生成的 9444 条 Alpaca 格式 QA。
- 数据注册：`test/wpFuh_5cOk4Y/dataset_info.json`。
- 训练配置：`examples/train_lora/qwen3_8b_material_qa_autodl.yaml`。
- 上下文长度：默认 2048 token，优先保证 24 GB GPU 可运行。
- 输出目录：`saves/qwen3-8b/lora/material-qa`。
- 监督格式：`template: qwen3` 与 `enable_thinking: false`，因为当前 QA 没有显式思维链。

## AutoDL 环境

建议使用 Python 3.11、PyTorch 2.4.1、CUDA 12.1、Transformers 4.56.2。安装当前仓库并验证 GPU：

```bash
cd /root/Fine-Tuning4Material
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

ModelScope 与 Hugging Face 上的模型 ID 均为 `Qwen/Qwen3-8B`。国内网络启用 ModelScope：

```bash
export USE_MODELSCOPE_HUB=1
```

## 启动训练

```bash
cd /root/Fine-Tuning4Material
export PYTHONPATH=/root/Fine-Tuning4Material/src
export USE_MODELSCOPE_HUB=1

CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train \
  examples/train_lora/qwen3_8b_material_qa_autodl.yaml
```

实时查看显存：

```bash
watch -n 1 nvidia-smi
```

训练中断后，把 YAML 中的 `resume_from_checkpoint` 改成最近的 `checkpoint-*` 目录再重新运行。

## GPU 与上下文调整

- RTX 3090/4090 24 GB：保持 `cutoff_len: 2048`；若仍然显存不足，再降到 1024。
- A100/A6000 40--48 GB：可将 `cutoff_len` 提升到 4096。
- A100/H100 80 GB：可测试 4096--8192，并按数据长度评估收益。
- V100 或其他不支持 BF16 的 GPU：设置 `bf16: false`、`fp16: true`。

## 垂类基模与通用 API 双路线

本地 Qwen3-8B 材料模型负责私有材料问答、机理分析、实验方案草拟、专业信息抽取和受控工具调用。通用 API 模型负责复杂任务规划、跨领域推理、Agent 编排和最终质量复核。

推荐调用链：`用户请求 -> 路由器 -> 本地材料模型/工具 -> 通用 API 复核或补强 -> 最终答案`。普通材料问答优先走本地模型；复杂规划、工具链失败、高风险结论或低置信度回答再升级到通用 API。发送到通用 API 的内容应先脱敏并结构化。

当前 9444 条 QA 不包含标准函数调用轨迹，因此本轮只做材料知识 SFT，不额外伪造 Agent 数据。Qwen3 原有工具调用能力由低学习率 LoRA 尽量保留；后续若有真实的 `工具定义 -> 调用 -> 观察 -> 回答` 轨迹，再进行第二阶段 Agent SFT。

## 参考

- [Qwen3-8B 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen-Agent 配置文档](https://qwenlm.github.io/Qwen-Agent/en/guide/get_started/configuration/)
