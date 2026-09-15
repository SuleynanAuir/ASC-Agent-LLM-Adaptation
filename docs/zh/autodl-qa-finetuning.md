# AutoDL 上进行材料 QA LoRA 微调

本配置采用 `Qwen/Qwen3-8B`，在 AutoDL 上用 LoRA 训练 `test/wpFuh_5cOk4Y/alpaca.json`。Qwen3-8B 是 8.2B 参数文本模型，支持思考/非思考切换、工具调用和多语言指令跟随；材料领域知识由本项目 QA 数据注入。

## 数据与配置

- 数据集：由 1526 份材料资料生成的 9444 条 Alpaca 格式 QA。
- 数据注册：`test/wpFuh_5cOk4Y/dataset_info.json`。
- 冒烟测试配置：`examples/train_lora/qwen3_8b_material_qa_smoke_autodl.yaml`，只取 64 条数据跑 5 步，用于快速验证环境和数据链路。
- 正式训练配置：`examples/train_lora/qwen3_8b_material_qa_autodl.yaml`，使用全部材料数据训练 1 个 epoch。
- 正式训练上下文长度：默认 1536 token，在速度、材料信息保留和 24 GB GPU 显存之间折中。
- 输出目录：`/root/autodl-tmp/saves/qwen3-8b/lora/material-qa`，放在 AutoDL 数据盘以降低实例释放或系统盘空间不足带来的风险。
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

## 先做 5 步冒烟测试

模型已下载到 `/root/autodl-tmp/models/Qwen3-8B` 后，先运行：

```bash
cd /root/Fine-Tuning4Material
export PYTHONPATH=/root/Fine-Tuning4Material/src
unset USE_MODELSCOPE_HUB

CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train \
  examples/train_lora/qwen3_8b_material_qa_smoke_autodl.yaml
```

该测试用于确认模型、数据、LoRA 和 CUDA 全链路正常，不用于产出正式模型。

## 启动正式训练

```bash
cd /root/Fine-Tuning4Material
export PYTHONPATH=/root/Fine-Tuning4Material/src
unset USE_MODELSCOPE_HUB

CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train \
  examples/train_lora/qwen3_8b_material_qa_autodl.yaml
```

## Weights & Biases 监控

正式配置使用 `report_to: wandb`，并将 `logging_steps` 设为 5，以记录训练损失、梯度范数和学习率，同时减少网络上报开销。首次训练前安装并登录：

```bash
python -m pip install -U wandb
wandb login
```

启动时设置项目名，同时关闭模型文件和参数直方图上传，减少网络与训练开销：

```bash
export WANDB_PROJECT=Fine-Tuning4Material
export WANDB_LOG_MODEL=false
export WANDB_WATCH=false
```

## LlamaBoard 一键预设

先把 `llamaboard_config/qwen3_8b_material_qa_user_config.yaml` 复制为 `llamaboard_cache/user_config.yaml`，再设置 `LLAMABOARD_PRESET=llamaboard_config/qwen3_8b_material_qa.yaml` 启动 WebUI。页面会自动选择已注册的 `Qwen3-8B-Thinking`，但模型路径映射到本地 `/root/autodl-tmp/models/Qwen3-8B`，不会重新下载；其余 UI 参数也会自动载入。W&B API Key 不写入预设，仍通过 `wandb login` 提供。

实时查看显存：

```bash
watch -n 1 nvidia-smi
```

训练中断后，把 YAML 中的 `resume_from_checkpoint` 改成最近的 `checkpoint-*` 目录再重新运行。

## 为什么原配置会显示 80 多小时

原配置有 8971 条训练样本、3 个 epoch，并设置 `gradient_accumulation_steps: 8`。进度条中的一个 step 实际包含 8 次前向与反向传播；首个 step 又常包含 CUDA 内核初始化开销，因此仅凭第一步推算的 ETA 偏大。

正式快速版把 9444 条 QA 全部用于训练，但把 3 个 epoch 改为 1 个、`cutoff_len` 从 2048 降到 1536，并关闭本轮验证。每 100 个优化 step 保存一次断点并保留最近 2 个，以兼顾恢复能力和保存开销。不要仅为了让进度条变快而降低梯度累积：那会增加优化 step 数，总微批次数基本不变。

## GPU 与上下文调整

- RTX 3090/4090 24 GB：先使用 `cutoff_len: 1536`；如显存充足且更重视回答完整性，可恢复到 2048。
- A100/A6000 40--48 GB：可将 `cutoff_len` 提升到 4096。
- A100/H100 80 GB：可测试 4096--8192，并按数据长度评估收益。
- Tesla V100/V100S（Volta）：使用 `bf16: false`、`fp16: true`，以利用其原生 FP16 Tensor Core；不要依据 PyTorch 的 BF16 软件可用性检测结果启用 BF16。

## 垂类基模与通用 API 双路线

本地 Qwen3-8B 材料模型负责私有材料问答、机理分析、实验方案草拟、专业信息抽取和受控工具调用。通用 API 模型负责复杂任务规划、跨领域推理、Agent 编排和最终质量复核。

推荐调用链：`用户请求 -> 路由器 -> 本地材料模型/工具 -> 通用 API 复核或补强 -> 最终答案`。普通材料问答优先走本地模型；复杂规划、工具链失败、高风险结论或低置信度回答再升级到通用 API。发送到通用 API 的内容应先脱敏并结构化。

当前 9444 条 QA 不包含标准函数调用轨迹，因此本轮只做材料知识 SFT，不额外伪造 Agent 数据。Qwen3 原有工具调用能力由低学习率 LoRA 尽量保留；后续若有真实的 `工具定义 -> 调用 -> 观察 -> 回答` 轨迹，再进行第二阶段 Agent SFT。

## 参考

- [Qwen3-8B 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen-Agent 配置文档](https://qwenlm.github.io/Qwen-Agent/en/guide/get_started/configuration/)
