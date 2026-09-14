# AutoDL 上进行 ASC QA 数据 LoRA 微调

本配置采用科学垂类基模 `internlm/Intern-S1-mini`，在 AutoDL 上用 LoRA 训练 `test/wpFuh_5cOk4Y/alpaca.json`。建议使用单张 A100 40 GB 或更高规格 GPU。

Intern-S1-mini 由约 8B 的 Qwen3 语言骨干和 0.3B 视觉编码器组成，继续预训练数据中包含超过 2.5T 科学领域 token。模型支持科学推理、图文输入、思考模式切换和工具调用，适合作为本项目的材料领域本地模型。

## 数据与配置

- 数据集：9444 条 Alpaca 格式 QA 数据。
- 数据注册：`test/wpFuh_5cOk4Y/dataset_info.json`。
- 训练配置：`examples/train_lora/intern_s1_mini_qa_autodl.yaml`。
- 上下文长度：4096 token，可完整覆盖约 99.9% 的样本；超长样本由 LLaMA Factory 截断。
- 输出目录：`saves/intern-s1-mini/lora/asc-qa`。

## AutoDL 环境

建议选择 Python 3.11、PyTorch 2.4 或更高版本、CUDA 12.1 或更高版本的镜像。模型要求 `transformers>=4.55.2`；本项目的依赖范围已覆盖该版本。

```bash
cd /root/autodl-tmp/Fine-Tuning4Material

python -m pip install --upgrade pip
pip install -e .

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

国内网络可让 LLaMA Factory 通过 ModelScope 下载模型：

```bash
export USE_MODELSCOPE_HUB=1
```

如果已经手动下载模型，请把 YAML 中的 `model_name_or_path` 改为模型的绝对路径。

## 启动训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  examples/train_lora/intern_s1_mini_qa_autodl.yaml
```

实时查看显存：

```bash
watch -n 1 nvidia-smi
```

训练中断后，把 YAML 中的 `resume_from_checkpoint` 改成最近的 `checkpoint-*` 目录再重新运行。

## GPU 与显存调整

- A100 40 GB、A6000 48 GB、A100/H100 80 GB：保持当前 `cutoff_len: 4096`。
- RTX 3090/4090 24 GB：先把 `cutoff_len` 降为 2048；Intern-S1 官方 LoRA 示例给出的单卡最低显存约为 22 GB。
- V100 或其他不支持 BF16 的 GPU：设置 `bf16: false`、`fp16: true`。
- 显存仍不足时不要开启视觉模块训练；当前配置已经冻结视觉塔和多模态投影器。

## 垂类基模与通用 API 双路线

本地垂类模型负责：

- 私有材料数据问答、ASC 机理分析、实验方案草拟和专业信息抽取。
- 调用材料数据库、文献检索、计算和知识图谱等受控工具。
- 处理不适合发送到外部 API 的内部数据。

通用 API 模型负责：

- 多步骤任务规划、复杂 Agent 编排、跨领域问题和最终质量复核。
- 在本地模型低置信度、证据冲突或需要更强通用推理时接管。
- 接收本地模型整理后的脱敏事实与结构化结果，不直接接收敏感原始材料。

推荐调用链：`用户请求 -> 路由器 -> 本地材料模型/工具 -> 通用 API 复核或补强 -> 最终答案`。普通材料问答优先走本地模型；复杂规划、工具链失败和高风险结论再升级到通用 API。

## 参考

- [Intern-S1-mini 官方模型卡](https://huggingface.co/internlm/Intern-S1-mini)
- [Intern-S1 官方 LLaMA Factory 微调说明](https://github.com/InternLM/Intern-S1/blob/main/docs/sft.md)
