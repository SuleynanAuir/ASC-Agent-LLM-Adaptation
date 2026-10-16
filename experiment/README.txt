Qwen3-8B 项目材料领域：微调前后对比实验
实测日期：2026-10-14

打开方式
1. index.html：所有实验图与报告入口。
2. report/material_finetuning_review_7slides.pptx：7页可编辑会议汇报（图表为PNG）。
3. report/material_finetuning_review_7pages.pdf：相同7页的固定版式阅读版。
4. report/answer_comparisons.html：24条材料QA + 16条新编诊断，共40组完整成对输出。
5. figures/：15张PNG，其中10、11、13是投影大字号版本，不是新的独立实验。
6. results/paired_comparison_examples_8.json：8组材料问答，同一问题的微调前后实际回答、参考、逐题分数与差值。
7. results/paired_comparisons.json：全部40组同题对比（24条材料QA + 16条指令诊断），带评分规则、原始来源SHA256。
   before_finetuning / after_finetuning 分别是微调前后；answer保留实测原文。
   _100分数为0–100，null表示未评分，不是0分；delta_points为微调后减微调前。
   材料问答评分是参考文本重合度，不是事实准确率；指令诊断为严格JSON/期望值匹配。
   8例按每个主题case_id升序取前2条非空参考，不按提升幅度挑选；8例汇总只计这8例。
   本次JSON整理复用已有真实推理结果，没有再次占用GPU或改写模型答案。
8. figure_sum/index.html：已测指标学术组合图，4张分组图＋1张总览图；提供300 DPI PNG、SVG和中文图注。
   figure_sum/source_data.json保留绘图数值与区间，figure_manifest.json记录源文件SHA256及统计方法。
   图中未混入未测benchmark，也未把文本重合度标成事实准确率；原figures/和results/保持不变。

比较对象
基模：/root/autodl-tmp/models/Qwen3-8B。
微调模型：material-diagnostic-3h4h-20261014/checkpoint-600。
训练800步，约3小时18分钟；按原96条验证集选中600步。
输出根目录的adapter与checkpoint-600 SHA256相同，核验见configs/checkpoint_verification.json。
本轮只做推理和分析，没有修改原始数据、重新训练、修改模型权重或推送Git。

主要结果（均为实际测量）
- 完整保留问题、只截断参考答案，排除1条空参考后n=95：
  平均response-token NLL 1.509612 -> 1.074091，相对下降28.8499%。
  95/95条NLL降低；配对均值差0.435521，95% bootstrap区间[0.409746, 0.467429]。
  token加权PPL 4.4860 -> 2.9508。
- 原训练截断口径：n=96时1.705532 -> 1.185065，下降30.5164%；
  排除空参考后n=95时1.516900 -> 1.129030，下降25.5699%。
  两种截断保留的答案tokens不同，不直接用两口径loss高低对其排名。
- 开放式材料QA生成24条；23条非空参考计入重合分数。
  字符ROUGE-L F1 x100：19.923 -> 22.427；字符二元组F1：23.582 -> 26.495。
  512-token上限触及率：24/24 -> 19/24。
  这些只是文本重合指标，不是事实准确率，参考答案未经专家复核。
- 16条新编严格JSON/指令诊断：13/16 -> 13/16。
  纯JSON解析：16/16 -> 15/16；工具路由2/4 -> 2/4。
  未真实调用工具，不代表完整Agent能力，也不能证明通用能力无遗忘。
- V100S、FP16、SDPA、batch=4、greedy：QA聚合输出吞吐79.75 -> 44.75 tokens/s。
  中位每批耗时25.65 -> 42.90秒。LoRA未合并；预热排除；顺序测量，非服务压测。

关键数据与训练问题
- 9444条源数据存在43条空答案，其中训练集42条、验证集1条。
- 原96条验证中的空答案qa-04126仅计分2tokens，放大了逐样本等权NLL降幅。
- 原训练同款预处理：80/96条prompt token序列发生截断。
  解码后仅17/96条保留完整问题；79/96条参考答案也被截断。
  长system与source/target比例截断叠加，需要下一轮先修复，不能只增加训练轮次。
- 第17步grad_norm为NaN，第18步恢复有限；800条训练loss均有限。
  不能称“全程无异常”，也不保证后续训练不会溢出。
- 96条验证集参与过检查点选择，非独立测试；题目精确去重，不是文档级隔离。
  所有置信区间均条件于已选checkpoint，未校正选模偏差或文档相关性。

目录与可追溯性
configs/                         留出集、划分清单、checkpoint核验
results/training/                原训练参数、状态、loss/gradient、历史6题回答
results/comparison/              原预处理：192条NLL、80条生成、耗时、环境与协议
results/comparison/prompt_coverage.json  同款分词/截断的逐题覆盖审计
results/full_prompt_nll/         完整prompt补充实验：192条NLL，未再生成回答
results/summary.json             所有指标和限制的机器可读汇总
results/scored_generation.json   逐题严格评分，含原文、参考、期望、截断标记
results/paired_comparisons.json   全部40组成对JSON，含完整已生成回答与逐题量化
results/paired_comparison_examples_8.json  8组材料问答阅读版，4个主题各2组
figures/                        PNG图，点击index.html可放大
scripts/                        GPU评测、覆盖审计、分析、单元测试

图表索引
01 训练loss与裁剪前梯度；02 检查点验证loss；03 非空参考配对NLL；
04 原截断95条领域分组；05 开放QA重合度及截断；06 指令/Agent相关诊断；
07 推理耗时/吞吐；08 参考答案长度与截断；09 历史6题检查点轨迹；
10 训练概览投影版；11 指令诊断投影版；12 空参考敏感性；
13 完整问题口径领域分组投影版；14 完整prompt敏感性；15 问题覆盖审计。

复算（只重画图，不占GPU，不重新训练）
在项目根目录运行：
  python experiment/scripts/analyze_comparison.py
  python experiment/scripts/export_paired_comparisons.py
  python experiment/scripts/plot_figure_summary.py
  PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s experiment/scripts -p 'test_*.py' -v
需要numpy、matplotlib及中文字体。Linux缺少中文字体时可安装Noto CJK；
本次交付的PNG已渲染，可直接使用，不需要重新安装PyTorch。
分析脚本会更新派生JSON、PNG和HTML，不会覆盖原始NLL/回答或模型。
仅导出成对JSON时只需Python标准库，运行export_paired_comparisons.py即可，不用先重新分析或推理。

AutoDL复测命令（已实际完成，无需再跑；新运行使用新目录，防止覆盖本轮）
  cd /root/Fine-Tuning4Material
  export PYTHONPATH=/root/Fine-Tuning4Material/src
  export WANDB_MODE=disabled
  EXPERIMENT_RERUN_TAG=$(date +%Y%m%d-%H%M%S)
  /root/miniconda3/envs/llama/bin/python experiment/scripts/run_comparison.py \
    --output "experiment/results/rerun-$EXPERIMENT_RERUN_TAG"
  /root/miniconda3/envs/llama/bin/python experiment/scripts/run_comparison.py \
    --nll-only --preserve-prompt \
    --output "experiment/results/full-prompt-rerun-$EXPERIMENT_RERUN_TAG"
两次GPU推理合计约10分钟（本机实测量级，不是时限保证）。
本轮原始目录禁止重复覆盖，脚本检测completed.json后会退出。
重新分析原预处理复测可用 --comparison results/rerun-时间戳；
当前完整prompt敏感性图默认仍读取results/full_prompt_nll，请勿把两次运行混作一次结果。

本次与AutoDL同步
已经同步新增experiment目录；未使用git pull/reset/restore。
若今后需要从本机再次同步同名实验文件（会覆盖目标同名文件，不删除其他文件）：
  cd /Users/suleynan_suir/Desktop/Research/ZGC/Fine-Tuning4Material
  scp -P 39291 -r experiment root@region-9.autodl.pro:/root/Fine-Tuning4Material/
密码交互输入，不要写入脚本或Git。

训练W&B记录
https://wandb.ai/suleynanaiur-taiwan/Fine-Tuning4Material/runs/khahvcye
本轮额外评测保存在experiment，没有把它伪装成新的训练结果写回原W&B run。

下一步建议（未自动执行）
先验证完整问题保留、清洗空答案并复核专业术语；再按文档隔离独立测试集。
补做专家盲评、完整长回答与真实工具调用验收；LoRA合并后再测部署吞吐。
当前证据支持作为垂类候选继续研究，不足以承诺领域事实正确率或自主Agent上线。
