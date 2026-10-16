微调前后已测指标：学术组合图

所有图片源于2026-10-14的真实评测；本次只绘图，无新训练/推理，无外部上传。
PNG为300 DPI；SVG保留矢量元素及可编辑文字。图内英文，以下提供中文图注。
颜色固定：蓝色Base、橙色LoRA checkpoint-600；图03逐题矩阵的深色/浅色表示通过/失败。

00_overview.png / .svg
核心结果总览：6个子图汇总NLL、PPL、文本重合、指令/格式通过率及吞吐/批次耗时。不同量纲分别作图，不合成为总分，也不使用雷达图混合尺度。

01_reference_likelihood.png / .svg
参考答案拟合与领域分组：95条非空参考，保留完整prompt；分别展示逐样本等权NLL、token加权PPL、同题散点及4个主题分组。它们衡量参考答案似然，不是生成答案事实正确率。

02_answer_overlap.png / .svg
材料回答文本重合度：23组非空参考问答的字符ROUGE-L与二元组F1，以及主题分组和逐题差值。新增主题二元组分组由已有逐题分数聚合，未重新进行GPU推理。负向变化不隐藏。

03_instruction_format.png / .svg
指令遵循与JSON格式：16组自编题：总体通过率、4类子任务及逐题成对结果。柱上标成功数/总数，误差线为Wilson区间。数字字符串或Markdown围栏可能导致严格评分失败，不能直接解释为计算错误。

04_generation_efficiency.png / .svg
生成效率与长度限制：同一V100S环境下的输出吞吐、6批实际耗时、24题输出长度及512-token触顶率。LoRA未合并；批次耗时不是单请求延迟，未测TTFT或P95请求延迟。

统计口径与限制
1. 连续指标为10000次按问题配对重采样的95% percentile bootstrap区间，seed=20261014。
2. PPL为exp(token加权NLL)，不等于exp(逐样本等权平均NLL)。
3. 指令通过率区间为95% Wilson区间，不校正自编题便利抽样偏差，不代表总体性能。
4. 所有区间条件于已选checkpoint，未校正选模偏差、文档相关性或多重比较；没有多训练种子复现实验。
5. 1条空参考不计入NLL/重合分数，但保留在24题长度和耗时统计中。
6. QA输出截断影响重合分数；参考答案未经专家盲评，不能把ROUGE/F1当事实准确率。
7. 性能为单次顺序评测，吞吐按总输出token/总生成时间；未测单请求P95、TTFT或峰值显存。
8. 未测试的标准benchmark没有绘制，也没有用官方基模分数代替本项目评测。

复现（项目根目录，环境需numpy和matplotlib）
python experiment/scripts/plot_figure_summary.py
python -m unittest discover -s experiment/scripts -p test_figure_summary.py -v

source_data.json：所有绘图数值及区间；figure_manifest.json：原始文件SHA256、方法和软件版本。
原始experiment/results及原图experiment/figures不作修改。
