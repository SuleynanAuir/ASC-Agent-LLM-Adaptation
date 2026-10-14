"""Rebuild quantitative PNGs and an inspectable answer gallery from raw results."""
import html
import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/material-experiment-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = 'results/comparison'
FIG = ROOT / 'figures'
RESULT = ROOT / 'results'
COLORS = {'base':'#77818C','tuned':'#D15B2C','ink':'#1D252D','blue':'#246A91'}
available_fonts={f.name for f in font_manager.fontManager.ttflist}
preferred_fonts=[f for f in ['Arial Unicode MS','Noto Sans CJK SC','Noto Serif CJK SC','WenQuanYi Zen Hei','DejaVu Sans'] if f in available_fonts]
plt.rcParams.update({'font.family':preferred_fonts, 'font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B6BCC2',
    'axes.labelcolor':COLORS['ink'],'text.color':COLORS['ink'],'figure.facecolor':'white',
    'axes.titleweight':'bold','axes.titlesize':15,'savefig.dpi':190})


def read(path):
    if path.startswith('results/comparison/'):
        path = path.replace('results/comparison', COMPARISON, 1)
    return json.loads((ROOT/path).read_text())


def write(path, value):
    (ROOT/path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def save(fig, name, note=''):
    if note:
        fig.text(.02,.015,note,fontsize=10,color='#57616C')
        fig.tight_layout(rect=(0,.055,1,.98))
    else:
        fig.tight_layout()
    fig.savefig(FIG/name)
    plt.close(fig)


def normalize(s):
    return ''.join(c for c in s.casefold() if c.isalnum())


def lcs_length(a,b):
    # Exact bit-parallel longest common subsequence on normalized characters.
    masks={}
    for i,c in enumerate(b): masks[c]=masks.get(c,0)|(1<<i)
    state=0
    for c in a:
        x=state|masks.get(c,0)
        state=x & ~(x-((state<<1)|1))
    return state.bit_count()


def overlaps(answer,reference):
    a,b=normalize(answer),normalize(reference)
    length=lcs_length(a,b)
    rouge=2*length/(len(a)+len(b)) if a and b else 0.
    ac=Counter(a[i:i+2] for i in range(len(a)-1))
    bc=Counter(b[i:i+2] for i in range(len(b)-1))
    bigram=2*sum((ac&bc).values())/(sum(ac.values())+sum(bc.values())) if ac and bc else 0.
    return {'char_rouge_l_f1':rouge,'char_bigram_f1':bigram,'answer_chars':len(a),'reference_chars':len(b)}


def strict_equal(got,expected):
    if isinstance(expected,dict):
        return isinstance(got,dict) and got.keys()==expected.keys() and all(strict_equal(got[k],v) for k,v in expected.items())
    if isinstance(expected,list):
        return isinstance(got,list) and len(got)==len(expected) and all(strict_equal(a,b) for a,b in zip(got,expected))
    if isinstance(expected,(int,float)) and not isinstance(expected,bool):
        return isinstance(got,(int,float)) and not isinstance(got,bool) and math.isclose(got,expected,abs_tol=1e-8)
    return type(got) is type(expected) and got==expected


def bootstrap(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261014)
    estimates=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(estimates,[.025,.975])]


def mean(xs): return float(np.mean(list(xs)))


def nll_stats(base,tuned,ids):
    b=np.array([base[k]['nll'] for k in ids]);t=np.array([tuned[k]['nll'] for k in ids])
    w=np.array([base[k]['scored_tokens'] for k in ids])
    return {'n':len(ids),'base_mean':float(b.mean()),'tuned_mean':float(t.mean()),
        'relative_reduction_pct':float(100*(1-t.mean()/b.mean())),
        'mean_paired_reduction':float((b-t).mean()),'paired_reduction_ci95':bootstrap(b-t),
        'improved_count':int((b>t).sum()),'base_token_weighted_ppl':float(np.exp(np.average(b,weights=w))),
        'tuned_token_weighted_ppl':float(np.exp(np.average(t,weights=w)))}


def main():
    global COMPARISON
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', default='results/comparison', help='Raw results directory, relative to experiment/')
    args = parser.parse_args()
    COMPARISON = args.comparison
    FIG.mkdir(parents=True,exist_ok=True)
    assert lcs_length('abc','ac')==2 and lcs_length('abc','xyz')==0
    assert strict_equal({'a':1},{'a':1}) and not strict_equal({'a':True},{'a':1})
    metrics=read('results/training/diagnostics/metrics.json')
    training=[r for r in metrics if 'loss' in r]
    checkpoints=read('results/training/diagnostics/summary.json')['eval_history']
    finite_grad=[r for r in training if isinstance(r.get('grad_norm'),(int,float)) and math.isfinite(r['grad_norm'])]
    bad_grad=[r['step'] for r in training if r not in finite_grad]
    nll=read('results/comparison/nll.json')
    base={r['case_id']:r for r in nll if r['mode']=='base'}
    tuned={r['case_id']:r for r in nll if r['mode']=='tuned'}
    assert len(base)==len(tuned)==96 and base.keys()==tuned.keys()
    split=read('configs/split_manifest.json')
    validation=read('configs/validation.json')
    invalid_ids={f'qa-{i:05d}' for i,r in zip(split['validation_source_indices'],validation) if not r['output'].strip()}
    valid_ids=[k for k in base if k not in invalid_ids]
    b=np.array([r['nll'] for r in base.values()]); t=np.array([tuned[k]['nll'] for k in base])
    deltas=b-t
    generation=read('results/comparison/generation.json')
    assert len(generation)==80, 'Expected 24 QA + 16 probes, each evaluated twice'
    for r in generation:
        if r['kind']=='qa':
            r['empty_reference']=not r['reference'].strip()
            if not r['empty_reference']:r.update(overlaps(r['answer'],r['reference']))
        else:
            try:
                parsed=json.loads(r['answer'].strip())
                r['valid_json']=True
                r['exact_pass']=strict_equal(parsed,r['expected'])
            except (ValueError,TypeError):
                r['valid_json']=False
                r['exact_pass']=False
    write('results/scored_generation.json',generation)
    topic_stats=[]
    for group in sorted({r['group'] for r in base.values()}):
        ids=[k for k in valid_ids if base[k]['group']==group]
        db=[base[k]['nll']-tuned[k]['nll'] for k in ids]
        topic_stats.append({'group':group,'n':len(ids),'base':mean(base[k]['nll'] for k in ids),
            'tuned':mean(tuned[k]['nll'] for k in ids),'delta':mean(db),'delta_ci95':bootstrap(db)})
    qa_stats={};probe_stats={};timing={}
    for mode in ['base','tuned']:
        qa=[r for r in generation if r['kind']=='qa' and r['mode']==mode]
        pr=[r for r in generation if r['kind']=='probes' and r['mode']==mode]
        scored_qa=[r for r in qa if not r['empty_reference']]
        qa_stats[mode]={'n':len(qa),'lexical_n':len(scored_qa),
            **{key:mean(r[key] for r in scored_qa) for key in ['char_rouge_l_f1','char_bigram_f1']},
            **{key:mean(r[key] for r in qa) for key in ['new_tokens','hit_token_limit']}}
        probe_stats[mode]={'n':len(pr),'valid_json':sum(r['valid_json'] for r in pr),'exact_pass':sum(r['exact_pass'] for r in pr),
            'groups':{g:{'n':sum(r['group']==g for r in pr),'pass':sum(r['group']==g and r['exact_pass'] for r in pr)} for g in ['提取','计算','工具路由','证据边界']}}
        batch=[r for r in read('results/comparison/timing.json') if r['mode']==mode and r['kind']=='qa']
        timing[mode]={'batches':len(batch),'median_batch_seconds':float(np.median([r['seconds'] for r in batch])),
            'total_seconds':sum(r['seconds'] for r in batch),'aggregate_tokens_per_second':sum(r['new_tokens'] for r in batch)/sum(r['seconds'] for r in batch)}
    paired_qa=[]
    gbase={r['case_id']:r for r in generation if r['mode']=='base' and r['kind']=='qa'}
    gtuned={r['case_id']:r for r in generation if r['mode']=='tuned' and r['kind']=='qa'}
    for k in gbase:
        if gbase[k]['empty_reference']:continue
        paired_qa.append({'case_id':k,'rouge_delta':gtuned[k]['char_rouge_l_f1']-gbase[k]['char_rouge_l_f1']})
    weights=np.array([r['scored_tokens'] for r in base.values()])
    summary={'training':{'steps':len(training),'runtime_seconds':read('results/training/train_results.json')['train_runtime'],
            'nonfinite_gradient_steps':bad_grad,'finite_loss_count':sum(math.isfinite(r['loss']) for r in training),
            'gradient_median':float(np.median([r['grad_norm'] for r in finite_grad])),
            'gradient_p95':float(np.quantile([r['grad_norm'] for r in finite_grad],.95)),
            'first50_mean_loss':mean(r['loss'] for r in training[:50]),'last50_mean_loss':mean(r['loss'] for r in training[-50:]),
            'best_step':600,'eval_history':checkpoints},
        'nll':nll_stats(base,tuned,list(base)),
        'nll_valid_reference':nll_stats(base,tuned,valid_ids),
        'topics':topic_stats,'qa':qa_stats,'qa_paired_rouge_delta_ci95':bootstrap([r['rouge_delta'] for r in paired_qa]),
        'probes':probe_stats,'qa_timing':timing,
        'data_audit':{'validation_n':len(base),'empty_reference_ids':sorted(invalid_ids),
            'invalid_reference_exclusion':'Post-hoc exclusion by empty-reference rule only. Original data, selected checkpoint and raw 96-case metrics retained; no retraining.',
            'target_truncated_count':sum(r['target_truncated'] for r in base.values()),
            'median_full_target_tokens':float(np.median([r['target_tokens_full'] for r in base.values()])),
            'median_scored_tokens':float(np.median(weights)),
            'median_full_prompt_tokens':float(np.median([r['source_tokens_full'] for r in base.values()]))},
        'statistics':'95% paired percentile bootstrap, 10000 resamples, seed20261014. Conditional on this selected checkpoint; ignores document clustering and selection bias.',
        'limitations':read('results/comparison/manifest.json')['limitations']}
    source=ROOT.parent/'test/wpFuh_5cOk4Y/alpaca.json'
    if source.exists():
        source_rows=json.loads(source.read_text())
        holdout=set(split['validation_source_indices'])
        empties=[i for i,r in enumerate(source_rows) if not r['output'].strip()]
        summary['data_audit'].update(source_empty_references=len(empties),
            train_empty_references=sum(i not in holdout for i in empties),
            source_nonempty_inputs=sum(bool(r.get('input','').strip()) for r in source_rows))
    coverage_path=ROOT/'results/comparison/prompt_coverage.json'
    if coverage_path.exists():
        coverage=read('results/comparison/prompt_coverage.json')
        summary['prompt_coverage']={k:v for k,v in coverage.items() if k!='rows'}
    full_path=ROOT/'results/full_prompt_nll/nll.json'
    full_topics=[]
    if full_path.exists():
        full=read('results/full_prompt_nll/nll.json')
        fb={r['case_id']:r for r in full if r['mode']=='base'}
        ft={r['case_id']:r for r in full if r['mode']=='tuned'}
        assert len(fb)==len(ft)==96
        summary['nll_full_prompt']=nll_stats(fb,ft,valid_ids)
        for group in sorted({r['group'] for r in base.values()}):
            ids=[k for k in valid_ids if base[k]['group']==group]
            full_topics.append({'group':group,'n':len(ids),'base':mean(fb[k]['nll'] for k in ids),'tuned':mean(ft[k]['nll'] for k in ids)})
        summary['topics_full_prompt']=full_topics
    write('results/summary.json',summary)

    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    steps=[r['step'] for r in training]; losses=np.array([r['loss'] for r in training])
    axes[0].plot(steps,losses,color='#BFC7CD',lw=.6,label='逐步 loss')
    axes[0].plot(steps[19:],np.convolve(losses,np.ones(20)/20,mode='valid'),color=COLORS['tuned'],lw=2,label='20 步滑动平均')
    axes[0].set(title='训练 loss 持续下降；早期发生一次非有限梯度',ylabel='训练 loss');axes[0].legend(loc='upper right')
    axes[1].plot(steps,[r['grad_norm'] if r in finite_grad else np.nan for r in training],color=COLORS['blue'],lw=.8)
    for s in bad_grad:
        axes[1].axvline(s,color=COLORS['tuned'],ls='--',lw=1)
        axes[1].annotate(f'第 {s} 步：NaN',(s,.53),xytext=(90,.54),arrowprops={'arrowstyle':'->','color':COLORS['tuned']},color=COLORS['tuned'])
    axes[1].set(xlabel='优化器步数',ylabel='裁剪前梯度范数')
    for ax in axes:ax.grid(alpha=.16)
    save(fig,'01_training_stability.png','来源：原始训练日志 800 步；NaN 点不连接。第 18 步起梯度恢复有限，不能据此保证后续训练永不溢出。')

    fig,ax=plt.subplots(figsize=(9,5.6))
    ax.plot([r['step'] for r in checkpoints],[r['eval_loss'] for r in checkpoints],marker='o',color=COLORS['tuned'],lw=2)
    for r in checkpoints:ax.annotate(f"{r['eval_loss']:.4f}",(r['step'],r['eval_loss']),xytext=(0,12),textcoords='offset points',ha='center')
    ax.axvline(600,color=COLORS['ink'],ls='--',lw=1,label='最佳 checkpoint-600')
    ax.set(title='验证损失在 600 步附近进入平台期',xlabel='优化器步数',ylabel='验证集平均 response-token NLL',ylim=(1.12,1.80),xticks=[0,200,400,600,800])
    ax.grid(alpha=.16);ax.legend()
    save(fig,'02_checkpoint_validation.png','n=96；该验证集用于检查点选择，不是独立测试集。图中 800 步为最后检查点，不是回载最佳模型后的评估。')

    b=np.array([base[k]['nll'] for k in valid_ids]);t=np.array([tuned[k]['nll'] for k in valid_ids]);deltas=b-t
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    axes[0].scatter(b,t,s=26,alpha=.8,color=COLORS['tuned'])
    lim=[min(b.min(),t.min())-.08,max(b.max(),t.max())+.08]
    axes[0].plot(lim,lim,ls='--',color=COLORS['base']);axes[0].set(xlim=lim,ylim=lim,xlabel='原始模型 NLL',ylabel='微调模型 NLL',title=f'{int((deltas>0).sum())}/95 条有效参考样本 NLL 降低')
    axes[1].hist(deltas,bins=16,color=COLORS['tuned'],alpha=.85);axes[1].axvline(0,color=COLORS['ink'],lw=1)
    axes[1].set(xlabel='原始 NLL − 微调 NLL（大于 0 为改善）',ylabel='样本数',title=f'配对平均降低 {deltas.mean():.3f}')
    save(fig,'03_paired_nll.png','96 条验证样本中排除 1 条空参考，保留 95 条；同一基模、模板及截断。原始 96 条结果保留在 results/comparison/nll.json。')

    fig,ax=plt.subplots(figsize=(9,5.5))
    x=np.arange(len(topic_stats))
    for offset,mode,label in [(-.19,'base','原始模型'),(.19,'tuned','微调 checkpoint-600')]:
        bars=ax.bar(x+offset,[r[mode] for r in topic_stats],.36,color=COLORS[mode],label=label)
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=11)
    ax.set(xticks=x,xticklabels=[f"{r['group']}\nn={r['n']}" for r in topic_stats],ylabel='平均 response-token NLL（越低越好）',title='四类材料问题的参考答案拟合均改善',ylim=(0,max(b.mean(),t.mean())*1.4))
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    save(fig,'04_topic_nll.png','排除 1 条空参考，n=95；分组是固定关键词规则，非专家标注。NLL 降低不等于事实正确率提高。')

    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for mode,offset,label in [('base',-.19,'原始模型'),('tuned',.19,'微调模型')]:
        vals=[100*qa_stats[mode][k] for k in ['char_rouge_l_f1','char_bigram_f1']]
        bars=axes[0].bar(np.arange(2)+offset,vals,.36,label=label,color=COLORS[mode]);axes[0].bar_label(bars,fmt='%.1f',padding=3)
    axes[0].set(xticks=[0,1],xticklabels=['字符 ROUGE-L F1','字符二元组 F1'],ylabel='参考答案重合分数 ×100',title='开放式 QA：文本重合度',ylim=(0,100));axes[0].legend(frameon=False)
    bars=axes[1].bar(['原始模型','微调模型'],[100*qa_stats[m]['hit_token_limit'] for m in ['base','tuned']],color=[COLORS['base'],COLORS['tuned']]);axes[1].bar_label(bars,fmt='%.1f%%',padding=3)
    axes[1].set(ylabel='触及 512-token 上限的比例',title='输出截断影响解释',ylim=(0,115))
    save(fig,'05_generation_overlap.png','生成 24 条；重合度仅计 23 条非空参考，截断统计仍为 24 条。greedy / 512 tokens；重合度不是事实准确率。')

    fig,ax=plt.subplots(figsize=(9,5.5))
    groups=['提取','计算','工具路由','证据边界'];x=np.arange(4)
    for mode,offset,label in [('base',-.19,'原始模型'),('tuned',.19,'微调模型')]:
        counts=[probe_stats[mode]['groups'][g]['pass'] for g in groups]
        bars=ax.bar(x+offset,counts,.36,label=label,color=COLORS[mode]);ax.bar_label(bars,labels=[f'{c}/4' for c in counts],padding=4)
    ax.set(xticks=x,xticklabels=groups,yticks=[0,1,2,3,4],ylim=(0,4.9),ylabel='严格 JSON + 期望字段全部正确',title='16 条新编小样本指令诊断');ax.legend(frameon=False)
    save(fig,'06_instruction_agent_probes.png','每类 4 题；仅根据题内信息。工具路由只测选择/参数，不执行工具，不代表完整 Agent 能力。')

    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for ax,key,title,unit in zip(axes,['median_batch_seconds','aggregate_tokens_per_second'],['每批完成耗时中位数','批量输出吞吐'],['秒 / batch=4','输出 tokens / 秒']):
        bars=ax.bar(['原始模型','微调模型'],[timing[m][key] for m in ['base','tuned']],color=[COLORS['base'],COLORS['tuned']]);ax.bar_label(bars,fmt='%.1f',padding=5)
        ax.set(title=title,ylabel=unit);ax.set_ylim(0,max(timing[m][key] for m in ['base','tuned'])*1.3)
    save(fig,'07_inference_cost.png','V100S 32 GB / FP16 / SDPA / 24 QA / batch=4；预热排除，LoRA 未合并；顺序测量、非服务压力测试。')

    fig,axes=plt.subplots(1,2,figsize=(11,5))
    axes[0].hist([r['target_tokens_full'] for r in base.values()],bins=20,color=COLORS['base']);axes[0].axvline(np.median(weights),color=COLORS['tuned'],lw=2,label='实际计分 token 中位数')
    axes[0].set(title='验证参考答案长度分布',xlabel='完整参考答案 tokens',ylabel='样本数');axes[0].legend(fontsize=10)
    truncated=summary['data_audit']['target_truncated_count']
    bars=axes[1].bar(['参考答案被截断','参考答案未截断'],[truncated,96-truncated],color=[COLORS['tuned'],COLORS['base']]);axes[1].bar_label(bars,padding=4)
    axes[1].set(title='训练/验证 cutoff=1536 的覆盖限制',ylabel='验证样本数',ylim=(0,110))
    save(fig,'08_context_coverage.png','使用与训练一致的 LLaMA Factory 编码/截断算法；只对保留的 response tokens 计算 NLL。')

    trajectory=[]
    for step,rows in [(0,read('results/training/diagnostics/baseline.json'))]+[(s,read(f'results/training/diagnostics/comparison-step-{s:04d}.json')) for s in [200,400,600,800]]:
        trajectory.append({'step':step,'n':len(rows),'rouge_l':mean(overlaps(r['answer'],r['reference'])['char_rouge_l_f1'] for r in rows),'cap_count':sum(r['hit_token_limit'] for r in rows)})
    write('results/checkpoint_generation_summary.json',trajectory)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot([r['step'] for r in trajectory],[100*r['rouge_l'] for r in trajectory],marker='o',color=COLORS['tuned'])
    ax.set(title='训练期间 6 条固定问题的回答重合度',xlabel='检查点步数（0 = 原模型）',ylabel='字符 ROUGE-L F1 ×100',xticks=[0,200,400,600,800]);ax.grid(alpha=.2)
    save(fig,'09_checkpoint_answer_trajectory.png','历史固定 6 条验证问题，greedy / 上限 384 tokens。与本轮 24 条 / 512 tokens 不是同一实验，不直接横比。')

    fig,ax=plt.subplots(figsize=(9,5.5))
    for mode,offset,label in [('base',-.19,'原始模型'),('tuned',.19,'微调模型')]:
        bars=ax.bar(np.arange(2)+offset,[summary[k][mode+'_mean'] for k in ['nll','nll_valid_reference']],.36,color=COLORS[mode],label=label)
        ax.bar_label(bars,fmt='%.3f',padding=4)
    ax.set(xticks=[0,1],xticklabels=['原始验证口径 / n=96','非空参考口径 / n=95'],ylabel='逐样本平均 NLL',ylim=(0,2.1),title='1 条空参考放大了宏平均损失降幅');ax.legend(frameon=False)
    save(fig,'12_empty_reference_sensitivity.png','原始口径降低 30.5%；排除空参考后降低 25.6%。空参考仅计分 2 tokens；未重新训练或重新选择检查点。')

    with plt.rc_context({'font.size':17,'axes.titlesize':20,'axes.labelsize':17,'legend.fontsize':15}):
        fig,ax=plt.subplots(figsize=(9,5.5))
        ax.plot(steps[19:],np.convolve(losses,np.ones(20)/20,mode='valid'),color=COLORS['base'],lw=2,label='训练 loss / 20 步均值')
        ax.plot([r['step'] for r in checkpoints],[r['eval_loss'] for r in checkpoints],marker='o',color=COLORS['tuned'],lw=2,label='验证 loss / 96 条')
        ax.axvline(600,color=COLORS['ink'],ls=':',lw=1)
        ax.set(xlabel='优化器步数',ylabel='Response-token NLL',xticks=[0,200,400,600,800],ylim=(1.05,1.8))
        ax.legend(frameon=False);ax.grid(alpha=.15)
        save(fig,'10_training_overview.png')

        fig,ax=plt.subplots(figsize=(9,5.5))
        groups=['提取','计算','工具路由','证据边界'];x=np.arange(4)
        for mode,offset,label in [('base',-.19,'原始模型'),('tuned',.19,'微调模型')]:
            counts=[probe_stats[mode]['groups'][g]['pass'] for g in groups]
            bars=ax.bar(x+offset,counts,.36,label=label,color=COLORS[mode]);ax.bar_label(bars,labels=[f'{c}/4' for c in counts],padding=5,fontsize=17)
        ax.set(xticks=x,xticklabels=groups,yticks=[0,1,2,3,4],ylim=(0,5.2),ylabel='严格通过数');ax.legend(frameon=False)
        save(fig,'11_probe_slide.png')

        fig,ax=plt.subplots(figsize=(9,5.5))
        slide_topics=full_topics or topic_stats
        x=np.arange(len(slide_topics))
        for mode,offset,label in [('base',-.19,'原始模型'),('tuned',.19,'微调模型')]:
            bars=ax.bar(x+offset,[r[mode] for r in slide_topics],.36,label=label,color=COLORS[mode])
            ax.bar_label(bars,fmt='%.3f',padding=5,fontsize=16)
        ax.set(xticks=x,xticklabels=[f"{label}\nn={r['n']}" for label,r in zip(['工艺施工','氧化侵蚀','热震力学','结构表征'],slide_topics)],ylabel='平均 NLL（越低越好）',ylim=(0,2.15))
        ax.legend(frameon=False);ax.grid(axis='y',alpha=.12)
        save(fig,'13_topic_slide.png')

    if full_topics:
        fig,ax=plt.subplots(figsize=(10,5.5))
        for mode,offset,label in [('base',-.19,'原始模型'),('tuned',.19,'微调模型')]:
            bars=ax.bar(np.arange(2)+offset,[summary[k][mode+'_mean'] for k in ['nll_valid_reference','nll_full_prompt']],.36,color=COLORS[mode],label=label)
            ax.bar_label(bars,fmt='%.3f',padding=4)
        ax.set(xticks=[0,1],xticklabels=['原训练截断 / 95 条非空参考','完整问题优先 / 同样 95 条'],ylabel='逐样本平均 NLL',ylim=(0,2.05),title='补充评测：完整保留问题后仍有拟合改善');ax.legend(frameon=False)
        save(fig,'14_full_prompt_sensitivity.png','只在各口径内成对比较模型；两种截断保留的答案 token 不同，loss 不宜直接跨口径排名。未修改模型或重新训练。')
    if coverage_path.exists():
        fig,ax=plt.subplots(figsize=(9,5.5))
        retained=coverage['full_question_retained_count']
        bars=ax.bar(['完整问题保留','问题不完整'],[retained,coverage['n']-retained],color=[COLORS['blue'],COLORS['tuned']])
        ax.bar_label(bars,padding=4,fontsize=15)
        ax.set(ylabel='验证样本数',ylim=(0,100),title='现有预处理下，仅 17/96 条完整保留问题')
        save(fig,'15_question_coverage.png','将训练同款 processor 保留的 prompt tokens 解码，核查是否包含完整问题文本；长 system + source/target 比例截断造成风险。')

    esc=html.escape
    sections=[]
    for kind in ['qa','probes']:
        ids=[r['case_id'] for r in generation if r['kind']==kind and r['mode']=='base']
        for cid in ids:
            pair={r['mode']:r for r in generation if r['case_id']==cid}
            ref=pair['base'].get('reference') or json.dumps(pair['base']['expected'],ensure_ascii=False)
            cols=''.join(f'<div><h3>{label}</h3><p>tokens={pair[m]["new_tokens"]}; capped={pair[m]["hit_token_limit"]}</p><pre>{esc(pair[m]["answer"])}</pre></div>' for m,label in [('base','微调前'),('tuned','微调后 checkpoint-600')])
            sections.append(f'<section><h2>{cid} · {esc(pair["base"]["group"])}</h2><p>{esc(pair["base"]["question"])}</p><div class="pair">{cols}</div><details><summary>参考答案 / 规则期望（参考答案未专家复核）</summary><pre>{esc(ref)}</pre></details></section>')
    page='<!doctype html><meta charset="utf-8"><title>微调前后完整回答对比</title><style>body{font:16px/1.65 system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#202830}section{border-top:1px solid #bbb;padding:25px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:32px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:#f6f6f6;padding:20px}h3{color:#b64b22}</style><h1>Qwen3-8B 微调前后逐题对比</h1><p>24 条材料 QA + 16 条新编规则诊断；固定 checkpoint-600。材料 QA 使用完整原提示；上限512 tokens，可能截断。本文不把文本重合度当作事实准确率。</p>'+''.join(sections)
    (ROOT/'report/answer_comparisons.html').write_text(page)
    items=''.join(f'<section><h2>{esc(p.stem)}</h2><a href="figures/{p.name}"><img src="figures/{p.name}" alt="{esc(p.stem)}"></a></section>' for p in sorted(FIG.glob('*.png')))
    headline='原截断口径：95条非空参考NLL降低25.6%；原始96条降幅30.5%受空参考影响。'
    if full_topics:headline+=f"完整保留问题后，95条NLL降低{summary['nll_full_prompt']['relative_reduction_pct']:.1f}%。"
    overview='<!doctype html><meta charset="utf-8"><title>材料微调对比实验</title><style>body{font:17px/1.7 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#202830}img{width:100%;height:auto}section{margin:55px 0;border-top:1px solid #ddd}a{color:#b64b22}</style><h1>Qwen3-8B 材料领域：微调前后对比实验</h1><p>2026-10-14 实测。'+headline+'指令诊断13/16→13/16，未见净提升。现有预处理只有17/96条保留完整问题，需先修复截断。数据与脚本均保留在本目录。</p><p><a href="report/material_finetuning_review_7slides.pptx">7页会议汇报 PowerPoint</a> · <a href="report/material_finetuning_review_7pages.pdf">PDF</a> · <a href="report/answer_comparisons.html">40组逐题完整回答对比</a> · <a href="results/summary.json">完整量化汇总</a></p><p>01–09及12、14、15为分析图；10、11、13为会议投影用的大字号版本。全部图表可点击打开PNG原图。参考答案未专家复核，验证集参与选模，本轮不提供事实准确率结论。</p>'+items
    (ROOT/'index.html').write_text(overview)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
