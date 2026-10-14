"""Read-only, paired GPU evaluation of the base model and selected LoRA adapter.

Run from the repository with the same environment used for training.
No training, checkpoint mutation, external API, or reference-answer judge is used.
"""
import argparse
import hashlib
import importlib.metadata
import json
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def topic(question):
    if re.search('氧化|侵蚀|熔渣|抗渣|脱碳', question):
        return '氧化与侵蚀'
    if re.search('热震|强度|应力|蠕变|抗折|损耗|剥落|脱粘|裂纹', question):
        return '热震与力学'
    if re.search('浇注|施工|混|成型|烧成|干燥|流变|结合剂|陈化|软化', question):
        return '工艺与施工'
    return '结构与表征/其他'


def probes():
    """Authored evidence-grounded probes; exact expected JSON, no domain facts assumed."""
    rows = [
        ('提取', '记录：样品A，Al2O3含量85 wt%，烧成温度1500°C。输出且仅输出JSON，键为sample、alumina_wt_pct、temperature_c，数值用数字。', {'sample':'A','alumina_wt_pct':85,'temperature_c':1500}),
        ('提取', '记录：样品B，体积密度2.9 g/cm3，显气孔率18%。只输出JSON：sample、density_g_cm3、porosity_pct。', {'sample':'B','density_g_cm3':2.9,'porosity_pct':18}),
        ('提取', '记录：试样C经过5次循环后残余强度42 MPa。只输出JSON：sample、cycles、residual_strength_mpa。', {'sample':'C','cycles':5,'residual_strength_mpa':42}),
        ('提取', '记录：批次D混料20 min，干燥温度110°C，干燥24 h。只输出JSON：batch、mixing_min、drying_c、drying_h。', {'batch':'D','mixing_min':20,'drying_c':110,'drying_h':24}),
        ('计算', '原始强度80 MPa，处理后60 MPa。保持率=处理后/原始×100%。只输出JSON：retention_pct。', {'retention_pct':75}),
        ('计算', '配制200 kg干料，其中碳质量分数5%。按给定数字计算碳质量。只输出JSON：carbon_kg。', {'carbon_kg':10}),
        ('计算', '样品三次测量值分别为30、36、42 MPa。计算算术平均值，只输出JSON：mean_mpa。', {'mean_mpa':36}),
        ('计算', '侵蚀深度由10 mm降低到7 mm。降低率=(原值-现值)/原值×100%。只输出JSON：reduction_pct。', {'reduction_pct':30}),
        ('工具路由', '允许的工具：search_docs用于查文献，calculator用于数值计算，ask_user用于缺失输入。请求是计算(80-60)/80。不要实际调用工具，只输出JSON：tool、expression。', {'tool':'calculator','expression':'(80-60)/80'}),
        ('工具路由', '允许工具：search_docs查文献，calculator算数，ask_user补输入。用户要求查找关键词“镁碳砖 抗氧化”的文献。只输出JSON：tool、query；query必须逐字保留关键词。', {'tool':'search_docs','query':'镁碳砖 抗氧化'}),
        ('工具路由', '允许工具：search_docs查文献，calculator算数，ask_user补输入。用户要算保持率但没有原始强度。只输出JSON：tool、missing；missing写“原始强度”。', {'tool':'ask_user','missing':'原始强度'}),
        ('工具路由', '规则：计算选calculator；资料查询选search_docs；缺输入选ask_user。任务：先查文献，再计算均值。只输出JSON：tools，值为按顺序排列的两个工具名数组。', {'tools':['search_docs','calculator']}),
        ('证据边界', '仅有记录：样品A密度3.1 g/cm3。没有强度记录，不允许推测。只输出JSON：sample、strength_mpa；未知值必须为null。', {'sample':'A','strength_mpa':None}),
        ('证据边界', '仅有记录：样品B抗折强度45 MPa。没有测试温度，不允许补全。只输出JSON：sample、test_temperature_c；未知值为null。', {'sample':'B','test_temperature_c':None}),
        ('证据边界', '记录只说“强度提高”，没有百分比。只输出JSON：improvement_pct、has_numeric_evidence；未知数值为null，布尔值表示是否存在数值证据。', {'improvement_pct':None,'has_numeric_evidence':False}),
        ('证据边界', '资料A称密度2.8 g/cm3，资料B称3.2 g/cm3；未说明何者正确。禁止自行选取或平均。只输出JSON：density_g_cm3、conflict；未决数值为null。', {'density_g_cm3':None,'conflict':True}),
    ]
    return [{'case_id':f'probe-{i+1:02d}','group':group,'instruction':q,'input':'',
             'system':'严格依据用户给定信息作答。遵守输出格式，不添加解释、Markdown或未经提供的事实。',
             'expected':expected} for i,(group,q,expected) in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/root/autodl-tmp/models/Qwen3-8B')
    parser.add_argument('--adapter', default='/root/autodl-tmp/saves/qwen3-8b/lora/material-diagnostic-3h4h-20261014/checkpoint-600')
    parser.add_argument('--data', default='data/material_diagnostic_20261014')
    parser.add_argument('--output', default='experiment/results/comparison')
    parser.add_argument('--preserve-prompt', action='store_true', help='Sensitivity evaluation: preserve the entire prompt and truncate only the reference answer.')
    parser.add_argument('--nll-only', action='store_true', help='Skip generation; useful for paired preprocessing sensitivity tests.')
    args = parser.parse_args()
    out = Path(args.output)
    if (out/'completed.json').exists():
        raise SystemExit('Results already complete; choose a new --output to preserve them.')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments
    set_seed(20261014)
    torch.set_num_threads(4)
    data = json.loads((Path(args.data)/'validation.json').read_text())
    split = json.loads((Path(args.data)/'split_manifest.json').read_text())
    for i,row in enumerate(data):
        row['case_id'] = f"qa-{split['validation_source_indices'][i]:05d}"
        row['group'] = topic(row['instruction'])
    groups = sorted({r['group'] for r in data})
    rng = random.Random(20261014)
    selected = []
    for group in groups:
        pool = [r for r in data if r['group']==group]
        selected += rng.sample(pool, min(6,len(pool)))
    extra = probes()
    train = json.loads((Path(args.data)/'train.json').read_text())
    normalize = lambda s: re.sub(r'\s+','',s).casefold()
    train_questions = {normalize(r['instruction']+r.get('input','')) for r in train}
    assert not any(normalize(r['instruction']+r.get('input','')) in train_questions for r in data+extra)
    publish(out/'cases.json', {'generation':selected,'probes':extra})
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    data_args = DataArguments(template='qwen3', enable_thinking=False, cutoff_len=1536)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = SupervisedDatasetProcessor(template=template,tokenizer=tokenizer,processor=None,data_args=data_args)
    encoded = []
    for row in data:
        prompt = row['instruction'] + ('\n'+row['input'] if row.get('input') else '')
        messages = [{'role':'user','content':prompt},{'role':'assistant','content':row['output']}]
        source, target = template.encode_oneturn(tokenizer,messages,row.get('system'))
        ids, labels = processor._encode_data_example(messages[:1],messages[1:],row.get('system'),None,[],[],[])
        if args.preserve_prompt:
            remaining=1536-len(source)
            if remaining<1:
                raise ValueError(f'Prompt exceeds budget: {row["case_id"]}')
            ids=source+target[:remaining]
            labels=[-100]*len(source)+target[:remaining]
        encoded.append((ids,labels,{'source_tokens_full':len(source),'target_tokens_full':len(target),
            'sequence_tokens':len(ids),'scored_tokens':sum(t!=-100 for t in labels[1:]),
            'target_truncated':sum(t!=-100 for t in labels)<len(target)}))
    model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,
            attn_implementation='sdpa',device_map={'':0})
    model = PeftModel.from_pretrained(model,args.adapter,is_trainable=False).eval()
    manifest = {'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'model':args.model,'adapter':args.adapter,
        'adapter_sha256':hashlib.sha256((Path(args.adapter)/'adapter_model.safetensors').read_bytes()).hexdigest(),
        'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
        'dtype':'float16','attention':'sdpa','seed':20261014,'validation_count':len(data),
        'generation_count':len(selected),'probe_count':len(extra),'max_new_tokens':{'qa':512,'probes':256},
        'generation_batch_size':4,'decoding':'greedy, enable_thinking=false; adapter disabled vs enabled on one base',
        'nll':('Preserve full LLaMA Factory prompt, truncate response only, cutoff=1536' if args.preserve_prompt else 'Per-example mean of teacher-forced response-token NLL, LLaMA Factory supervised encoding, cutoff=1536'),
        'preserve_prompt':args.preserve_prompt,'nll_only':args.nll_only,
        'limitations':['Validation used for checkpoint selection, not an untouched test.',
          'Question-disjoint, not document-disjoint.', 'Reference answers are not independently expert verified.',
          'QA generation is token-capped; lexical overlap is not factual accuracy.',
          'Hand-authored small probes measure JSON/routing decisions, not actual agent tool execution.'],
        'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','llamafactory']}}
    publish(out/'manifest.json',manifest)
    nll_rows=[]
    generations=[]
    batches=[]
    start_all=time.time()
    for mode in ['base','tuned']:
        context = model.disable_adapter() if mode=='base' else nullcontext()
        with context,torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for i,(row,(ids,labels,meta)) in enumerate(zip(data,encoded)):
                inputs=torch.tensor([ids],device='cuda')
                targets=torch.tensor([labels],device='cuda')
                loss=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),labels=targets,use_cache=False).loss
                value=float(loss)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite evaluation loss: {mode}/{row["case_id"]}')
                nll_rows.append({'mode':mode,'case_id':row['case_id'],'group':row['group'],'nll':value,**meta})
                if (i+1)%16==0:
                    publish(out/'nll.json',nll_rows)
                    print(f'{mode}: NLL {i+1}/{len(data)}',flush=True)
            publish(out/'nll.json',nll_rows)
            if args.nll_only:
                continue
            for kind,cases,limit in [('qa',selected,512),('probes',extra,256)]:
                for offset in range(0,len(cases),4):
                    batch=cases[offset:offset+4]
                    prompts=[]
                    for row in batch:
                        prompt=row['instruction']+('\n'+row['input'] if row.get('input') else '')
                        messages=([{'role':'system','content':row['system']}] if row.get('system') else [])
                        messages.append({'role':'user','content':prompt})
                        prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
                    inputs=tokenizer(prompts,padding=True,return_tensors='pt').to('cuda')
                    if offset==0:
                        model.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=True)
                    torch.cuda.synchronize()
                    started=time.perf_counter()
                    output=model.generate(**inputs,max_new_tokens=limit,do_sample=False,use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    duration=time.perf_counter()-started
                    counts=[]
                    for row,sequence in zip(batch,output[:,inputs.input_ids.shape[1]:].tolist()):
                        if tokenizer.eos_token_id in sequence:
                            sequence=sequence[:sequence.index(tokenizer.eos_token_id)+1]
                        else:
                            while sequence and sequence[-1]==tokenizer.pad_token_id:
                                sequence.pop()
                        counts.append(len(sequence))
                        generations.append({'mode':mode,'kind':kind,'case_id':row['case_id'],'group':row['group'],
                            'question':row['instruction'],'reference':row.get('output'),'expected':row.get('expected'),
                            'answer':tokenizer.decode(sequence,skip_special_tokens=True),
                            'new_tokens':len(sequence),'hit_token_limit':len(sequence)>=limit,
                            'batch_seconds':duration,'batch_id':f'{mode}-{kind}-{offset//4:02d}'})
                    batches.append({'mode':mode,'kind':kind,'batch_id':f'{mode}-{kind}-{offset//4:02d}',
                        'seconds':duration,'new_tokens':sum(counts),'count':len(batch)})
                    publish(out/'generation.json',generations)
                    publish(out/'timing.json',batches)
                    print(f'{mode}: {kind} {offset+len(batch)}/{len(cases)}, {duration:.1f}s',flush=True)
    publish(out/'completed.json',{'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'seconds':time.time()-start_all,
        'nll_rows':len(nll_rows),'generation_rows':len(generations)})
    print('PAIRED EVALUATION COMPLETE',flush=True)


if __name__=='__main__':
    main()
