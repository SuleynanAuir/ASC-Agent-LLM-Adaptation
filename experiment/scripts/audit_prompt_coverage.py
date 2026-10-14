"""Read-only verification that the user's complete question survives SFT truncation."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.hparams import DataArguments

root=Path(__file__).resolve().parents[1]
tokenizer=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen3-8B')
args=DataArguments(template='qwen3',enable_thinking=False,cutoff_len=1536)
template=get_template_and_fix_tokenizer(tokenizer,args)
processor=SupervisedDatasetProcessor(template=template,tokenizer=tokenizer,processor=None,data_args=args)
data_dir=root.parent/'data/material_diagnostic_20261014'
data=json.loads((data_dir/'validation.json').read_text())
split=json.loads((data_dir/'split_manifest.json').read_text())
rows=[]
for index,row in zip(split['validation_source_indices'],data):
    q=row['instruction']+('\n'+row['input'] if row.get('input') else '')
    messages=[{'role':'user','content':q},{'role':'assistant','content':row['output']}]
    full_source,_=template.encode_oneturn(tokenizer,messages,row.get('system'))
    ids,labels=processor._encode_data_example(messages[:1],messages[1:],row.get('system'),None,[],[],[])
    first_target=next(i for i,label in enumerate(labels) if label!=-100)
    kept_source=tokenizer.decode(ids[:first_target],skip_special_tokens=False)
    rows.append({'case_id':f'qa-{index:05d}','source_tokens_full':len(full_source),
        'source_tokens_retained':first_target,'full_question_retained':q in kept_source,
        'user_role_header_retained':'<|im_start|>user' in kept_source})
result={'cutoff_len':1536,'n':len(rows),'full_question_retained_count':sum(r['full_question_retained'] for r in rows),
        'source_truncated_count':sum(r['source_tokens_retained']<r['source_tokens_full'] for r in rows),
        'method':'Decode the actual retained prompt produced by the same LLaMA Factory supervised processor; exact complete question string membership.',
        'note':'Coverage audit of existing training/evaluation preprocessing, not a new model run; no data or model modified.',
        'rows':rows}
dest=root/'results/comparison/prompt_coverage.json'
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print({k:v for k,v in result.items() if k!='rows'})
