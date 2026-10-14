"""Package measured before/after answers as self-contained paired JSON.

Uses the standard library only. Does not run inference, alter raw results,
judge factual correctness, or send data to an external service.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    'results/comparison/manifest.json',
    'results/comparison/cases.json',
    'results/comparison/generation.json',
    'results/scored_generation.json',
    'results/full_prompt_nll/nll.json',
    'results/summary.json',
    'configs/checkpoint_verification.json',
)
SIDES = {'base': 'before_finetuning', 'tuned': 'after_finetuning'}


def reject_constant(value):
    raise ValueError(f'Non-finite JSON value: {value}')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_unique(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, f'Duplicate record: {key}')
        result[key] = row
    return result


def percent(value):
    require(math.isfinite(value) and 0 <= value <= 1, 'Score must be finite in [0, 1]')
    return round(100 * value, 6)


def delta(before, after):
    return None if before is None or after is None else round(after - before, 6)


def aggregate(pairs):
    qa = [p for p in pairs if p['kind'] == 'material_qa']
    scored_qa = [p for p in qa if p['reference_answer'].strip()]
    probes = [p for p in pairs if p['kind'] == 'instruction_probe']
    result = {
        'paired_questions': len(pairs),
        'material_qa_count': len(qa),
        'material_qa_scored_count': len(scored_qa),
        'excluded_empty_reference_ids': [p['case_id'] for p in qa if not p['reference_answer'].strip()],
        'instruction_probe_count': len(probes),
        'material_qa': {},
        'instruction_probes': {},
    }
    for side in SIDES.values():
        result['material_qa'][side] = {
            metric: round(mean(p[side]['scores'][metric] for p in scored_qa), 6) if scored_qa else None
            for metric in ('char_rouge_l_f1_100', 'char_bigram_f1_100')
        }
        result['material_qa'][side]['hit_token_limit_count'] = sum(p[side]['hit_token_limit'] for p in qa)
        result['instruction_probes'][side] = {
            'strict_pass_count': sum(p[side]['scores']['strict_instruction_100'] == 100 for p in probes),
            'strict_pass_rate_pct': round(mean(p[side]['scores']['strict_instruction_100'] for p in probes), 6) if probes else None,
            'valid_json_count': sum(p[side]['scores']['valid_json_100'] == 100 for p in probes),
            'valid_json_rate_pct': round(mean(p[side]['scores']['valid_json_100'] for p in probes), 6) if probes else None,
        }
    result['material_qa']['delta_points_after_minus_before'] = {
        metric: delta(result['material_qa']['before_finetuning'][metric], result['material_qa']['after_finetuning'][metric])
        for metric in ('char_rouge_l_f1_100', 'char_bigram_f1_100')
    }
    return result


def build_document(root=ROOT):
    manifest = read(root / INPUTS[0])
    case_file = read(root / INPUTS[1])
    raw = index_unique(read(root / INPUTS[2]), ('case_id', 'mode'))
    scored = index_unique(read(root / INPUTS[3]), ('case_id', 'mode'))
    nll = index_unique(read(root / INPUTS[4]), ('case_id', 'mode'))
    summary = read(root / INPUTS[5])
    checkpoint = read(root / INPUTS[6])
    cases = case_file['generation'] + case_file['probes']
    index_unique(cases, ('case_id',))
    expected_keys = {(c['case_id'], mode) for c in cases for mode in SIDES}
    require(raw.keys() == scored.keys() == expected_keys, 'Missing or extra before/after answers')
    require(len(case_file['generation']) == manifest['generation_count'], 'QA count differs from manifest')
    require(len(case_file['probes']) == manifest['probe_count'], 'Probe count differs from manifest')
    require(checkpoint['equal'] and checkpoint['checkpoint_600_adapter_sha256'] == manifest['adapter_sha256'],
            'Checkpoint hash does not match the evaluated adapter')
    pairs = []
    for case in sorted(cases, key=lambda c: (not c['case_id'].startswith('qa-'), c['group'], c['case_id'])):
        case_id = case['case_id']
        is_qa = case_id.startswith('qa-')
        expected_kind = 'qa' if is_qa else 'probes'
        question = case['instruction'] + ('\n' + case['input'] if case.get('input') else '')
        pair = {
            'case_id': case_id,
            'kind': 'material_qa' if is_qa else 'instruction_probe',
            'group': case['group'],
            'question': question,
            'system_prompt': case.get('system', ''),
            'reference_answer': case.get('output') if is_qa else None,
            'expected_json': None if is_qa else case['expected'],
            'reference_status': ('dataset_reference_not_expert_verified' if case.get('output', '').strip() else 'empty_reference')
                                if is_qa else 'deterministic_expected_json',
        }
        for mode, side in SIDES.items():
            original, row = raw[(case_id, mode)], scored[(case_id, mode)]
            require(all(row.get(k) == v for k, v in original.items()), f'Scored output changed raw data: {case_id}/{mode}')
            require(row['question'] == question and row['group'] == case['group'] and row['kind'] == expected_kind,
                    f'Prompt or group mismatch: {case_id}/{mode}')
            require(row['reference'] == pair['reference_answer'] and row['expected'] == pair['expected_json'],
                    f'Reference mismatch: {case_id}/{mode}')
            scores = {'factual_accuracy_100': None}
            output = {
                'answer': original['answer'],
                'generated_tokens': original['new_tokens'],
                'max_new_tokens': manifest['max_new_tokens'][expected_kind],
                'hit_token_limit': original['hit_token_limit'],
                'scores': scores,
            }
            if is_qa:
                empty = not pair['reference_answer'].strip()
                require(row['empty_reference'] == empty, f'Empty-reference mismatch: {case_id}')
                for metric in ('char_rouge_l_f1', 'char_bigram_f1'):
                    scores[metric + '_100'] = None if empty else percent(row[metric])
                output['score_status'] = 'not_scored_empty_reference' if empty else 'lexical_overlap_only'
                if not empty:
                    likelihood = nll[(case_id, mode)]
                    require(likelihood['sequence_tokens'] - likelihood['scored_tokens'] == likelihood['source_tokens_full'],
                            f'NLL prompt was truncated: {case_id}/{mode}')
                    require(math.isfinite(likelihood['nll']) and likelihood['nll'] >= 0, 'Invalid NLL')
                    output['reference_likelihood'] = {
                        'response_token_nll': likelihood['nll'],
                        'scored_reference_tokens': likelihood['scored_tokens'],
                        'reference_target_truncated': likelihood['target_truncated'],
                        'full_prompt_preserved': True,
                        'note': '单独计算参考答案的条件负对数似然，越低越好；不是该生成回答的准确率或0–100分。',
                    }
                else:
                    output['reference_likelihood'] = None
            else:
                scores['valid_json_100'] = 100 if row['valid_json'] else 0
                scores['strict_instruction_100'] = 100 if row['exact_pass'] else 0
                output['score_status'] = 'deterministic_strict_json_check'
                output['failure_type'] = None if row['exact_pass'] else ('invalid_json' if not row['valid_json'] else 'expected_json_mismatch')
                output['parsed_json'] = json.loads(original['answer'].strip(), parse_constant=reject_constant) if row['valid_json'] else None
            pair[side] = output
        before, after = pair['before_finetuning'], pair['after_finetuning']
        pair['delta_points_after_minus_before'] = {
            key: delta(before['scores'][key], after['scores'][key]) for key in before['scores']
        }
        if is_qa and before['reference_likelihood']:
            require(before['reference_likelihood']['scored_reference_tokens'] == after['reference_likelihood']['scored_reference_tokens'],
                    f'NLL target-token counts differ: {case_id}')
            pair['reference_nll_reduction_before_minus_after'] = round(
                before['reference_likelihood']['response_token_nll'] - after['reference_likelihood']['response_token_nll'], 8)
        pairs.append(pair)

    totals = aggregate(pairs)
    # Fail if packaging has changed the previously published aggregate metrics.
    for mode, side in SIDES.items():
        for metric in ('char_rouge_l_f1', 'char_bigram_f1'):
            require(math.isclose(totals['material_qa'][side][metric + '_100'], 100 * summary['qa'][mode][metric], abs_tol=1e-5),
                    f'QA summary mismatch: {metric}/{mode}')
        require(totals['instruction_probes'][side]['strict_pass_count'] == summary['probes'][mode]['exact_pass'], 'Probe summary mismatch')

    return {
        'schema_version': '1.0',
        'title': 'Qwen3-8B 同题微调前后实际回答与逐题量化分数',
        'provenance': {
            'evaluation_started_at': manifest['started_at'],
            'operation': '整理已有真实GPU评测；本次导出没有重新推理或训练，也未改写回答。',
            'before_model': manifest['model'],
            'after_model': manifest['adapter'],
            'adapter_sha256': manifest['adapter_sha256'],
            'source_sha256': {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in INPUTS},
        },
        'protocol': {key: manifest[key] for key in ('gpu', 'dtype', 'attention', 'seed', 'decoding', 'max_new_tokens', 'generation_batch_size')},
        'scoring_rules': {
            'scale': '以_100结尾的分数范围0–100，越高越好。null表示未评分/不适用，不是0分。不合成为一个总分。',
            'normalization': '材料文本先casefold，再仅保留Unicode字母和数字；忽略空白和标点。',
            'char_rouge_l_f1_100': '100 × 2 × 字符最长公共子序列长度 / (规范化回答长度 + 规范化参考长度)。',
            'char_bigram_f1_100': '100 × 2 × 字符二元组多重集合交集数量 / (回答二元组数量 + 参考二元组数量)。',
            'valid_json_100': '完整输出去除首尾空白后可作为JSON解析=100，否则=0。Markdown代码围栏不剥离。',
            'strict_instruction_100': '纯JSON可解析，且键、值、列表顺序和类型与expected_json匹配=100，否则=0。整数/浮点等值允许math.isclose(rel_tol=1e-9, abs_tol=1e-8)，数字字符串和布尔值不等于数字。',
            'factual_accuracy_100': '全部为null：未做独立材料专家盲评；不得用文本重合度冒充事实准确率。',
            'delta_points_after_minus_before': '微调后分数减微调前分数，正值表示该项分数提高；不是相对百分比增幅。',
            'hit_token_limit': '达到生成token上限，回答可能未完成；文件保留所有实际已生成文字，不补写未生成部分。',
            'reference_likelihood': '单独的teacher-forced参考答案NLL；完整保留prompt，仅在1536token预算内截断参考答案。越低越好，不能与0–100分混用。',
        },
        'limitations_zh': [
            '材料QA来自参与过checkpoint选择的验证集，不是独立测试集；题目精确去重，但未按来源文档隔离。',
            '材料参考答案未经独立专家核验，文本重合度只衡量表达与参考的相似性。',
            '材料生成上限512tokens，微调前24/24、微调后19/24触及上限，长度截断会影响分数。',
            '部分问题提到“输入文本/证据”，但原数据input为空；本次未额外补充文献或检索上下文。',
            '16条人工新编诊断仅检验给定信息处理、格式与路由决策，没有实际调用工具，不代表完整Agent能力。',
            '没有独立专家评分，也没有使用另一个LLM裁判；不能据这些分数承诺工程决策正确性。',
        ],
        'selection': {'scope': '全部已测生成样本：24条材料QA + 16条诊断，每题微调前后各1次；不是多随机种子重复实验。'},
        'summary': totals,
        'supplementary_reference_nll_summary': {
            'scope': '另一次完整prompt评测中的95条非空参考，不是仅此40组回答的汇总。',
            **summary['nll_full_prompt'],
        },
        'cases': pairs,
    }


def select_examples(document):
    examples = []
    groups = sorted({p['group'] for p in document['cases'] if p['kind'] == 'material_qa'})
    for group in groups:
        eligible = sorted((p for p in document['cases'] if p['kind'] == 'material_qa' and p['group'] == group and p['reference_answer'].strip()), key=lambda p: p['case_id'])
        examples.extend(eligible[:2])
    require(len(examples) == 8 and set(Counter(p['group'] for p in examples).values()) == {2}, 'Expected two QA examples per topic')
    result = {k: v for k, v in document.items() if k not in ('summary', 'cases', 'supplementary_reference_nll_summary')}
    result['title'] = '8组材料问答：微调前后实际回答与量化分数'
    result['selection'] = {
        'scope': '4个材料主题，每组2条，共8组；全量结果见paired_comparisons.json。',
        'rule': '每个主题排除空参考，然后按case_id升序取前2条，不按模型得分或提升幅度挑选。',
        'case_ids': [p['case_id'] for p in examples],
    }
    result['summary'] = aggregate(examples)
    result['cases'] = examples
    return result


def publish(path, payload):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Existing experiment directory')
    args = parser.parse_args()
    document = build_document(args.root)
    for name, payload in (('paired_comparisons.json', document), ('paired_comparison_examples_8.json', select_examples(document))):
        path = args.root / 'results' / name
        publish(path, payload)
        print(f'{path}: {len(payload["cases"])} pairs')


if __name__ == '__main__':
    main()
