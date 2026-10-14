"""Small deterministic tests for metric definitions and paired-result integrity."""
import json
import random
import unittest
from pathlib import Path

from analyze_comparison import lcs_length, overlaps, strict_equal

ROOT=Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def test_lcs_against_dynamic_programming(self):
        rng=random.Random(42)
        for _ in range(100):
            a=''.join(rng.choices('abc材料',k=rng.randrange(12)))
            b=''.join(rng.choices('abc材料',k=rng.randrange(12)))
            previous=[0]*(len(b)+1)
            for c in a:
                current=[0]
                for j,d in enumerate(b):
                    current.append(previous[j]+1 if c==d else max(previous[j+1],current[-1]))
                previous=current
            self.assertEqual(lcs_length(a,b),previous[-1])

    def test_overlap_bounds(self):
        self.assertEqual(overlaps('材料123','材料123')['char_rouge_l_f1'],1)
        self.assertEqual(overlaps('abc','xyz')['char_rouge_l_f1'],0)
        self.assertAlmostEqual(overlaps('abc','ac')['char_rouge_l_f1'],.8)

    def test_strict_json_types(self):
        self.assertTrue(strict_equal({'a':75.0},{'a':75}))
        self.assertFalse(strict_equal({'a':'75'},{'a':75}))
        self.assertFalse(strict_equal({'a':True},{'a':1}))
        self.assertFalse(strict_equal({'a':1,'extra':0},{'a':1}))

    def test_paired_outputs(self):
        data=json.loads((ROOT/'results/comparison/nll.json').read_text())
        base={r['case_id'] for r in data if r['mode']=='base'}
        tuned={r['case_id'] for r in data if r['mode']=='tuned'}
        self.assertEqual(base,tuned)
        self.assertEqual(len(base),96)
        gen=json.loads((ROOT/'results/comparison/generation.json').read_text())
        self.assertEqual(len(gen),80)
        for mode in ['base','tuned']:
            self.assertEqual(sum(r['mode']==mode and r['kind']=='qa' for r in gen),24)
            self.assertEqual(sum(r['mode']==mode and r['kind']=='probes' for r in gen),16)

    def test_reference_exclusion(self):
        summary=json.loads((ROOT/'results/summary.json').read_text())
        self.assertEqual(summary['nll']['n'],96)
        self.assertEqual(summary['nll_valid_reference']['n'],95)
        self.assertEqual(summary['qa']['base']['lexical_n'],23)
        self.assertEqual(summary['data_audit']['empty_reference_ids'],['qa-04126'])

    def test_full_prompt_integrity(self):
        rows=json.loads((ROOT/'results/full_prompt_nll/nll.json').read_text())
        self.assertEqual(len(rows),192)
        for row in rows:
            self.assertEqual(row['sequence_tokens']-row['scored_tokens'],row['source_tokens_full'])
        summary=json.loads((ROOT/'results/summary.json').read_text())
        self.assertEqual(summary['nll_full_prompt']['n'],95)
        self.assertEqual(summary['prompt_coverage']['full_question_retained_count'],17)
        self.assertEqual(summary['nll_full_prompt']['improved_count'],95)


if __name__=='__main__': unittest.main()
