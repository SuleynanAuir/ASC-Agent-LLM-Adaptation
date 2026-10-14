"""Integrity checks for JSON packaging of measured model outputs."""
import json
import math
import unittest

from analyze_comparison import overlaps, strict_equal
from export_paired_comparisons import ROOT, SIDES, build_document, index_unique, select_examples


class PairedExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = build_document()
        cls.pairs = {p['case_id']: p for p in cls.document['cases']}
        cls.raw = index_unique(json.loads((ROOT / 'results/comparison/generation.json').read_text()), ('case_id', 'mode'))

    def test_exact_original_answers_and_matching_questions(self):
        self.assertEqual(len(self.pairs), 40)
        for case_id, pair in self.pairs.items():
            for mode, side in SIDES.items():
                self.assertEqual(pair['question'], self.raw[(case_id, mode)]['question'])
                self.assertEqual(pair[side]['answer'], self.raw[(case_id, mode)]['answer'])
                self.assertEqual(pair[side]['hit_token_limit'], self.raw[(case_id, mode)]['hit_token_limit'])

    def test_recompute_scores_from_answers(self):
        for pair in self.pairs.values():
            for side in SIDES.values():
                row = pair[side]
                self.assertIsNone(row['scores']['factual_accuracy_100'])
                if pair['kind'] == 'material_qa' and pair['reference_answer'].strip():
                    recalculated = overlaps(row['answer'], pair['reference_answer'])
                    for key in ('char_rouge_l_f1', 'char_bigram_f1'):
                        self.assertAlmostEqual(row['scores'][key + '_100'], 100 * recalculated[key], places=5)
                elif pair['kind'] == 'instruction_probe':
                    try:
                        parsed = json.loads(row['answer'].strip())
                        valid, passed = True, strict_equal(parsed, pair['expected_json'])
                    except ValueError:
                        valid, passed = False, False
                    self.assertEqual(row['scores']['valid_json_100'], 100 * valid)
                    self.assertEqual(row['scores']['strict_instruction_100'], 100 * passed)

    def test_null_not_zero_for_missing_reference(self):
        pair = self.pairs['qa-04126']
        for side in SIDES.values():
            self.assertIsNone(pair[side]['scores']['char_rouge_l_f1_100'])
            self.assertIsNone(pair[side]['scores']['char_bigram_f1_100'])
            self.assertIsNone(pair[side]['reference_likelihood'])
        self.assertIsNone(pair['delta_points_after_minus_before']['char_rouge_l_f1_100'])

    def test_improvement_regression_and_unchanged_failure_retained(self):
        self.assertEqual(self.pairs['probe-08']['delta_points_after_minus_before']['strict_instruction_100'], 100)
        self.assertEqual(self.pairs['probe-05']['delta_points_after_minus_before']['strict_instruction_100'], -100)
        self.assertEqual(self.pairs['probe-11']['delta_points_after_minus_before']['strict_instruction_100'], 0)
        self.assertEqual(self.pairs['probe-05']['after_finetuning']['failure_type'], 'invalid_json')

    def test_summary_and_token_limits(self):
        summary = self.document['summary']
        self.assertEqual(summary['material_qa_count'], 24)
        self.assertEqual(summary['material_qa_scored_count'], 23)
        self.assertEqual(summary['instruction_probe_count'], 16)
        self.assertEqual(summary['material_qa']['before_finetuning']['hit_token_limit_count'], 24)
        self.assertEqual(summary['material_qa']['after_finetuning']['hit_token_limit_count'], 19)
        for side in SIDES.values():
            self.assertEqual(summary['instruction_probes'][side]['strict_pass_count'], 13)
            self.assertEqual(summary['instruction_probes'][side]['strict_pass_rate_pct'], 81.25)

    def test_example_selection_is_id_based_not_score_based(self):
        examples = select_examples(self.document)
        self.assertEqual(len(examples['cases']), 8)
        self.assertEqual(examples['summary']['paired_questions'], 8)
        self.assertEqual(examples['summary']['instruction_probe_count'], 0)
        for group in {p['group'] for p in examples['cases']}:
            expected = sorted(p['case_id'] for p in self.pairs.values() if p['kind'] == 'material_qa' and p['group'] == group and p['reference_answer'].strip())[:2]
            self.assertEqual([p['case_id'] for p in examples['cases'] if p['group'] == group], expected)

    def test_duplicates_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            index_unique([{'id': 1}, {'id': 1}], ('id',))

    def test_strict_serializable_no_nonfinite_values(self):
        encoded = json.dumps(self.document, ensure_ascii=False, allow_nan=False)
        self.assertEqual(json.loads(encoded), self.document)
        for pair in self.pairs.values():
            for side in SIDES.values():
                for score in pair[side]['scores'].values():
                    self.assertTrue(score is None or math.isfinite(score))


if __name__ == '__main__':
    unittest.main()
