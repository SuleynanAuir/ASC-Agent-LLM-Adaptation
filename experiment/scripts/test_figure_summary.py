"""Verify plotting statistics without rerunning models."""
import unittest
import numpy as np
from plot_figure_summary import ROOT, prepare, mean_ci, read, wilson


class FigureSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prepare()
        cls.summary = read(ROOT / 'results/summary.json')

    def test_paired_identical_vectors(self):
        result = mean_ci([1, 2, 3], [1, 2, 3])
        self.assertEqual(result['delta_after_minus_before'], 0)
        self.assertEqual(result['delta_ci95'], [0, 0])

    def test_reference_statistics(self):
        nll = self.summary['nll_full_prompt']
        self.assertEqual(self.data['nll']['n'], 95)
        self.assertAlmostEqual(self.data['nll']['mean'][0], nll['base_mean'])
        self.assertAlmostEqual(self.data['ppl']['mean'][1], nll['tuned_token_weighted_ppl'])
        self.assertTrue(np.allclose(self.data['nll']['delta_ci95'], [-nll['paired_reduction_ci95'][1], -nll['paired_reduction_ci95'][0]]))

    def test_scope_and_exclusion(self):
        self.assertEqual(len(self.data['qa_case_ids']), 24)
        self.assertEqual(len(self.data['lexical_case_ids']), 23)
        self.assertNotIn('qa-04126', self.data['lexical_case_ids'])
        self.assertIn('qa-04126', self.data['qa_case_ids'])
        self.assertEqual(sum(s['n'] for s in self.data['nll_topics'].values()), 95)

    def test_instruction_regression_not_hidden(self):
        self.assertEqual(self.data['instruction']['exact_pass']['counts'], [13, 13])
        self.assertEqual(self.data['instruction']['valid_json']['counts'], [16, 15])
        self.assertEqual([v[4] for v in self.data['probe_exact_pass']], [True, False])
        self.assertEqual([v[7] for v in self.data['probe_exact_pass']], [False, True])

    def test_generation_behavior(self):
        self.assertEqual(self.data['cap_rate']['counts'], [24, 19])
        self.assertAlmostEqual(np.mean(self.data['generation_lengths'][1]), 482.4166666666667)
        self.assertEqual(len(self.data['timing']['base']['batch_seconds']), 6)
        self.assertGreater(self.data['timing']['base']['aggregate_tokens_per_second'], self.data['timing']['tuned']['aggregate_tokens_per_second'])

    def test_wilson_bounds(self):
        self.assertAlmostEqual(wilson(0, 4)[0], 0)
        self.assertAlmostEqual(wilson(4, 4)[1], 100)
        self.assertGreater(wilson(4, 4)[1] - wilson(4, 4)[0], 40)


if __name__ == '__main__':
    unittest.main()
