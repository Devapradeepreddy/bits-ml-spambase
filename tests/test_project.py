"""Meaningful regression checks for leakage, artifact consistency and input errors."""
import io
import json
import unittest

import joblib
import numpy as np
import pandas as pd

from src.core import ROOT, FEATURES, TARGET, METRICS, MODEL_FILES, read_test_csv, validate_frame, feature_groups, evaluate


class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.X, cls.y = read_test_csv(ROOT / "test_data.csv")
        cls.models = {name: joblib.load(ROOT / "model" / file) for name, file in MODEL_FILES.items()}

    def test_saved_metrics_reproduce(self):
        expected = pd.read_csv(ROOT / "results" / "metrics.csv").set_index("Model")
        for name, model in self.models.items():
            actual = evaluate(model, self.X, self.y)["metrics"]
            for metric in METRICS:
                self.assertAlmostEqual(expected.loc[name, metric], actual[metric], places=8)

    def test_no_duplicate_feature_group_leakage(self):
        raw = pd.read_csv(ROOT / "data" / "spambase.data", header=None, names=FEATURES + [TARGET])
        groups = feature_groups(raw)
        membership = pd.read_csv(ROOT / "results" / "split_membership.csv")
        self.assertEqual(len(membership), len(raw))
        self.assertEqual(set(membership.source_row_0_based), set(range(len(raw))))
        train = membership.loc[membership.partition == "train", "source_row_0_based"]
        test = membership.loc[membership.partition == "test", "source_row_0_based"]
        self.assertFalse(set(groups[train]) & set(groups[test]))
        pd.testing.assert_frame_equal(raw.iloc[test].reset_index(drop=True), self.frame, check_dtype=False)

    def test_missing_column_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_frame(self.frame.drop(columns=FEATURES[0]))

    def test_bad_target_rejected(self):
        bad = self.frame.copy()
        bad.loc[0, TARGET] = 2
        with self.assertRaisesRegex(ValueError, "only 0"):
            validate_frame(bad)

    def test_reordered_columns_work(self):
        X, y = validate_frame(self.frame.loc[:, list(reversed(self.frame.columns))])
        pd.testing.assert_frame_equal(X, self.X)
        pd.testing.assert_series_equal(y, self.y)

    def test_duplicate_headers_rejected_before_pandas_renames(self):
        text = self.frame.head(2).to_csv(index=False)
        text = text.replace(FEATURES[1], FEATURES[0], 1)
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            read_test_csv(text.encode())

    def test_missing_cells_use_training_medians(self):
        changed = self.frame.head(10).copy()
        changed.loc[0, FEATURES[0]] = np.nan
        X, y = validate_frame(changed)
        for model in self.models.values():
            output = evaluate(model, X, y)
            self.assertTrue(np.isfinite(output["scores"]).all())

    def test_single_class_auc_is_unavailable(self):
        subset = self.frame.loc[self.frame.spam == 0].head(10)
        X, y = validate_frame(subset)
        for model in self.models.values():
            output = evaluate(model, X, y)
            self.assertIsNone(output["metrics"]["AUC"])
            self.assertEqual(np.array(output["confusion_matrix"]).shape, (2, 2))

    def test_empty_and_non_numeric_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "no data"):
            validate_frame(self.frame.head(0))
        bad = self.frame.head(3).astype(object)
        bad.loc[0, FEATURES[0]] = "invalid"
        with self.assertRaisesRegex(ValueError, "numeric"):
            validate_frame(bad)

    def test_infinite_and_negative_values_rejected(self):
        for value in [float("inf"), -1.0]:
            bad = self.frame.head(3).copy()
            bad.loc[0, FEATURES[0]] = value
            with self.assertRaises(ValueError):
                validate_frame(bad)

    def test_auc_uses_svm_decision_scores(self):
        model = self.models["Support Vector Machine"]
        result = evaluate(model, self.X, self.y)
        self.assertEqual(result["score_name"], "spam_decision_score")
        self.assertGreater(len(np.unique(result["scores"])), 2)

    def test_confusion_counts_sum_to_test_rows(self):
        for model in self.models.values():
            result = evaluate(model, self.X, self.y)
            self.assertEqual(np.array(result["confusion_matrix"]).sum(), len(self.y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
