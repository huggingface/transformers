import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../src/transformers/utils/production_debt.py",
)
spec = importlib.util.spec_from_file_location("transformers_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["transformers_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtEvaluator = production_debt_mod.ProductionDebtEvaluator
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = ProductionDebtEvaluator(
            never_equate_intent_to_approval=True,
            max_acceptable_mdi=12.0,
        )

    def test_clean_model_inference_passes_readiness(self) -> None:
        report = self.evaluator.evaluate_inference(
            model_id="meta-llama/Meta-Llama-3-70B",
            allocated_kv_cache_mb=2048.0,
            utilized_kv_cache_mb=1980.0,
            latency_ms_per_token=24.5,
            autoregressive_loop_count=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.mdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_model_inference_fails_debt(self) -> None:
        report = self.evaluator.evaluate_inference(
            model_id="legacy/unquantized-model-v1",
            allocated_kv_cache_mb=8192.0,  # High memory inflation
            utilized_kv_cache_mb=2048.0,
            latency_ms_per_token=120.0,  # High latency
            autoregressive_loop_count=4,  # 4 repetitive loops
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.mdi_score, 50.0)
        self.assertIn("HIGH_KV_CACHE_INFLATION_4.00X", report.critical_smells)
        self.assertIn("HIGH_TOKEN_LATENCY_120.0MS", report.critical_smells)
        self.assertIn("DETECTED_4_REPETITIVE_GENERATION_LOOPS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_TOOL_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.evaluator.evaluate_inference("model-1")
        self.evaluator.evaluate_inference("model-2")
        self.evaluator.evaluate_inference("model-3")

        entries = self.evaluator.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.evaluator.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
