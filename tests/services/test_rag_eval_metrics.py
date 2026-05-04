import unittest

from app.services.rag_eval_metrics import find_first_relevant_rank, hit_at_k, reciprocal_rank


class RagEvalMetricsTestCase(unittest.TestCase):
    def test_find_first_relevant_rank_matches_relative_and_absolute_sources(self):
        retrieved_docs = [
            {"source": "E:/develop/Python/super_biz_agent/aiops-docs/slow_response.md", "file_name": "slow_response.md"},
            {"source": "E:/develop/Python/super_biz_agent/aiops-docs/cpu_high_usage.md", "file_name": "cpu_high_usage.md"},
        ]

        rank = find_first_relevant_rank(["aiops-docs/cpu_high_usage.md"], retrieved_docs)

        self.assertEqual(rank, 2)

    def test_find_first_relevant_rank_matches_uploaded_file_with_same_basename(self):
        retrieved_docs = [
            {"source": "E:/develop/Python/super_biz_agent/uploads/cpu_high_usage.md", "file_name": "cpu_high_usage.md"},
            {"source": "E:/develop/Python/super_biz_agent/uploads/slow_response.md", "file_name": "slow_response.md"},
        ]

        rank = find_first_relevant_rank(["aiops-docs/cpu_high_usage.md"], retrieved_docs)

        self.assertEqual(rank, 1)

    def test_find_first_relevant_rank_returns_none_when_missing(self):
        retrieved_docs = [
            {"source": "E:/develop/Python/super_biz_agent/aiops-docs/slow_response.md", "file_name": "slow_response.md"},
        ]

        rank = find_first_relevant_rank(["aiops-docs/memory_high_usage.md"], retrieved_docs)

        self.assertIsNone(rank)

    def test_hit_and_mrr_metrics(self):
        self.assertEqual(hit_at_k(2, 3), 1)
        self.assertEqual(hit_at_k(4, 3), 0)
        self.assertEqual(reciprocal_rank(2), 0.5)
        self.assertEqual(reciprocal_rank(None), 0.0)


if __name__ == "__main__":
    unittest.main()
