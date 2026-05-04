import unittest

from langchain_core.documents import Document

from app.services.rag_trace import (
    begin_rag_trace,
    end_rag_trace,
    get_effective_top_k,
    get_rag_trace_payload,
    record_retrieval,
)


class RagTraceTestCase(unittest.TestCase):
    def test_trace_records_retrieval_calls_and_aggregates_unique_docs(self):
        token = begin_rag_trace(top_k=5)
        try:
            record_retrieval(
                "问题一",
                [
                    Document(
                        page_content="CPU 告警内容",
                        metadata={
                            "_source": "E:/develop/Python/super_biz_agent/aiops-docs/cpu_high_usage.md",
                            "_file_name": "cpu_high_usage.md",
                            "h1": "CPU 使用率过高",
                        },
                    ),
                    Document(
                        page_content="慢响应内容",
                        metadata={
                            "_source": "E:/develop/Python/super_biz_agent/aiops-docs/slow_response.md",
                            "_file_name": "slow_response.md",
                        },
                    ),
                ],
            )
            record_retrieval(
                "问题二",
                [
                    Document(
                        page_content="再次命中 CPU 告警内容",
                        metadata={
                            "_source": "E:/develop/Python/super_biz_agent/aiops-docs/cpu_high_usage.md",
                            "_file_name": "cpu_high_usage.md",
                        },
                    )
                ],
            )

            payload = get_rag_trace_payload()

            self.assertTrue(payload["enabled"])
            self.assertEqual(payload["requested_top_k"], 5)
            self.assertEqual(len(payload["retrieval_calls"]), 2)
            self.assertEqual(len(payload["retrieved_docs"]), 2)
            self.assertEqual(payload["retrieved_docs"][0]["file_name"], "cpu_high_usage.md")
            self.assertEqual(payload["retrieved_docs"][0]["rank"], 1)
            self.assertEqual(payload["retrieved_docs"][0]["queries"], ["问题一", "问题二"])
        finally:
            end_rag_trace(token)

    def test_get_effective_top_k_falls_back_when_trace_not_enabled(self):
        self.assertEqual(get_effective_top_k(3), 3)

    def test_payload_is_disabled_when_trace_not_enabled(self):
        payload = get_rag_trace_payload()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["retrieved_docs"], [])


if __name__ == "__main__":
    unittest.main()
