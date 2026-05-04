import unittest

from app.services.aiops_prompt import build_aiops_task_prompt


class AIOpsTaskPromptTestCase(unittest.TestCase):
    def test_aiops_task_prompt_requires_query_prometheus_alerts_first_when_enabled(self):
        prompt = build_aiops_task_prompt(prometheus_enabled=True)

        self.assertIn("query_prometheus_alerts", prompt)
        self.assertIn("第一步", prompt)
        self.assertIn("Prometheus", prompt)

    def test_aiops_task_prompt_uses_original_monitor_flow_when_disabled(self):
        prompt = build_aiops_task_prompt(prometheus_enabled=False)

        self.assertNotIn("query_prometheus_alerts", prompt)
        self.assertIn("CPU", prompt)
        self.assertIn("日志", prompt)


if __name__ == "__main__":
    unittest.main()
