import asyncio
import unittest
from unittest.mock import patch

import httpx

from mcp_servers import monitor_server
from app.services.prometheus_alert_service import (
    PrometheusAlertError,
    fetch_prometheus_alerts,
)


def build_transport(payload: dict, status_code: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=status_code, json=payload)

    return httpx.MockTransport(handler)


class PrometheusAlertServiceTestCase(unittest.TestCase):
    def test_fetch_alerts_extracts_required_fields(self):
        payload = {
            "status": "success",
            "data": {
                "alerts": [
                    {
                        "labels": {
                            "alertname": "ServiceDown",
                            "job": "ad-service",
                        },
                        "annotations": {
                            "description": "广告微服务下线",
                        },
                        "activeAt": "2025-10-29T08:48:42Z",
                    }
                ]
            },
        }

        result = fetch_prometheus_alerts(
            base_url="http://localhost:9090",
            timeout_seconds=5.0,
            transport=build_transport(payload),
        )

        self.assertEqual(
            result["alerts"],
            [
                {
                    "alertname": "ServiceDown",
                    "description": "广告微服务下线",
                    "activeAt": "2025-10-29T08:48:42Z",
                }
            ],
        )
        self.assertEqual(result["total"], 1)

    def test_fetch_alerts_returns_none_description_when_missing(self):
        payload = {
            "status": "success",
            "data": {
                "alerts": [
                    {
                        "labels": {
                            "alertname": "InstanceHighCpuUsage",
                        },
                        "annotations": {},
                        "activeAt": "2025-10-29T08:48:42Z",
                    }
                ]
            },
        }

        result = fetch_prometheus_alerts(
            base_url="http://localhost:9090",
            timeout_seconds=5.0,
            transport=build_transport(payload),
        )

        self.assertEqual(
            result["alerts"][0],
            {
                "alertname": "InstanceHighCpuUsage",
                "description": None,
                "activeAt": "2025-10-29T08:48:42Z",
            },
        )

    def test_fetch_alerts_raises_when_prometheus_status_is_not_success(self):
        payload = {
            "status": "error",
            "errorType": "bad_data",
            "error": "query timed out",
        }

        with self.assertRaises(PrometheusAlertError):
            fetch_prometheus_alerts(
                base_url="http://localhost:9090",
                timeout_seconds=5.0,
                transport=build_transport(payload),
            )

    def test_fetch_alerts_raises_on_http_error(self):
        payload = {
            "status": "success",
            "data": {
                "alerts": [],
            },
        }

        with self.assertRaises(PrometheusAlertError):
            fetch_prometheus_alerts(
                base_url="http://localhost:9090",
                timeout_seconds=5.0,
                transport=build_transport(payload, status_code=500),
            )

    def test_query_prometheus_alerts_tool_returns_structured_payload_when_enabled(self):
        expected = {
            "source": "prometheus",
            "status": "success",
            "total": 1,
            "alerts": [
                {
                    "alertname": "ServiceDown",
                    "description": "广告微服务下线",
                    "activeAt": "2025-10-29T08:48:42Z",
                }
            ],
        }

        with patch.object(
            monitor_server,
            "fetch_prometheus_alerts",
            return_value=expected,
        ) as mock_fetch:
            with patch.object(monitor_server.config, "prometheus_enabled", True):
                result = asyncio.run(monitor_server.query_prometheus_alerts.run({}))

        self.assertEqual(result.structured_content, expected)
        mock_fetch.assert_called_once()

    def test_query_prometheus_alerts_reports_disabled_when_not_enabled(self):
        with patch.object(monitor_server.config, "prometheus_enabled", False):
            with patch.object(monitor_server, "fetch_prometheus_alerts") as mock_fetch:
                result = asyncio.run(monitor_server.query_prometheus_alerts.run({}))

        self.assertEqual(
            result.structured_content,
            {
                "source": "prometheus",
                "status": "disabled",
                "alerts": [],
                "message": "Prometheus integration is disabled. Use the existing monitor tools instead.",
            },
        )
        mock_fetch.assert_not_called()

    def test_query_prometheus_alerts_reports_error_when_fetch_fails(self):
        with patch.object(monitor_server.config, "prometheus_enabled", True):
            with patch.object(
                monitor_server,
                "fetch_prometheus_alerts",
                side_effect=PrometheusAlertError("boom"),
            ):
                result = asyncio.run(monitor_server.query_prometheus_alerts.run({}))

        self.assertEqual(result.structured_content["source"], "prometheus")
        self.assertEqual(result.structured_content["status"], "error")
        self.assertEqual(result.structured_content["alerts"], [])
        self.assertEqual(result.structured_content["message"], "boom")


if __name__ == "__main__":
    unittest.main()
