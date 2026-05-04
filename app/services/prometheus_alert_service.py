"""Prometheus 告警查询服务。"""

from typing import Any, Dict, Optional

import httpx


class PrometheusAlertError(RuntimeError):
    """Prometheus 告警查询异常。"""


def fetch_prometheus_alerts(
    base_url: str,
    timeout_seconds: float,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    """查询 Prometheus 告警并提取大模型所需字段。"""
    url = f"{base_url.rstrip('/')}/api/v1/alerts"

    try:
        with httpx.Client(timeout=timeout_seconds, transport=transport) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise PrometheusAlertError(f"查询 Prometheus 告警失败: {exc}") from exc

    payload = response.json()
    if payload.get("status") != "success":
        error_message = payload.get("error") or "unknown error"
        raise PrometheusAlertError(f"Prometheus 告警接口返回失败: {error_message}")

    alerts = payload.get("data", {}).get("alerts", [])
    if not isinstance(alerts, list):
        raise PrometheusAlertError("Prometheus 告警接口返回的 alerts 字段格式无效")

    structured_alerts = []
    for alert in alerts:
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        structured_alerts.append(
            {
                "alertname": labels.get("alertname"),
                "description": annotations.get("description"),
                "activeAt": alert.get("activeAt"),
            }
        )

    return {
        "source": "prometheus",
        "status": "success",
        "total": len(structured_alerts),
        "alerts": structured_alerts,
    }
