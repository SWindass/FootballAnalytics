"""Alert system for strategy monitoring.

Provides logging-based alerts with extensibility for Slack/email notifications.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AlertType(str, Enum):
    """Types of monitoring alerts."""

    DRIFT_DETECTED = "drift_detected"
    NEGATIVE_ROI = "negative_roi"
    LOSING_STREAK = "losing_streak"
    STRATEGY_DISABLED = "strategy_disabled"
    STRATEGY_PAUSED = "strategy_paused"
    OPTIMIZATION_COMPLETE = "optimization_complete"
    PARAMETERS_UPDATED = "parameters_updated"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class Alert:
    """Represents a monitoring alert."""

    alert_type: AlertType
    timestamp: datetime
    strategy_name: str
    message: str
    details: dict[str, Any]
    severity: str  # "info", "warning", "critical"


class AlertManager:
    """Manages monitoring alerts with extensibility for notifications.

    Currently logs all alerts via structlog. Can be extended to support:
    - Slack webhooks
    - Email notifications
    - PagerDuty
    - Custom webhooks
    """

    def __init__(self, enable_slack: bool = False, slack_webhook: str | None = None):
        """Initialize alert manager.

        Args:
            enable_slack: Whether to send Slack notifications
            slack_webhook: Slack webhook URL (required if enable_slack is True)
        """
        self.enable_slack = enable_slack
        self.slack_webhook = slack_webhook
        self._alert_history: list[Alert] = []

    def send_alert(self, alert_type: AlertType, **kwargs) -> Alert:
        """Send an alert.

        Args:
            alert_type: Type of alert
            **kwargs: Alert-specific data

        Returns:
            The created Alert object
        """
        strategy_name = kwargs.get("strategy_name", "Unknown")
        message, severity = self._format_alert(alert_type, kwargs)

        alert = Alert(
            alert_type=alert_type,
            timestamp=datetime.utcnow(),
            strategy_name=strategy_name,
            message=message,
            details=kwargs,
            severity=severity,
        )

        # Log the alert
        self._log_alert(alert)

        # Send to Slack if enabled
        if self.enable_slack and self.slack_webhook:
            self._send_slack(alert)

        # Store in history
        self._alert_history.append(alert)

        return alert

    def _format_alert(self, alert_type: AlertType, data: dict) -> tuple[str, str]:
        """Format alert message and determine severity.

        Returns:
            Tuple of (message, severity)
        """
        strategy = data.get("strategy_name", "Unknown")

        if alert_type == AlertType.DRIFT_DETECTED:
            drift_type = data.get("drift_type", "unknown")
            z_score = data.get("z_score", 0)
            cusum = data.get("cusum", 0)
            message = (
                f"Performance drift detected for {strategy}. "
                f"Type: {drift_type}, Z-score: {z_score:.2f}, CUSUM: {cusum:.2f}"
            )
            severity = "warning"

        elif alert_type == AlertType.NEGATIVE_ROI:
            roi = data.get("rolling_roi", 0)
            n_bets = data.get("n_bets", 0)
            message = (
                f"Strategy {strategy} has negative rolling ROI: "
                f"{roi:.2%} over {n_bets} bets"
            )
            severity = "warning"

        elif alert_type == AlertType.LOSING_STREAK:
            streak = data.get("streak_length", 0)
            message = f"Strategy {strategy} on {streak}-bet losing streak"
            severity = "warning"

        elif alert_type == AlertType.STRATEGY_DISABLED:
            reason = data.get("reason", "Unknown")
            message = f"Strategy {strategy} has been AUTO-DISABLED. Reason: {reason}"
            severity = "critical"

        elif alert_type == AlertType.STRATEGY_PAUSED:
            reason = data.get("reason", "Unknown")
            message = f"Strategy {strategy} has been paused. Reason: {reason}"
            severity = "warning"

        elif alert_type == AlertType.OPTIMIZATION_COMPLETE:
            improvement = data.get("improvement", 0)
            applied = data.get("was_applied", False)
            message = (
                f"Optimization complete for {strategy}. "
                f"Improvement: {improvement:.2%}, Applied: {applied}"
            )
            severity = "info"

        elif alert_type == AlertType.PARAMETERS_UPDATED:
            version = data.get("version", 0)
            message = f"Strategy {strategy} parameters updated to version {version}"
            severity = "info"

        elif alert_type == AlertType.VALIDATION_FAILED:
            reason = data.get("reason", "Unknown")
            message = f"Quarterly validation failed for {strategy}. Reason: {reason}"
            severity = "critical"

        else:
            message = f"Unknown alert for {strategy}: {data}"
            severity = "info"

        return message, severity

    def _log_alert(self, alert: Alert) -> None:
        """Log alert using structlog."""
        log_method = {
            "info": logger.info,
            "warning": logger.warning,
            "critical": logger.error,
        }.get(alert.severity, logger.info)

        log_method(
            alert.message,
            alert_type=alert.alert_type.value,
            strategy=alert.strategy_name,
            **alert.details,
        )

    def _send_slack(self, alert: Alert) -> None:
        """Send alert to Slack webhook.

        Note: This is a placeholder. Implement actual Slack integration
        when webhook is configured.
        """
        # TODO: Implement Slack webhook integration
        # import httpx
        # payload = {
        #     "text": f"[{alert.severity.upper()}] {alert.message}",
        #     "blocks": [...]
        # }
        # httpx.post(self.slack_webhook, json=payload)
        pass

    def get_recent_alerts(
        self, limit: int = 10, alert_type: AlertType | None = None
    ) -> list[Alert]:
        """Get recent alerts from history.

        Args:
            limit: Maximum number of alerts to return
            alert_type: Filter by alert type (optional)

        Returns:
            List of recent alerts
        """
        alerts = self._alert_history
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        return list(reversed(alerts[-limit:]))

    def clear_history(self) -> None:
        """Clear alert history."""
        self._alert_history.clear()
