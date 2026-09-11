import os
import datetime
from typing import Dict, Any, Optional

class EarlyWarningAlertEngine:
    """
    Evaluates fire detection confidence, temporal persistence, severity, and satellite confirmation
    to trigger Early Warning alerts.
    """
    def __init__(self, min_confidence: float = 0.60, min_persistence_frames: int = 4):
        self.min_confidence = min_confidence
        self.min_persistence_frames = min_persistence_frames

    def evaluate_alert_condition(self, fusion_result: Dict[str, Any],
                                 persistence_frames: int = 1,
                                 satellite_confirmed: bool = False,
                                 location_name: str = "Station Lookout Alpha",
                                 lat: float = 37.7749, lon: float = -122.4194) -> Optional[Dict[str, Any]]:
        
        fire_prob = fusion_result.get("fire_probability", 0.0)
        risk_level = fusion_result.get("risk_level", "LOW")
        
        # Trigger Condition: (Fire prob > threshold AND Persistent N frames) OR (Satellite Confirmed)
        is_persistent = (persistence_frames >= self.min_persistence_frames)
        should_trigger = (fire_prob >= self.min_confidence and is_persistent) or satellite_confirmed or (risk_level in ["HIGH", "CRITICAL"])
        
        if not should_trigger:
            return None

        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        message = (
            f"🔥 EARLY WARNING FIRE ALERT 🔥\n"
            f"Risk Level: {risk_level}\n"
            f"Location: {location_name}\n"
            f"Coordinates: Lat {lat}, Lon {lon}\n"
            f"Timestamp: {now_str}\n"
            f"Fire Confidence: {round(fire_prob * 100, 1)}%\n"
            f"Smoke Confidence: {round(fusion_result.get('smoke_probability', 0.0) * 100, 1)}%\n"
            f"Severity: {fusion_result.get('risk_level')}\n"
            f"Detection Source: {'NASA Satellite + Multi-Model Fusion' if satellite_confirmed else 'Camera Temporal Sequence'}\n"
        )

        alert_payload = {
            "title": f"🔥 FIRE ALERT: {risk_level} Risk at {location_name}",
            "message": message,
            "timestamp": now_str,
            "location": location_name,
            "latitude": lat,
            "longitude": lon,
            "severity": risk_level,
            "risk": risk_level,
            "fire_confidence": round(fire_prob, 4),
            "source": "NASA Satellite" if satellite_confirmed else "Ground Camera Fusion",
            "status": "ACTIVE"
        }

        # Optional Environment Notification Integrations
        self._dispatch_email_if_configured(alert_payload)
        self._dispatch_telegram_if_configured(alert_payload)

        return alert_payload

    def _dispatch_email_if_configured(self, alert: Dict[str, Any]):
        sender = os.getenv("EMAIL_SENDER")
        if sender:
            print(f"[Alert Engine] Email dispatch initialized to {sender} (Simulated/Configured)")

    def _dispatch_telegram_if_configured(self, alert: Dict[str, Any]):
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if bot_token:
            print(f"[Alert Engine] Telegram bot notification dispatched (Simulated/Configured)")
