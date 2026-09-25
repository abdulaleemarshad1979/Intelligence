"""Multi-Channel Surveillance Alert Dispatcher.

Supports:
1. Instant WebSocket audio chimes & dashboard popups for command center operators.
2. Telegram Bot instant photo alerts with cryptographic SHA-256 evidence digests.
3. SMS alerts via Twilio and MSG91.
4. Generic JSON Webhook notifications (Slack, Discord, police CAD systems).
5. Asynchronous, zero-blocking dispatch via background thread pool.
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List
import httpx

logger = logging.getLogger(__name__)


class NotificationDispatcher:
    """Asynchronous multi-channel dispatch engine for real-time facial recognition alerts."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="AlertDispatch")
        self.stats = {
            "total_dispatched": 0,
            "telegram_sent": 0,
            "sms_sent": 0,
            "webhook_sent": 0,
            "websocket_sent": 0,
            "dispatch_errors": 0
        }

        # Dynamic runtime configuration (can also be loaded from environment)
        self.config = {
            "telegram_enabled": bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")),
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
            "twilio_enabled": bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN")),
            "twilio_account_sid": os.getenv("TWILIO_ACCOUNT_SID", ""),
            "twilio_auth_token": os.getenv("TWILIO_AUTH_TOKEN", ""),
            "twilio_from_number": os.getenv("TWILIO_FROM_NUMBER", ""),
            "twilio_to_number": os.getenv("TWILIO_TO_NUMBER", ""),
            "msg91_enabled": bool(os.getenv("MSG91_AUTH_KEY") and os.getenv("MSG91_TEMPLATE_ID")),
            "msg91_auth_key": os.getenv("MSG91_AUTH_KEY", ""),
            "msg91_template_id": os.getenv("MSG91_TEMPLATE_ID", ""),
            "msg91_sender_id": os.getenv("MSG91_SENDER_ID", "APPDIS"),
            "msg91_mobile": os.getenv("MSG91_MOBILE", ""),
            "webhook_enabled": bool(os.getenv("ALERT_WEBHOOK_URL")),
            "webhook_url": os.getenv("ALERT_WEBHOOK_URL", ""),
            "webhook_secret": os.getenv("ALERT_WEBHOOK_SECRET", "")
        }

    def update_config(self, new_config: Dict[str, Any]):
        """Update notification credentials and channels at runtime."""
        for k, v in new_config.items():
            if k in self.config:
                self.config[k] = v
        # Update enabled flags if tokens supplied
        if self.config.get("telegram_bot_token") and self.config.get("telegram_chat_id"):
            self.config["telegram_enabled"] = True
        if self.config.get("twilio_account_sid") and self.config.get("twilio_auth_token"):
            self.config["twilio_enabled"] = True
        if self.config.get("webhook_url"):
            self.config["webhook_enabled"] = True

    def dispatch_alert(self, alert_event: Dict[str, Any]):
        """Dispatches alert across all active channels in the background."""
        self.stats["total_dispatched"] += 1
        self._executor.submit(self._dispatch_worker, alert_event)

    def _dispatch_worker(self, alert: Dict[str, Any]):
        """Worker executing channel notifications."""
        # 1. WebSocket Broadcast to Command Center
        try:
            self._broadcast_websocket(alert)
        except Exception as ex:
            logger.debug(f"WebSocket alert broadcast note: {ex}")

        # 2. Telegram Alert
        if self.config.get("telegram_enabled"):
            try:
                self._send_telegram(alert)
            except Exception as ex:
                self.stats["dispatch_errors"] += 1
                logger.error(f"Telegram dispatch failed: {ex}")

        # 3. Twilio SMS
        if self.config.get("twilio_enabled"):
            try:
                self._send_twilio(alert)
            except Exception as ex:
                self.stats["dispatch_errors"] += 1
                logger.error(f"Twilio dispatch failed: {ex}")

        # 4. MSG91 SMS
        if self.config.get("msg91_enabled"):
            try:
                self._send_msg91(alert)
            except Exception as ex:
                self.stats["dispatch_errors"] += 1
                logger.error(f"MSG91 dispatch failed: {ex}")

        # 5. Generic Webhook
        if self.config.get("webhook_enabled"):
            try:
                self._send_webhook(alert)
            except Exception as ex:
                self.stats["dispatch_errors"] += 1
                logger.error(f"Webhook dispatch failed: {ex}")

    def _broadcast_websocket(self, alert: Dict[str, Any]):
        """Sends instant notification packet to all live WebSocket consoles."""
        try:
            from app.api.routes_alerts import alert_connection_manager
            packet = {
                "type": "FACE_MATCH_ALERT",
                "alert_id": alert.get("alert_id"),
                "target_id": alert.get("target_id"),
                "target_name": alert.get("target_name"),
                "camera_id": alert.get("camera_id"),
                "confidence": alert.get("confidence"),
                "similarity_pct": alert.get("similarity_pct"),
                "timestamp": alert.get("timestamp", time.time()),
                "full_frame_url": alert.get("full_frame_url"),
                "face_crop_url": alert.get("face_crop_url"),
                "raw_frame_hash": alert.get("raw_frame_hash"),
                "play_sound": True,
                "notes": alert.get("notes", "")
            }
            alert_connection_manager.broadcast_sync(packet)
            self.stats["websocket_sent"] += 1
        except Exception as ex:
            logger.debug(f"Alert WS broadcast skipped: {ex}")

    def _send_telegram(self, alert: Dict[str, Any]):
        """Sends rich photo or text alert to Telegram."""
        token = self.config["telegram_bot_token"]
        chat_id = self.config["telegram_chat_id"]
        if not token or not chat_id:
            return

        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert.get("timestamp", time.time())))
        caption = (
            f"🚨 <b>CCTV FACIAL RECOGNITION ALERT</b>\n\n"
            f"👤 <b>Suspect:</b> {alert.get('target_name')} (<code>{alert.get('target_id')}</code>)\n"
            f"📹 <b>Camera:</b> {alert.get('camera_id')}\n"
            f"🎯 <b>Match Confidence:</b> {alert.get('similarity_pct', 0.0)}%\n"
            f"⏰ <b>Time:</b> {ts_str}\n"
            f"🔒 <b>SHA-256 Hash:</b> <code>{alert.get('raw_frame_hash', '')[:16]}...</code>\n"
            f"📝 <b>Notes:</b> {alert.get('notes', 'None')}"
        )

        photo_path = alert.get("face_crop_path") or alert.get("full_frame_path")

        with httpx.Client(timeout=10.0) as client:
            if photo_path and os.path.isfile(photo_path):
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                with open(photo_path, "rb") as f:
                    files = {"photo": f}
                    data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
                    resp = client.post(url, data=data, files=files)
            else:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                resp = client.post(url, json={"chat_id": chat_id, "text": caption, "parse_mode": "HTML"})

            if resp.status_code == 200:
                self.stats["telegram_sent"] += 1
                logger.info(f"Telegram alert sent for target {alert.get('target_id')}")
            else:
                logger.warning(f"Telegram API responded with {resp.status_code}: {resp.text}")

    def _send_twilio(self, alert: Dict[str, Any]):
        """Sends SMS via Twilio API."""
        sid = self.config["twilio_account_sid"]
        token = self.config["twilio_auth_token"]
        from_num = self.config["twilio_from_number"]
        to_num = self.config["twilio_to_number"]
        if not (sid and token and from_num and to_num):
            return

        ts_str = time.strftime("%H:%M:%S", time.localtime(alert.get("timestamp", time.time())))
        msg_body = (
            f"CCTV ALERT: Suspect {alert.get('target_name')} detected at {alert.get('camera_id')} "
            f"with {alert.get('similarity_pct')}% confidence at {ts_str}. Alert ID: {alert.get('alert_id')}."
        )

        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        data = {
            "From": from_num,
            "To": to_num,
            "Body": msg_body
        }

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, data=data, auth=(sid, token))
            if resp.status_code in (200, 201):
                self.stats["sms_sent"] += 1
                logger.info(f"Twilio SMS sent to {to_num}")
            else:
                logger.warning(f"Twilio API responded with {resp.status_code}: {resp.text}")

    def _send_msg91(self, alert: Dict[str, Any]):
        """Sends SMS via MSG91 flow API."""
        auth_key = self.config["msg91_auth_key"]
        template_id = self.config["msg91_template_id"]
        mobile = self.config["msg91_mobile"]
        if not (auth_key and template_id and mobile):
            return

        url = "https://control.msg91.com/api/v5/flow/"
        headers = {
            "authkey": auth_key,
            "content-type": "application/json"
        }
        payload = {
            "template_id": template_id,
            "short_url": "0",
            "recipients": [
                {
                    "mobiles": mobile,
                    "name": alert.get("target_name", "Target"),
                    "camera": alert.get("camera_id", "CAM"),
                    "confidence": str(alert.get("similarity_pct", 0.0))
                }
            ]
        }

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code in (200, 201):
                self.stats["sms_sent"] += 1
                logger.info(f"MSG91 SMS sent to {mobile}")
            else:
                logger.warning(f"MSG91 API responded with {resp.status_code}: {resp.text}")

    def _send_webhook(self, alert: Dict[str, Any]):
        """Sends JSON alert payload to external endpoint (CAD, Slack, Discord)."""
        url = self.config["webhook_url"]
        if not url:
            return

        headers = {"Content-Type": "application/json"}
        if self.config.get("webhook_secret"):
            headers["Authorization"] = f"Bearer {self.config['webhook_secret']}"

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=alert, headers=headers)
            if resp.status_code in (200, 201, 202, 204):
                self.stats["webhook_sent"] += 1
                logger.info(f"Alert webhook delivered to {url}")
            else:
                logger.warning(f"Webhook endpoint returned {resp.status_code}: {resp.text}")

    def get_status(self) -> Dict[str, Any]:
        """Current status and telemetry of notification channels."""
        return {
            "channels": {
                "websocket": True,
                "telegram": self.config.get("telegram_enabled", False),
                "twilio_sms": self.config.get("twilio_enabled", False),
                "msg91_sms": self.config.get("msg91_enabled", False),
                "webhook": self.config.get("webhook_enabled", False)
            },
            "stats": dict(self.stats)
        }


# Singleton dispatcher
notification_dispatcher = NotificationDispatcher()
