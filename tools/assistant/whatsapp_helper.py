import os
from typing import Optional

def send_whatsapp(to: str, body: str, dry_run: bool = True) -> str:
    """Send a WhatsApp message via Twilio if credentials are present.

    Environment variables used (optional): TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM
    If `dry_run` is True or credentials are missing, the function returns a dry-run message.
    """
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_num = os.environ.get("TWILIO_WHATSAPP_FROM")
    if dry_run or not (sid and token and from_num):
        return f"DRY_RUN: Would send WhatsApp to {to}: {body}"

    try:
        from twilio.rest import Client
    except Exception as e:
        return f"Twilio library not installed: {e}"

    try:
        client = Client(sid, token)
        msg = client.messages.create(body=body, from_=f"whatsapp:{from_num}", to=f"whatsapp:{to}")
        return f"Sent: SID={msg.sid}"
    except Exception as e:
        return f"Twilio send error: {e}"
