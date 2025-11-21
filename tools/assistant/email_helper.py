import os
from email.message import EmailMessage
import smtplib
from typing import Optional

def send_email_smtp(smtp_host: str, smtp_port: int, username: Optional[str], password: Optional[str],
                    to: str, subject: str, body: str, dry_run: bool = True) -> str:
    """Send an email via SMTP. If `dry_run` is True or credentials are missing, returns a dry-run message.
    """
    if dry_run or not (username and password):
        return f"DRY_RUN: Would send email to {to} subject {subject}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = to
    msg.set_content(body)

    try:
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        return "Email sent"
    except Exception as e:
        return f"SMTP send error: {e}"

def send_email_sendgrid(api_key: Optional[str], from_email: str, to: str, subject: str, body: str, dry_run: bool = True) -> str:
    """Send via SendGrid API if api_key provided. If dry_run or missing key, returns dry-run message."""
    if dry_run or not api_key:
        return f"DRY_RUN: Would send email (SendGrid) to {to} subject {subject}"
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
    except Exception as e:
        return f"SendGrid library not installed: {e}"
    try:
        message = Mail(from_email=from_email, to_emails=to, subject=subject, plain_text_content=body)
        sg = SendGridAPIClient(api_key)
        resp = sg.send(message)
        return f"SendGrid response: {resp.status_code}"
    except Exception as e:
        return f"SendGrid error: {e}"
