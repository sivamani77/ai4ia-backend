from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from app.core.settings import settings

logger = logging.getLogger("ai4ia.services.mail")


class MailDeliveryError(Exception):
    """Raised when an email fails to send. Callers should log/alert loudly —
    a silently undeliverable OTP is a support fire, not something to swallow."""


def _send(*, to: str, subject: str, text_body: str, html_body: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f"{settings.mail_from_name} <{settings.mail_from_address}>"
    message["To"] = to
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(settings.mail_smtp_host, settings.mail_smtp_port, timeout=15) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(settings.mail_smtp_user, settings.mail_smtp_password)
            smtp.send_message(message)
    except Exception as exc:
        logger.exception("Failed to send email to %s (subject=%r)", to, subject)
        raise MailDeliveryError(f"Failed to send email to {to}") from exc


_OTP_PURPOSE_TEXT = {
    "signup": ("Verify your AI4IA account", "complete your sign-up"),
    "password_reset": ("Reset your AI4IA password", "reset your password"),
    "password_change": ("Confirm your AI4IA password change", "confirm changing your password"),
    "email_change": ("Confirm your new AI4IA email address", "confirm this is your new email address"),
    "email_change_old": (
        "Confirm you want to change your AI4IA email",
        "confirm you want to change your account's email address",
    ),
}


def send_otp_email(to: str, code: str, purpose: str) -> None:
    subject, intent = _OTP_PURPOSE_TEXT.get(purpose, _OTP_PURPOSE_TEXT["password_reset"])

    text_body = (
        f"Your verification code is: {code}\n\n"
        f"Enter this code to {intent}. It expires in {settings.otp_ttl_minutes} minutes.\n\n"
        "If you didn't request this, you can safely ignore this email."
    )
    html_body = f"""\
<div style="font-family: sans-serif; max-width: 480px; margin: 0 auto;">
  <p>Your verification code is:</p>
  <p style="font-size: 28px; font-weight: bold; letter-spacing: 4px;">{code}</p>
  <p>Enter this code to {intent}. It expires in {settings.otp_ttl_minutes} minutes.</p>
  <p style="color: #666; font-size: 13px;">
    If you didn't request this, you can safely ignore this email.
  </p>
</div>
"""
    _send(to=to, subject=subject, text_body=text_body, html_body=html_body)


def send_password_changed_notice(to: str) -> None:
    subject = "Your AI4IA password was changed"
    text_body = (
        "Your password was just changed. If this wasn't you, please contact support "
        "immediately — all of your active sessions have been signed out."
    )
    html_body = f"""\
<div style="font-family: sans-serif; max-width: 480px; margin: 0 auto;">
  <p>{text_body}</p>
</div>
"""
    _send(to=to, subject=subject, text_body=text_body, html_body=html_body)


def send_email_changed_notice(old_email: str, new_email: str) -> None:
    """Sent to the OLD address after an email change completes — so if an
    attacker with a hijacked session redirects the account's email, the real
    owner still finds out at the address they actually control."""
    subject = "Your AI4IA account email was changed"
    text_body = (
        f"Your account's email was just changed to {new_email}. "
        "If this wasn't you, please contact support immediately."
    )
    html_body = f"""\
<div style="font-family: sans-serif; max-width: 480px; margin: 0 auto;">
  <p>{text_body}</p>
</div>
"""
    _send(to=old_email, subject=subject, text_body=text_body, html_body=html_body)
