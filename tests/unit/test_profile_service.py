"""Unit tests for the self-service profile endpoints in auth_service —
update_profile, request_password_change_otp, change_password,
request_email_change_otp, change_email."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.services import auth_service


def _uid() -> str:
    return f"user_{uuid.uuid4().hex[:8]}"


def _email() -> str:
    return f"{uuid.uuid4().hex}@example.com"


# ---------------------------------------------------------------------------
# update_profile
# ---------------------------------------------------------------------------

class TestUpdateProfile:
    def test_updates_only_provided_fields(self):
        user_id = _uid()
        existing = {"id": user_id, "username": "u", "first_name": "Old", "last_name": "Name",
                    "email_id": "u@example.com", "persona": "user"}
        updated = {**existing, "first_name": "New"}
        mock_users = MagicMock()
        mock_users.get_by_id.side_effect = [existing, updated]

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.get_auth_audit_repository", return_value=MagicMock()):
            result = auth_service.update_profile(user_id, first_name="New", last_name=None)

        mock_users.update_user.assert_called_once_with(user_id, {"first_name": "New"})
        assert result["first_name"] == "New"

    def test_no_fields_skips_update_call(self):
        user_id = _uid()
        existing = {"id": user_id, "username": "u", "email_id": "u@example.com", "persona": "user"}
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = existing

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.get_auth_audit_repository", return_value=MagicMock()):
            auth_service.update_profile(user_id, first_name=None, last_name=None)

        mock_users.update_user.assert_not_called()

    def test_user_not_found_raises_404(self):
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = None

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.update_profile(_uid(), first_name="X", last_name=None)

        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# request_password_change_otp
# ---------------------------------------------------------------------------

class TestRequestPasswordChangeOtp:
    def test_issues_otp_to_own_email(self):
        user_id = _uid()
        email = _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": email}

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service._issue_otp") as mock_issue:
            auth_service.request_password_change_otp(user_id)

        mock_issue.assert_called_once_with(email=email, purpose="password_change", user_id=user_id)

    def test_user_not_found_raises_404(self):
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = None

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_password_change_otp(_uid())

        assert exc_info.value.status_code == 404

    def test_no_email_on_account_raises_400(self):
        user_id = _uid()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": None}

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_password_change_otp(user_id)

        assert exc_info.value.status_code == 400

    def test_second_request_within_window_is_rate_limited(self):
        user_id = _uid()
        email = _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": email}

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service._issue_otp"):
            auth_service.request_password_change_otp(user_id)  # consumes the 1 token
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_password_change_otp(user_id)

        assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# change_password
# ---------------------------------------------------------------------------

class TestChangePassword:
    def _user(self, user_id, email):
        return {"id": user_id, "email_id": email, "password_hash": "hashed"}

    def test_success_updates_password_and_revokes_sessions(self):
        user_id = _uid()
        email = _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = self._user(user_id, email)
        mock_refresh_tokens = MagicMock()

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.get_auth_audit_repository", return_value=MagicMock()), \
             patch("app.services.auth_service.verify_password", return_value=True), \
             patch("app.services.auth_service.verify_otp") as mock_verify_otp, \
             patch("app.db.repositories.get_refresh_token_repository", return_value=mock_refresh_tokens), \
             patch("app.services.session_service.end_all_sessions_for_user") as mock_end_sessions, \
             patch("app.services.auth_service.send_password_changed_notice") as mock_notice:
            auth_service.change_password(
                user_id, current_password="old-pw", code="123456", new_password="NewPassw0rd!"
            )

        mock_verify_otp.assert_called_once_with(email=email, code="123456", purpose="password_change")
        assert mock_users.update_user.call_args[0][0] == user_id
        assert "password_hash" in mock_users.update_user.call_args[0][1]
        mock_users.reset_login_attempts.assert_called_once_with(user_id)
        mock_refresh_tokens.revoke_all_for_user.assert_called_once_with(user_id)
        mock_end_sessions.assert_called_once()
        mock_notice.assert_called_once_with(email)

    def test_wrong_current_password_raises_401_without_touching_otp(self):
        user_id = _uid()
        email = _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = self._user(user_id, email)

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.verify_password", return_value=False), \
             patch("app.services.auth_service.verify_otp") as mock_verify_otp:
            with pytest.raises(HTTPException) as exc_info:
                auth_service.change_password(
                    user_id, current_password="wrong", code="123456", new_password="NewPassw0rd!"
                )

        assert exc_info.value.status_code == 401
        mock_verify_otp.assert_not_called()
        mock_users.update_user.assert_not_called()

    def test_invalid_otp_propagates_and_does_not_update_password(self):
        user_id = _uid()
        email = _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = self._user(user_id, email)

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.verify_password", return_value=True), \
             patch("app.services.auth_service.verify_otp",
                   side_effect=HTTPException(status_code=400, detail="Incorrect verification code")):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.change_password(
                    user_id, current_password="old-pw", code="000000", new_password="NewPassw0rd!"
                )

        assert exc_info.value.status_code == 400
        mock_users.update_user.assert_not_called()

    def test_user_not_found_raises_404(self):
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = None

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.change_password(
                    _uid(), current_password="x", code="123456", new_password="NewPassw0rd!"
                )

        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# request_email_change_otp
# ---------------------------------------------------------------------------

class TestRequestEmailChangeOtp:
    def test_issues_otp_to_both_current_and_new_email(self):
        user_id = _uid()
        old_email = _email()
        new_email = _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": old_email}
        mock_users.email_exists.return_value = False

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service._issue_otp") as mock_issue:
            auth_service.request_email_change_otp(user_id, new_email=new_email)

        assert mock_issue.call_count == 2
        mock_issue.assert_any_call(email=old_email, purpose="email_change_old", user_id=user_id)
        mock_issue.assert_any_call(email=new_email, purpose="email_change", user_id=user_id)

    def test_no_email_on_account_raises_400(self):
        user_id = _uid()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": None}

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_email_change_otp(user_id, new_email=_email())

        assert exc_info.value.status_code == 400

    def test_same_as_current_email_raises_400(self):
        user_id = _uid()
        email = _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": email}

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_email_change_otp(user_id, new_email=email)

        assert exc_info.value.status_code == 400

    def test_already_taken_email_raises_409(self):
        user_id = _uid()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": _email()}
        mock_users.email_exists.return_value = True

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_email_change_otp(user_id, new_email=_email())

        assert exc_info.value.status_code == 409

    def test_second_request_within_window_is_rate_limited(self):
        user_id = _uid()
        new_email = _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": _email()}
        mock_users.email_exists.return_value = False

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service._issue_otp"):
            auth_service.request_email_change_otp(user_id, new_email=new_email)
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_email_change_otp(user_id, new_email=new_email)

        assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# change_email
# ---------------------------------------------------------------------------

class TestChangeEmail:
    def test_success_verifies_both_codes_swaps_email_and_notifies_old_address(self):
        user_id = _uid()
        old_email = _email()
        new_email = _email()
        mock_users = MagicMock()
        mock_users.get_by_id.side_effect = [
            {"id": user_id, "email_id": old_email},
            {"id": user_id, "email_id": new_email, "username": "u", "persona": "user"},
        ]
        mock_users.email_exists.return_value = False

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.get_auth_audit_repository", return_value=MagicMock()), \
             patch("app.services.auth_service.verify_otp") as mock_verify_otp, \
             patch("app.services.auth_service.send_email_changed_notice") as mock_notice:
            result = auth_service.change_email(
                user_id, new_email=new_email, current_email_code="111111", new_email_code="222222"
            )

        assert mock_verify_otp.call_count == 2
        mock_verify_otp.assert_any_call(email=old_email, code="111111", purpose="email_change_old")
        mock_verify_otp.assert_any_call(email=new_email, code="222222", purpose="email_change")
        mock_users.update_user.assert_called_once_with(user_id, {"email_id": new_email})
        mock_notice.assert_called_once_with(old_email, new_email)
        assert result["email"] == new_email

    def test_already_taken_at_confirm_time_raises_409(self):
        user_id = _uid()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": _email()}
        mock_users.email_exists.return_value = True

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.verify_otp") as mock_verify_otp:
            with pytest.raises(HTTPException) as exc_info:
                auth_service.change_email(
                    user_id, new_email=_email(), current_email_code="111111", new_email_code="222222"
                )

        assert exc_info.value.status_code == 409
        mock_verify_otp.assert_not_called()

    def test_invalid_old_email_otp_propagates_without_checking_new_code(self):
        user_id = _uid()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": _email()}
        mock_users.email_exists.return_value = False

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.verify_otp",
                   side_effect=HTTPException(status_code=400, detail="Incorrect verification code")) as mock_verify_otp:
            with pytest.raises(HTTPException) as exc_info:
                auth_service.change_email(
                    user_id, new_email=_email(), current_email_code="000000", new_email_code="222222"
                )

        assert exc_info.value.status_code == 400
        mock_verify_otp.assert_called_once()  # stopped after the first (old-email) check
        mock_users.update_user.assert_not_called()

    def test_no_email_on_account_raises_400(self):
        user_id = _uid()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {"id": user_id, "email_id": None}

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.change_email(
                    user_id, new_email=_email(), current_email_code="111111", new_email_code="222222"
                )

        assert exc_info.value.status_code == 400

    def test_user_not_found_raises_404(self):
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = None

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.change_email(
                    _uid(), new_email=_email(), current_email_code="111111", new_email_code="222222"
                )

        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# request_persona_upgrade_otp / upgrade_persona
# ---------------------------------------------------------------------------

def _general_user(user_id: str, email: str) -> dict:
    return {"id": user_id, "username": "u", "first_name": "A", "last_name": "B",
            "email_id": email, "persona": "user", "status": "active"}


class TestRequestPersonaUpgradeOtp:
    def test_issues_otp_to_own_email(self):
        user_id, email = _uid(), _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = _general_user(user_id, email)

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service._issue_otp") as mock_issue:
            auth_service.request_persona_upgrade_otp(user_id)

        mock_issue.assert_called_once_with(email=email, purpose="persona_upgrade", user_id=user_id)

    def test_already_researcher_raises_409(self):
        user_id, email = _uid(), _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {**_general_user(user_id, email), "persona": "researcher"}

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service._issue_otp") as mock_issue:
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_persona_upgrade_otp(user_id)

        assert exc_info.value.status_code == 409
        mock_issue.assert_not_called()

    def test_admin_cannot_self_downgrade_via_upgrade(self):
        user_id, email = _uid(), _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {**_general_user(user_id, email), "persona": "admin"}

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_persona_upgrade_otp(user_id)

        assert exc_info.value.status_code == 403

    def test_inactive_account_raises_403(self):
        user_id, email = _uid(), _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {**_general_user(user_id, email), "status": "deactivated"}

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_persona_upgrade_otp(user_id)

        assert exc_info.value.status_code == 403

    def test_user_not_found_raises_404(self):
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = None

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.request_persona_upgrade_otp(_uid())

        assert exc_info.value.status_code == 404


class TestUpgradePersona:
    def test_promotes_and_returns_token_carrying_new_persona(self):
        user_id, email = _uid(), _email()
        before = _general_user(user_id, email)
        after = {**before, "persona": "researcher"}
        mock_users = MagicMock()
        mock_users.get_by_id.side_effect = [before, after]

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.get_auth_audit_repository", return_value=MagicMock()), \
             patch("app.services.auth_service.verify_otp") as mock_verify, \
             patch("app.services.auth_service.create_access_token", return_value="tok") as mock_token:
            user, access_token = auth_service.upgrade_persona(user_id, code="123456")

        mock_verify.assert_called_once_with(email=email, code="123456", purpose="persona_upgrade")
        mock_users.update_user.assert_called_once_with(user_id, {"persona": "researcher"})
        # The new token must carry the upgraded role, not the old one.
        mock_token.assert_called_once_with(user_id=user_id, persona="expert")
        assert user["persona"] == "expert"
        assert access_token == "tok"

    def test_bad_code_leaves_persona_untouched(self):
        user_id, email = _uid(), _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = _general_user(user_id, email)

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.verify_otp",
                   side_effect=HTTPException(status_code=400, detail="Incorrect verification code")):
            with pytest.raises(HTTPException) as exc_info:
                auth_service.upgrade_persona(user_id, code="000000")

        assert exc_info.value.status_code == 400
        mock_users.update_user.assert_not_called()

    def test_persona_changed_between_request_and_confirm_is_rejected(self):
        """An admin could have promoted them in the window; the OTP must not
        be spendable against an account that is no longer general."""
        user_id, email = _uid(), _email()
        mock_users = MagicMock()
        mock_users.get_by_id.return_value = {**_general_user(user_id, email), "persona": "researcher"}

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.verify_otp") as mock_verify:
            with pytest.raises(HTTPException) as exc_info:
                auth_service.upgrade_persona(user_id, code="123456")

        assert exc_info.value.status_code == 409
        mock_verify.assert_not_called()
        mock_users.update_user.assert_not_called()
