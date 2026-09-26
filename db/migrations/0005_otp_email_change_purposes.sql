-- Add 'email_change' and 'email_change_old' to otp_codes.purpose
--
-- IMPORTANT — run this in the AUTH project's SQL editor:
-- https://yiwycwaziewmyenfruyg.supabase.co
--
-- Email change now requires OTP confirmation from BOTH the current and
-- new email (see auth_service.request_email_change_otp /
-- confirm_email_change), but otp_codes_purpose_check was never updated
-- to allow these two new purpose values — every request-otp call for an
-- email change fails with a check constraint violation. Existing
-- purposes (signup, password_reset, password_change) are unchanged.

alter table otp_codes drop constraint if exists otp_codes_purpose_check;

alter table otp_codes
    add constraint otp_codes_purpose_check
    check (purpose in ('signup', 'password_reset', 'password_change', 'email_change', 'email_change_old'));
