-- Add 'persona_upgrade' to otp_codes.purpose
--
-- The general -> researcher self-upgrade (auth_service.request_persona_upgrade_otp
-- / upgrade_persona) issues an OTP with purpose 'persona_upgrade', but
-- otp_codes_purpose_check did not allow that value — every request hit
-- "new row for relation otp_codes violates check constraint
-- otp_codes_purpose_check". Same failure mode as 0005, one purpose later.
--
-- This is a reviewable copy: backend/app/db/schema.py applies the equivalent
-- guarded ALTER on every API startup, so a deployed database picks it up
-- without anyone running this file by hand.

alter table otp_codes drop constraint if exists otp_codes_purpose_check;

alter table otp_codes
    add constraint otp_codes_purpose_check
    check (purpose in ('signup', 'password_reset', 'password_change',
                       'email_change', 'email_change_old', 'persona_upgrade'));
