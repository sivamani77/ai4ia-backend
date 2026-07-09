-- Phase 9: Admin user management — allow a 'deactivated' status
--
-- IMPORTANT — run this in the AUTH project's SQL editor:
-- https://yiwycwaziewmyenfruyg.supabase.co
--
-- Adds 'deactivated' to app_users.status for soft-delete (admin can
-- deactivate an account without losing history — reversible via
-- reactivate). Existing statuses (pending_verification, active, locked)
-- are unchanged.

alter table app_users drop constraint if exists app_users_status_check;

alter table app_users
    add constraint app_users_status_check
    check (status in ('pending_verification', 'active', 'locked', 'deactivated'));
