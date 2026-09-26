from fastapi import APIRouter

from app.services.dashboard_service import get_admin_dashboard, get_public_summary

# Public: the landing page calls this before login.
router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

# Admin only: registered with the admin dependency in main.py.
admin_router = APIRouter(prefix="/admin/dashboard", tags=["Dashboard"])


@router.get("")
def dashboard():
    return get_public_summary()


@admin_router.get("")
def admin_dashboard():
    return get_admin_dashboard()
