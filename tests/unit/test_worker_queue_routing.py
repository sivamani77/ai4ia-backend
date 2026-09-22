"""Video jobs must be routable to a per-environment queue.

release and uat share one Redis, so both workers consumed the same
"video_jobs" queue and whichever claimed a task first ran the job. Jobs
submitted to uat were therefore rendered by release code roughly at random —
observed as two jobs 70 minutes apart running different ffmpeg commands, one
correct and one producing a video track a third the length of its audio.
"""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.core.settings import settings  # noqa: E402


class TestQueueRouting:
    def test_video_task_queue_follows_the_setting(self):
        from app.worker.tasks import run_video_job_task
        assert run_video_job_task.queue == settings.worker_queue

    def test_default_is_unchanged(self):
        """release must keep consuming "video_jobs" with no config change."""
        assert settings.worker_queue == "video_jobs"

    def test_queue_is_not_a_hardcoded_literal(self):
        """The defect was `queue="video_jobs"` on the decorator."""
        source = (PROJECT_ROOT / "backend/app/worker/tasks.py").read_text()
        assert 'queue="video_jobs"' not in source
        assert "queue=settings.worker_queue" in source

    def test_env_override_reroutes_the_task(self, monkeypatch):
        monkeypatch.setenv("WORKER_QUEUE", "video_jobs_uat")
        import app.core.settings as settings_module
        reloaded = importlib.reload(settings_module)
        try:
            assert reloaded.settings.worker_queue == "video_jobs_uat"
        finally:
            monkeypatch.delenv("WORKER_QUEUE", raising=False)
            importlib.reload(settings_module)

    def test_ingest_queue_is_separate_and_untouched(self):
        from app.worker.tasks import run_ingest_job_task
        assert run_ingest_job_task.queue == "ingest_jobs"
