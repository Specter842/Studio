"""HTTP trigger for the pipeline, so n8n can start a run without shelling out.

    python src/orchestrator_cli.py            # http://127.0.0.1:8712

The route that matters is `POST /render`. `GET /jobs/{id}` and `GET /health`
exist to serve it — n8n has to be able to find out how a run it started
finished, and a job that takes four minutes cannot always be a blocking
request.

Runs are executed as **subprocesses**, not in-process. The server stays
responsive and answerable while a render is going, a segfault in ffmpeg takes
down one job instead of the orchestrator, and each job's log is captured
separately. n8n never touches media; it calls this and reacts to the result.

Two deliberate safety choices, because this endpoint accepts filesystem paths
and runs an encoder:

  * It binds to 127.0.0.1 by default. Exposing it needs an explicit --host.
  * Outputs are confined to the output root. A job cannot write to an
    arbitrary path just because it asked to.

Set ORCHESTRATOR_TOKEN in .env to require a bearer token. Without one the
server refuses to bind to anything but loopback.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

log = logging.getLogger("orchestrator")

PROJECT_ROOT = config.PROJECT_ROOT
WEB_DIR = PROJECT_ROOT / "web"
DEFAULT_PORT = 8712
# Renders are slow; this is a ceiling, not an expectation.
JOB_TIMEOUT_SECONDS = 3600


# --------------------------------------------------------------------------
# Job tracking
# --------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | running | succeeded | failed
    argv: list[str] = field(default_factory=list)
    out: str = ""
    returncode: int | None = None
    log: str = ""
    error: str = ""
    started_at: str = ""
    finished_at: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "out": self.out,
            "returncode": self.returncode,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            # The tail is enough to see what happened without shipping a
            # megabyte of ffmpeg progress back through n8n.
            "log_tail": "\n".join(self.log.splitlines()[-40:]),
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


JOBS = JobStore()


# --------------------------------------------------------------------------
# Turning a request into a pipeline invocation
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_output(name: str | None, output_root: Path) -> Path:
    """Place a requested output inside the output root, and only there.

    A job supplying `../../../etc/whatever.mp4` gets a filename, not a path
    traversal: only the final component is honoured.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    stem = Path(name).name if name else f"render_{uuid.uuid4().hex[:8]}.mp4"
    if not stem.lower().endswith((".mp4", ".mov", ".mkv")):
        stem = f"{stem}.mp4"
    return (output_root / stem).resolve()


def build_argv(spec: dict[str, Any], output_root: Path) -> tuple[list[str], Path]:
    """Translate a job document into pipeline.py command-line arguments."""
    audio = str(spec.get("audio") or "").strip()
    if not audio:
        raise ValueError("'audio' is required.")

    out_path = resolve_output(spec.get("out"), output_root)
    argv = [
        sys.executable,
        str(SRC_DIR / "pipeline.py"),
        "--audio", audio,
        "--out", str(out_path),
    ]

    # Straight pass-through flags: name in the job, flag on the CLI.
    for key, flag in (
        ("clips", "--clips"),
        ("brief", "--brief"),
        ("title", "--title"),
        ("title_style", "--title-style"),
        ("title_position", "--title-position"),
        ("transition", "--transition"),
        ("generator", "--generator"),
        ("cache_dir", "--cache-dir"),
        ("fit", "--fit"),
        ("look", "--look"),
        ("lut", "--lut"),
        ("chroma_key", "--chroma-key"),
        ("chroma_key_background", "--chroma-key-background"),
        ("ken_burns_direction", "--ken-burns-direction"),
    ):
        value = spec.get(key)
        if value not in (None, ""):
            argv += [flag, str(value)]

    for key, flag in (
        ("duration", "--duration"),
        ("start", "--start"),
        ("seed", "--seed"),
        ("width", "--width"),
        ("height", "--height"),
        ("fps", "--fps"),
        ("start_bpm", "--start-bpm"),
        ("stock_per_query", "--stock-per-query"),
        ("generate", "--generate"),
        ("title_at", "--title-at"),
        ("title_seconds", "--title-seconds"),
        ("max_spend_usd", "--max-spend-usd"),
        ("max_inputs_per_pass", "--max-inputs-per-pass"),
        ("tolerance", "--tolerance"),
        # Phase 4 post effects — see editing/effects.py for what each does.
        ("denoise", "--denoise"),
        ("sharpen", "--sharpen"),
        ("vignette", "--vignette"),
        ("grain", "--grain"),
        ("motion_blur", "--motion-blur"),
        ("light_leak", "--light-leak"),
        ("ken_burns", "--ken-burns"),
        ("ken_burns_steps", "--ken-burns-steps"),
        ("punch_zoom", "--punch-zoom"),
        ("punch_zoom_seconds", "--punch-zoom-seconds"),
        ("punch_zoom_every_nth", "--punch-zoom-every-nth"),
        ("glitch", "--glitch"),
        ("glitch_seconds", "--glitch-seconds"),
        ("glitch_every_nth", "--glitch-every-nth"),
        ("speed_ramp_fraction", "--speed-ramp-fraction"),
        ("speed_ramp_slow", "--speed-ramp-slow"),
        ("speed_ramp_fast", "--speed-ramp-fast"),
    ):
        value = spec.get(key)
        if value is not None:
            argv += [flag, str(value)]

    for key, flag in (
        ("no_local", "--no-local"),
        ("verify", "--verify"),
        ("ken_burns_alternate", "--ken-burns-alternate"),
        ("speed_ramp", "--speed-ramp"),
    ):
        if spec.get(key):
            argv.append(flag)

    return argv, out_path


def run_job(job: Job) -> None:
    """Execute one pipeline run in a subprocess and record the outcome."""
    job.status = "running"
    job.started_at = _now()
    log.info("job %s starting: %s", job.id, " ".join(job.argv[2:]))

    try:
        completed = subprocess.run(
            job.argv,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=JOB_TIMEOUT_SECONDS,
        )
        job.returncode = completed.returncode
        job.log = (completed.stdout or "") + (completed.stderr or "")
        job.status = "succeeded" if completed.returncode == 0 else "failed"
        if job.status == "failed":
            job.error = f"pipeline exited {completed.returncode}"
    except subprocess.TimeoutExpired:
        job.status = "failed"
        job.error = f"timed out after {JOB_TIMEOUT_SECONDS}s"
    except Exception as exc:  # noqa: BLE001 - a job must not kill the server
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"

    job.finished_at = _now()
    log.info("job %s %s", job.id, job.status)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

class RenderRequest(BaseModel):
    """Everything a job can ask for. Only `audio` is required.

    Defined at module level on purpose. Nested inside `create_app` it is a
    local name, and with `from __future__ import annotations` every annotation
    is a string — FastAPI resolves those against module globals, fails to find
    a local class, and silently downgrades the request body to a query
    parameter. The symptom is a 422 for a body that is perfectly valid.
    """

    audio: str
    clips: str | None = None
    brief: str = ""
    out: str | None = None
    title: str = ""
    title_at: float | None = None
    title_seconds: float | None = None
    title_style: str | None = None
    title_position: str | None = None
    duration: float | None = None
    start: float | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    fit: str | None = None
    transition: str | None = None
    start_bpm: float | None = None
    stock_per_query: int | None = None
    generate: int | None = None
    generator: str | None = None
    no_local: bool = False
    verify: bool = False
    tolerance: float | None = None
    cache_dir: str | None = None
    max_spend_usd: float | None = None
    max_inputs_per_pass: int | None = None

    # Phase 4 post effects — see editing/effects.py for what each does. All
    # optional, all off by default; a request that names none of these
    # renders exactly as it always did.
    look: str | None = None
    lut: str | None = None
    denoise: float | None = None
    sharpen: float | None = None
    vignette: float | None = None
    grain: int | None = None
    motion_blur: int | None = None
    light_leak: float | None = None
    chroma_key: str | None = None
    chroma_key_background: str | None = None
    ken_burns: float | None = None
    ken_burns_direction: str | None = None
    ken_burns_alternate: bool = False
    ken_burns_steps: int | None = None
    punch_zoom: float | None = None
    punch_zoom_seconds: float | None = None
    punch_zoom_every_nth: int | None = None
    glitch: int | None = None
    glitch_seconds: float | None = None
    glitch_every_nth: int | None = None
    speed_ramp: bool = False
    speed_ramp_fraction: float | None = None
    speed_ramp_slow: float | None = None
    speed_ramp_fast: float | None = None

    #: Block until the render finishes. False returns a job id immediately.
    wait: bool = Field(default=True)


def create_app(output_root: Path | None = None):
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    root = Path(output_root or PROJECT_ROOT / "output").resolve()

    def authorise(authorization: str | None = Header(default=None)) -> None:
        expected = config.secret("ORCHESTRATOR_TOKEN")
        if not expected:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="bad or missing token")

    app = FastAPI(
        title="video_pipeline orchestrator",
        description="One route to start a render. Called by n8n, or the studio UI.",
        version="1.0",
    )

    # The studio frontend: a static page that calls the routes below. Served
    # from here rather than a separate dev server, so there is exactly one
    # thing to start and no CORS to configure.
    if WEB_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")

        @app.get("/")
        def studio() -> FileResponse:
            return FileResponse(str(WEB_DIR / "index.html"))

        @app.get("/media/{filename}")
        def media(filename: str, _=Depends(authorise)) -> FileResponse:
            # Name-only lookup, exactly like resolve_output: a filename can
            # never walk out of the output root by construction.
            path = (root / Path(filename).name).resolve()
            if root not in path.parents or not path.is_file():
                raise HTTPException(status_code=404, detail="no such render")
            return FileResponse(str(path), media_type="video/mp4")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "output_root": str(root),
            "auth_required": bool(config.secret("ORCHESTRATOR_TOKEN")),
            "jobs": len(JOBS.all()),
        }

    @app.post("/render")
    def render(request: RenderRequest, _=Depends(authorise)) -> dict[str, Any]:
        try:
            argv, out_path = build_argv(request.model_dump(), root)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        job = JOBS.create()
        job.argv = argv
        job.out = str(out_path)

        if request.wait:
            run_job(job)
            return job.public()

        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        return job.public()

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str, _=Depends(authorise)) -> dict[str, Any]:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job.public()

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HTTP trigger for the pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(message)s", force=True
    )
    config.load_env()

    if args.host not in ("127.0.0.1", "localhost", "::1") and not config.secret(
        "ORCHESTRATOR_TOKEN"
    ):
        # This endpoint takes filesystem paths and runs an encoder. Reachable
        # from the network with no token is not a default anyone should get by
        # accident.
        log.error(
            "Refusing to bind %s without ORCHESTRATOR_TOKEN set in .env. "
            "Set a token, or bind 127.0.0.1 and reach it over a tunnel.",
            args.host,
        )
        return 2

    import uvicorn

    log.info("Orchestrator on http://%s:%d  (POST /render)", args.host, args.port)
    uvicorn.run(
        create_app(args.output_root), host=args.host, port=args.port, log_level="info"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
