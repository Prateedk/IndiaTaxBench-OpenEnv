"""FastAPI application for IndiaTaxBench environment."""

from fastapi.responses import RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import IndiaTaxBenchAction, IndiaTaxBenchObservation
    from .india_tax_bench_environment import IndiaTaxBenchEnvironment
except (ImportError, SystemError):
    from models import IndiaTaxBenchAction, IndiaTaxBenchObservation
    from server.india_tax_bench_environment import IndiaTaxBenchEnvironment


# OpenEnv's REST `/reset` and `/step` handlers call `env_factory()` and `env.close()`
# on every request, which destroys per-episode state (submitted_predictions, step_count,
# task selection). We expose a process-wide singleton so multi-step episodes work
# over plain REST. Concurrent multi-session use should switch to the MCP endpoint.
_singleton_env: IndiaTaxBenchEnvironment | None = None


def _singleton_factory() -> IndiaTaxBenchEnvironment:
    global _singleton_env
    if _singleton_env is None:
        _singleton_env = IndiaTaxBenchEnvironment()
        # `close()` is invoked after each REST request; keep the singleton alive.
        _singleton_env.close = lambda: None  # type: ignore[method-assign]
    return _singleton_env


# Mark stateful so create_app forces max_concurrent_envs=1 (single shared instance).
IndiaTaxBenchEnvironment.SUPPORTS_CONCURRENT_SESSIONS = False  # type: ignore[attr-defined]

app = create_app(
    _singleton_factory,
    IndiaTaxBenchAction,
    IndiaTaxBenchObservation,
    env_name="india-tax-bench",
    max_concurrent_envs=1,
)

if not any(getattr(r, "path", None) == "/" for r in app.routes):

    @app.get("/", include_in_schema=False)
    async def _root_redirect_to_docs():
        return RedirectResponse(url="/docs")


from starlette.requests import Request


@app.middleware("http")
async def _no_store_api_responses(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(
        ("/reset", "/step", "/state", "/schema", "/metadata", "/health")
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
