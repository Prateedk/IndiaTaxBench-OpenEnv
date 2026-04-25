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

app = create_app(
    IndiaTaxBenchEnvironment,
    IndiaTaxBenchAction,
    IndiaTaxBenchObservation,
    env_name="india-tax-bench",
    max_concurrent_envs=4,
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
