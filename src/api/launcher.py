"""Official Launcher and Deployment Configuration for PaladinMedia API.

Enforces:
- Single-worker execution contract for in-memory session store.
- Strict deployment mode validation ('local' or 'remote').
- Loopback binding invariant for local development mode.
- Controlled startup configuration validation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import uvicorn

from src.api.security import APISettings, reset_api_settings
from src.api.server import create_app

logger = logging.getLogger("paladinmedia.launcher")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate_launcher_config(
    host: str,
    port: int,
    mode: str,
    workers: int = 1,
    api_token: Optional[str] = None,
) -> APISettings:
    """Strictly validates operational launcher configuration.

    Args:
        host: Host address to bind.
        port: Port number.
        mode: Deployment mode ("local" or "remote").
        workers: Worker process count. Must be exactly 1.
        api_token: Optional bearer token for authentication.

    Returns:
        APISettings: Initialized and validated settings instance.

    Raises:
        ValueError: If workers != 1, invalid mode, or non-loopback binding in local mode.
        RuntimeError: If remote mode invariants are violated.
    """
    if workers != 1:
        raise ValueError(
            f"INVALID_WORKER_CONFIG: PaladinMedia qualified deployment requires single-worker "
            f"execution (workers=1) due to in-memory session storage. Found workers={workers}. "
            "Multi-worker deployments require session affinity or external shared storage."
        )

    norm_mode = mode.lower()
    if norm_mode in ("production", "prod"):
        norm_mode = "remote"

    if norm_mode not in ("local", "remote"):
        raise ValueError(
            f"CONFIGURATION_ERROR: Invalid deployment mode '{mode}'. "
            "Supported modes are 'local' or 'remote'."
        )

    if norm_mode == "local" and host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"BINDING_VIOLATION: Local deployment mode is restricted to loopback addresses "
            f"({', '.join(sorted(LOOPBACK_HOSTS))}). Configured host '{host}' exposes an unauthenticated "
            "service on non-loopback interfaces. For remote access, configure PALADINMEDIA_MODE=remote "
            "and set a secure PALADINMEDIA_API_TOKEN."
        )

    # Initialize settings and validate startup
    settings = APISettings(mode=norm_mode, api_token=api_token)
    settings.validate_startup()
    return settings


def run_launcher(
    host: Optional[str] = None,
    port: Optional[int] = None,
    mode: Optional[str] = None,
    workers: int = 1,
    api_token: Optional[str] = None,
) -> None:
    """Launches the Uvicorn application with validated deployment configuration."""
    effective_host = host or os.environ.get("PALADINMEDIA_HOST", "127.0.0.1")
    effective_port = port or int(os.environ.get("PALADINMEDIA_PORT", "8000"))
    effective_mode = mode or os.environ.get("PALADINMEDIA_MODE", "local")
    effective_token = api_token or os.environ.get("PALADINMEDIA_API_TOKEN")

    settings = validate_launcher_config(
        host=effective_host,
        port=effective_port,
        mode=effective_mode,
        workers=workers,
        api_token=effective_token,
    )

    reset_api_settings(settings)
    app = create_app(settings)

    logger.info(
        "Starting PaladinMedia Qualified Deployment: host=%s, port=%d, mode=%s, workers=%d",
        effective_host,
        effective_port,
        settings.mode,
        workers,
    )

    uvicorn.run(app, host=effective_host, port=effective_port, workers=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PaladinMedia Qualified Deployment Launcher",
        epilog=(
            "Note: Authentication tokens must be supplied via the PALADINMEDIA_API_TOKEN "
            "environment variable or secrets manager. Command-line token passing is prohibited "
            "to prevent token exposure in process tables and shell history."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("PALADINMEDIA_HOST", "127.0.0.1"),
        help="Bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PALADINMEDIA_PORT", "8000")),
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("PALADINMEDIA_MODE", "local"),
        help="Deployment mode: 'local' (default) or 'remote'",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process workers (must be 1 for qualified in-memory session store)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_launcher(
            host=args.host,
            port=args.port,
            mode=args.mode,
            workers=args.workers,
        )
    except Exception as e:
        logger.error("Launch aborted: %s", e)
        sys.exit(1)
