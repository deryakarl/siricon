"""Registry HTTP server."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request

from .node import NodeCapabilities
from .registry import Registry
from .security import generate_api_key


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def _make_rate_limiter(max_calls: int, window_secs: int = 60) -> Callable:
    """
    Return a FastAPI dependency that enforces a per-IP sliding-window rate limit.

    When *max_calls* is 0 the returned callable is a no-op (disabled).
    """
    if max_calls == 0:
        async def _noop(request: Request) -> None:
            return
        return _noop

    hits: dict[str, list[float]] = defaultdict(list)

    async def _limit(request: Request) -> None:
        ip = request.client.host if request.client else "local"
        now = time.monotonic()
        cutoff = now - window_secs
        valid = [t for t in hits[ip] if t > cutoff]
        if len(valid) >= max_calls:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
            )
        valid.append(now)
        hits[ip] = valid

    return _limit


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def make_registry_app(
    registry:   Registry | None = None,
    admin_key:  str | None = None,
    rate_limit: bool = False,
) -> FastAPI:
    """
    Build the FastAPI application for the capability registry.

    Parameters
    ----------
    registry:
        An existing :class:`~zilver.registry.Registry` instance to wrap.
        If ``None``, a fresh in-memory registry is created.  Passing an
        existing instance is useful in tests where the caller needs direct
        access to registry state.
    admin_key:
        Bearer token required to deregister nodes via
        ``DELETE /nodes/{node_id}``.  When ``None`` (the default) the
        endpoint is unprotected — suitable for local development and tests.
    rate_limit:
        When ``True``, apply per-IP sliding-window rate limits:
        ``POST /nodes`` → 5/min, ``GET /match`` → 60/min.
        Default ``False`` for test and dev compatibility.

    Returns
    -------
    FastAPI
        The application instance.  Use ``uvicorn.run`` for production or
        ``fastapi.testclient.TestClient`` for tests.
    """
    reg = registry if registry is not None else Registry()

    # Maps node_id → advertised URL so clients can connect directly.
    node_urls: dict[str, str] = {}
    # Maps node_id → issued API key for identity verification.
    node_keys: dict[str, str] = {}

    app = FastAPI(title="zilver-registry", version="0.1.0")

    # --- Rate limiters (per-endpoint) ----------------------------------------

    _register_limit = _make_rate_limiter(5  if rate_limit else 0)
    _match_limit    = _make_rate_limiter(60 if rate_limit else 0)

    # --- Auth dependencies ---------------------------------------------------

    async def _require_admin(request: Request) -> None:
        """Verify admin Bearer token.  No-op when admin_key is None."""
        if admin_key is None:
            return
        header = request.headers.get("Authorization", "")
        if not (header.startswith("Bearer ") and header[7:] == admin_key):
            raise HTTPException(status_code=401, detail="Invalid or missing admin key")

    # --- Registration -------------------------------------------------------

    @app.post("/nodes", status_code=201, dependencies=[Depends(_register_limit)])
    async def register(body: dict[str, Any]) -> dict[str, Any]:
        """
        Register or re-register a node.

        Request body
        ~~~~~~~~~~~~
        A JSON object with two fields:

        - ``caps`` — ``NodeCapabilities.to_dict()``
        - ``url``  — the node's reachable HTTP base URL,
          e.g. ``"http://192.168.1.5:7700"``

        Response
        ~~~~~~~~
        ``{"registered": true, "node_id": "<id>", "api_key": "<key>"}``

        The ``api_key`` is a 32-byte random hex string issued by the registry.
        The node should store it securely (e.g. macOS Keychain) and present it
        in ``Authorization: Bearer <key>`` on subsequent requests.

        Idempotent: re-registering an existing node refreshes its capabilities
        and last-seen timestamp, and returns a new API key.
        """
        try:
            caps = NodeCapabilities(**body["caps"])
            url: str = body["url"]
        except (KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid registration body: {exc}")

        reg.register(caps)
        node_urls[caps.node_id] = url

        key = generate_api_key()
        node_keys[caps.node_id] = key

        return {"registered": True, "node_id": caps.node_id, "api_key": key}

    @app.delete("/nodes/{node_id}", dependencies=[Depends(_require_admin)])
    async def deregister(node_id: str) -> dict[str, Any]:
        """
        Mark a node offline.

        Requires ``Authorization: Bearer <admin_key>`` when the registry was
        started with ``--admin-key`` / ``ZILVER_REGISTRY_KEY``.

        The node remains in the registry's history (for diagnostics) but
        will no longer be returned by ``/match`` or ``/nodes``.

        Returns ``{"deregistered": true}`` if found, ``{"deregistered": false}``
        if the node was not registered.
        """
        found = reg.deregister(node_id)
        node_urls.pop(node_id, None)
        node_keys.pop(node_id, None)
        return {"deregistered": found, "node_id": node_id}

    @app.post("/nodes/{node_id}/heartbeat")
    async def heartbeat(node_id: str) -> dict[str, Any]:
        """
        Refresh the last-seen timestamp for a node.

        Called by the node daemon on a fixed interval (default 30 s) so the
        registry can detect stale nodes.  Returns 404 if the node is unknown.
        """
        found = reg.heartbeat(node_id)
        if not found:
            raise HTTPException(status_code=404, detail=f"Node {node_id!r} not found")
        return {"status": "ok", "node_id": node_id}

    # --- Discovery ----------------------------------------------------------

    @app.get("/nodes")
    async def list_nodes() -> list[dict[str, Any]]:
        """
        Return all currently online nodes.

        Each element is ``NodeCapabilities.to_dict()`` extended with a
        ``"url"`` field containing the node's reachable HTTP address.
        """
        entries = reg.all_entries()
        result = []
        for entry in entries:
            d = entry.caps.to_dict()
            d["url"] = node_urls.get(entry.caps.node_id, "")
            result.append(d)
        return result

    @app.get("/match", dependencies=[Depends(_match_limit)])
    async def match(
        backend:   str,
        n_qubits:  int,
        min_stake: int = 0,
    ) -> dict[str, Any]:
        """
        Find the best available node for a job.

        Query parameters
        ~~~~~~~~~~~~~~~~
        - ``backend``   — ``"sv"``, ``"dm"``, or ``"tn"``
        - ``n_qubits``  — qubit count required by the job
        - ``min_stake`` — minimum stake (default 0)

        Response
        ~~~~~~~~
        On success: ``NodeCapabilities.to_dict()`` plus ``"url"``.

        Raises **404** if no eligible node exists.
        """
        entry = reg.match(backend, n_qubits, min_stake=min_stake)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No eligible node for backend={backend!r} "
                    f"n_qubits={n_qubits} min_stake={min_stake}"
                ),
            )
        d = entry.caps.to_dict()
        d["url"] = node_urls.get(entry.caps.node_id, "")
        return d

    @app.get("/summary")
    async def summary() -> dict[str, Any]:
        """
        Aggregate registry statistics.

        Returns a dict with ``online``, ``total_registered``, ``backends``,
        ``max_sv_qubits``, ``max_dm_qubits``, and ``total_stake``.
        Useful for monitoring dashboards and CLI status commands.
        """
        return reg.summary()

    return app


# ---------------------------------------------------------------------------
# Blocking server entrypoint
# ---------------------------------------------------------------------------

def serve_registry(
    registry:     Registry | None = None,
    host:         str = "0.0.0.0",
    port:         int = 7701,
    log_level:    str = "warning",
    admin_key:    str | None = None,
    rate_limit:   bool = False,
    ssl_keyfile:  str | None = None,
    ssl_certfile: str | None = None,
) -> None:
    """
    Start a uvicorn HTTP(S) server for the capability registry and block until
    interrupted.

    This is called by the CLI (``zilver-registry start``).  For tests, use
    ``make_registry_app`` with ``fastapi.testclient.TestClient`` instead.

    Parameters
    ----------
    registry:
        Registry instance to expose.  A fresh one is created if ``None``.
    host:
        Interface to bind.  Default ``"0.0.0.0"`` (all interfaces).
    port:
        TCP port number.  Default 7701 (separate from node default 7700).
    log_level:
        uvicorn log level.  Default ``"warning"``.
    admin_key:
        Bearer token required for node deregistration.  ``None`` disables.
    rate_limit:
        Enable per-IP rate limiting on ``POST /nodes`` and ``GET /match``.
    ssl_keyfile:
        Path to the TLS private key (PEM).  When set, the server uses HTTPS.
    ssl_certfile:
        Path to the TLS certificate (PEM).
    """
    import uvicorn
    app = make_registry_app(registry, admin_key=admin_key, rate_limit=rate_limit)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )
