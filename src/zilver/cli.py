"""CLI entry points."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Heartbeat daemon thread
# ---------------------------------------------------------------------------

def _start_heartbeat(reg_client: "RegistryClient", node_id: str, interval: int = 30) -> None:
    """
    Send periodic heartbeats to the registry from a background daemon thread.

    The thread is marked as a daemon so it is killed automatically when the
    main process exits.  ``interval`` is the sleep duration in seconds between
    heartbeat calls.  Failures are silently swallowed — a transient registry
    outage should not crash the node.
    """
    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                reg_client.heartbeat(node_id)
            except Exception:
                pass  # transient failure — next tick will retry

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# TLS certificate helpers
# ---------------------------------------------------------------------------

_ZILVER_DIR = Path.home() / ".zilver"


def _resolve_tls(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """
    Return (ssl_keyfile, ssl_certfile) to pass to uvicorn.

    If both ``--ssl-key`` and ``--ssl-cert`` are provided, use them directly.
    If neither is provided, auto-generate a self-signed cert into ``~/.zilver/``
    on first run and reuse it on subsequent runs.
    If only one is provided, raise an error.
    """
    key  = getattr(args, "ssl_key",  None)
    cert = getattr(args, "ssl_cert", None)

    if key and cert:
        return key, cert

    if (key is None) != (cert is None):
        sys.exit("Error: --ssl-key and --ssl-cert must be provided together.")

    # Auto-generate if neither is set
    auto_key  = _ZILVER_DIR / "node.key"
    auto_cert = _ZILVER_DIR / "node.crt"

    if not (auto_key.exists() and auto_cert.exists()):
        print("Generating self-signed TLS certificate …", file=sys.stderr)
        try:
            from .security import generate_self_signed_cert
            generate_self_signed_cert(_ZILVER_DIR)
            print(f"Certificate written to {_ZILVER_DIR}/node.{{key,crt}}", file=sys.stderr)
        except ImportError:
            print(
                "Warning: cryptography package not installed; "
                "starting without TLS. Install with: pip install zilver[network]",
                file=sys.stderr,
            )
            return None, None

    return str(auto_key), str(auto_cert)


# ---------------------------------------------------------------------------
# zilver-node commands
# ---------------------------------------------------------------------------

def _cmd_node_start(args: argparse.Namespace) -> None:
    """
    Detect hardware, register with the registry, and serve simulation jobs.

    Flow
    ----
    1. Auto-detect chip, RAM, and qubit ceilings via ``NodeCapabilities.detect()``.
    2. Initialise a ``Node`` with the requested backends.
    3. Resolve TLS certificate (explicit or auto-generated self-signed).
    4. Resolve API key: explicit flag → Keychain → register and store.
    5. Register capabilities and advertised URL with the registry server.
    6. Spawn a daemon thread that sends a heartbeat every 30 s.
    7. Register a SIGINT/SIGTERM handler that deregisters the node cleanly.
    8. Start uvicorn — blocks until the process is killed.
    """
    from .node import Node
    from .server import serve
    from .client import RegistryClient

    backends = [b.strip() for b in args.backends.split(",")]
    node = Node.start(backends=backends, node_id=args.node_id, wallet=args.wallet)

    print(f"Node {node.caps.node_id[:8]} | chip: {node.caps.chip} | "
          f"RAM: {node.caps.ram_gb}GB | backends: {node.caps.backends} | "
          f"sv_max: {node.caps.sv_qubits_max}q")

    # --- TLS ----------------------------------------------------------------
    ssl_key, ssl_cert = _resolve_tls(args)
    scheme = "https" if ssl_cert else "http"

    # Construct the URL this node will advertise to the registry
    advertised_host = args.host if args.host != "0.0.0.0" else _local_ip()
    node_url = f"{scheme}://{advertised_host}:{args.port}"

    # --- API key ------------------------------------------------------------
    api_key: str | None = getattr(args, "api_key", None)

    reg_client: RegistryClient | None = None

    if args.registry:
        reg_client = RegistryClient(args.registry)
        try:
            if api_key is None:
                # Try Keychain first
                try:
                    from .security import keychain_get
                    api_key = keychain_get("zilver", node.caps.node_id)
                except Exception:
                    pass

            if api_key is None:
                # Register and receive a new key
                reg_client.register(node.caps, node_url)
                api_key = reg_client.last_api_key
                if api_key:
                    try:
                        from .security import keychain_store
                        keychain_store("zilver", node.caps.node_id, api_key)
                        print("API key stored in macOS Keychain.")
                    except Exception as exc:
                        print(f"Warning: could not store API key in Keychain: {exc}",
                              file=sys.stderr)
            else:
                reg_client.register(node.caps, node_url)

            print(f"Registered with registry at {args.registry}")
            _start_heartbeat(reg_client, node.caps.node_id)
        except Exception as exc:
            print(f"Warning: could not register with registry: {exc}", file=sys.stderr)

    def _deregister(sig: int, frame: object) -> None:
        if reg_client is not None:
            try:
                reg_client.deregister(node.caps.node_id)
                print("\nDeregistered from registry.")
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _deregister)
    signal.signal(signal.SIGTERM, _deregister)

    proto = "HTTPS" if ssl_cert else "HTTP (no TLS)"
    print(f"Serving {proto} on {args.host}:{args.port}  (Ctrl-C to stop)")
    if ssl_cert and not getattr(args, "ssl_cert", None):
        print("Warning: using self-signed certificate — clients need --no-verify or verify=False",
              file=sys.stderr)

    serve(
        node,
        host=args.host,
        port=args.port,
        log_level="warning",
        api_key=api_key,
        ssl_keyfile=ssl_key,
        ssl_certfile=ssl_cert,
    )


def _cmd_node_status(args: argparse.Namespace) -> None:
    """Print a summary of the registry to stdout."""
    from .client import RegistryClient
    reg = RegistryClient(args.registry)
    s = reg.summary()
    print(f"Registry: {args.registry}")
    print(f"  Online nodes  : {s.get('online', 0)}")
    print(f"  Registered    : {s.get('total_registered', 0)}")
    print(f"  Backends      : {', '.join(s.get('backends', []))}")
    print(f"  Max SV qubits : {s.get('max_sv_qubits', 0)}")
    print(f"  Max DM qubits : {s.get('max_dm_qubits', 0)}")
    print(f"  Total stake   : {s.get('total_stake', 0)}")


def _cmd_node_list(args: argparse.Namespace) -> None:
    """List all online nodes in the registry."""
    from .client import RegistryClient
    reg = RegistryClient(args.registry)
    nodes = reg.nodes()
    if not nodes:
        print("No online nodes.")
        return
    print(f"{'NODE ID':36}  {'CHIP':20}  {'BACKENDS':12}  {'SV MAX':7}  URL")
    print("-" * 100)
    for n in nodes:
        nid      = n.get("node_id", "")[:36]
        chip     = n.get("chip", "")[:20]
        backends = ",".join(n.get("backends", []))
        sv_max   = n.get("sv_qubits_max", 0)
        url      = n.get("url", "")
        print(f"{nid:36}  {chip:20}  {backends:12}  {sv_max:7}  {url}")


# ---------------------------------------------------------------------------
# zilver-registry commands
# ---------------------------------------------------------------------------

def _cmd_registry_start(args: argparse.Namespace) -> None:
    """Start an in-memory capability registry server."""
    from .registry_server import serve_registry

    admin_key = getattr(args, "admin_key", None) or os.environ.get("ZILVER_REGISTRY_KEY")

    ssl_key, ssl_cert = _resolve_tls(args)

    if admin_key:
        print("Registry admin key is set — deregistration requires Authorization header.")
    else:
        print("Warning: no --admin-key set; deregistration endpoint is unprotected.",
              file=sys.stderr)

    proto = "HTTPS" if ssl_cert else "HTTP (no TLS)"
    print(f"Registry server {proto} on {args.host}:{args.port}  (Ctrl-C to stop)")

    serve_registry(
        host=args.host,
        port=args.port,
        admin_key=admin_key,
        rate_limit=True,
        ssl_keyfile=ssl_key,
        ssl_certfile=ssl_cert,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_ip() -> str:
    """
    Best-effort detection of the machine's LAN IP address.

    Falls back to ``"127.0.0.1"`` if detection fails (e.g. no network).
    Used to build the node URL advertised to the registry so that other
    machines can reach this node directly.
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def _build_node_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zilver-node",
        description="Zilver simulation node daemon.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- start --------------------------------------------------------------
    p_start = sub.add_parser("start", help="Start the node daemon.")
    p_start.add_argument(
        "--backends", default="sv",
        help="Comma-separated list of backends to enable: sv,dm,tn (default: sv).",
    )
    p_start.add_argument(
        "--port", type=int, default=7700,
        help="TCP port to listen on (default: 7700).",
    )
    p_start.add_argument(
        "--host", default="0.0.0.0",
        help="Interface to bind (default: 0.0.0.0).",
    )
    p_start.add_argument(
        "--registry", default=None,
        help="Registry server URL, e.g. https://host:7701. "
             "Omit to run standalone without registry registration.",
    )
    p_start.add_argument(
        "--node-id", dest="node_id", default=None,
        help="Explicit node identifier (auto-detected from IOPlatformUUID if omitted).",
    )
    p_start.add_argument(
        "--wallet", default=None,
        help="Wallet address for future reward settlement (stored, not yet used).",
    )
    p_start.add_argument(
        "--ssl-cert", dest="ssl_cert", default=None,
        help="Path to TLS certificate (PEM). Auto-generates self-signed if omitted.",
    )
    p_start.add_argument(
        "--ssl-key", dest="ssl_key", default=None,
        help="Path to TLS private key (PEM). Must be paired with --ssl-cert.",
    )
    p_start.add_argument(
        "--api-key", dest="api_key", default=None,
        help="API key issued by the registry. "
             "If omitted, reads from Keychain or registers automatically.",
    )

    # --- status -------------------------------------------------------------
    p_status = sub.add_parser("status", help="Print registry summary.")
    p_status.add_argument(
        "--registry", required=True,
        help="Registry server URL.",
    )

    # --- nodes --------------------------------------------------------------
    p_nodes = sub.add_parser("nodes", help="List online nodes in the registry.")
    p_nodes.add_argument(
        "--registry", required=True,
        help="Registry server URL.",
    )

    return parser


def _build_registry_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zilver-registry",
        description="Zilver capability registry server.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start the registry server.")
    p_start.add_argument(
        "--host", default="0.0.0.0",
        help="Interface to bind (default: 0.0.0.0).",
    )
    p_start.add_argument(
        "--port", type=int, default=7701,
        help="TCP port to listen on (default: 7701).",
    )
    p_start.add_argument(
        "--ssl-cert", dest="ssl_cert", default=None,
        help="Path to TLS certificate (PEM). Auto-generates self-signed if omitted.",
    )
    p_start.add_argument(
        "--ssl-key", dest="ssl_key", default=None,
        help="Path to TLS private key (PEM). Must be paired with --ssl-cert.",
    )
    p_start.add_argument(
        "--admin-key", dest="admin_key", default=None,
        help="Bearer token required to deregister nodes. "
             "Also read from ZILVER_REGISTRY_KEY env var.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry points registered in pyproject.toml
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``zilver-node`` script."""
    parser = _build_node_parser()
    args = parser.parse_args()

    dispatch = {
        "start":  _cmd_node_start,
        "status": _cmd_node_status,
        "nodes":  _cmd_node_list,
    }
    dispatch[args.command](args)


def main_registry() -> None:
    """Entry point for the ``zilver-registry`` script."""
    parser = _build_registry_parser()
    args = parser.parse_args()

    dispatch = {
        "start": _cmd_registry_start,
    }
    dispatch[args.command](args)
