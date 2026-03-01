"""Security helpers: API keys, TLS certificates, macOS Keychain."""

from __future__ import annotations

import secrets
import subprocess
from pathlib import Path


def generate_api_key() -> str:
    """Return a 32-byte random hex string suitable for use as an API key."""
    return secrets.token_hex(32)


def generate_self_signed_cert(path: Path) -> tuple[Path, Path]:
    """
    Generate a self-signed TLS certificate and private key.

    Writes ``node.key`` (RSA-2048 private key, PEM) and ``node.crt``
    (self-signed X.509 cert, PEM) into *path*, creating the directory
    if it does not exist.  The certificate is valid for 365 days.

    Parameters
    ----------
    path:
        Directory to write the key and cert files into.

    Returns
    -------
    tuple[Path, Path]
        ``(key_path, cert_path)``
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import datetime

    path.mkdir(parents=True, exist_ok=True)
    key_path = path / "node.key"
    crt_path = path / "node.crt"

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "zilver-node"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
        )
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    crt_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    return key_path, crt_path


def keychain_store(service: str, account: str, password: str) -> None:
    """
    Store a password in the macOS Keychain.

    Uses the ``security add-generic-password -U`` command, which updates
    the entry if it already exists.

    Parameters
    ----------
    service:
        Keychain service name (e.g. ``"zilver"``).
    account:
        Keychain account name (e.g. node_id).
    password:
        Secret value to store.

    Raises
    ------
    subprocess.CalledProcessError
        If the ``security`` command fails (macOS only).
    """
    subprocess.run(
        [
            "security", "add-generic-password",
            "-U",
            "-s", service,
            "-a", account,
            "-w", password,
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )


def keychain_get(service: str, account: str) -> str | None:
    """
    Retrieve a password from the macOS Keychain.

    Parameters
    ----------
    service:
        Keychain service name.
    account:
        Keychain account name.

    Returns
    -------
    str | None
        The stored password, or ``None`` if not found.
    """
    try:
        out = subprocess.check_output(
            [
                "security", "find-generic-password",
                "-s", service,
                "-a", account,
                "-w",
            ],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except subprocess.CalledProcessError:
        return None
