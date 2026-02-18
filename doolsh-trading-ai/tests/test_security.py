"""Tests for JWT and password utilities."""

from __future__ import annotations

import pytest

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_password,
    verify_password,
)


def test_password_hash_verify():
    plain = "Str0ng!P@ssword"
    hashed = hash_password(plain)
    assert hashed != plain
    assert verify_password(plain, hashed)
    assert not verify_password("wrong", hashed)


def test_access_token_roundtrip():
    token = create_access_token("42", extra={"role": "admin"})
    payload = decode_token(token)
    assert payload["sub"] == "42"
    assert payload["role"] == "admin"
    assert payload["type"] == "access"


def test_refresh_token_roundtrip():
    token = create_refresh_token("7")
    payload = decode_token(token)
    assert payload["sub"] == "7"
    assert payload["type"] == "refresh"


def test_api_key_format():
    key = generate_api_key()
    assert key.startswith("doolsh_")
    assert len(key) > 20
