"""Bearer-token guards: one token for the rig, one for staff dashboard reads."""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Annotated

from fastapi import Header, HTTPException, status


def bearer_guard(token: str, role: str) -> Callable[[str | None], None]:
    """A FastAPI dependency that accepts only ``Authorization: Bearer <token>``.

    An empty configured token refuses every request, so a forgotten env var
    fails closed.
    """

    def guard(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = f"Bearer {token}"
        supplied = authorization or ""
        if not token or not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                f"invalid {role} token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return guard
