"""What a caller sees when something fails unexpectedly.

An unexpected exception's own text can name internal paths, hosts, settings
or the content being processed, so it never goes back to the caller or into
a record the caller reads. The route logs the exception with
``logger.exception`` and sends :func:`public_error` instead.

An error written for the person on purpose is raised as :class:`PublicError`
(or a subclass); :func:`public_error` passes its text through. An
``HTTPException`` detail is written on purpose too and never reaches here.
"""

from __future__ import annotations

GENERIC_FAILURE = "Something went wrong. Try again."


class PublicError(RuntimeError):
    """An error whose text is written for the person and is safe to show."""


def public_error(exc: BaseException) -> str:
    """The text to show a caller for ``exc``.

    A :class:`PublicError` keeps its own text; anything else gets the generic
    line, and its details belong in the log.
    """
    if isinstance(exc, PublicError) and str(exc).strip():
        return str(exc)
    return GENERIC_FAILURE
