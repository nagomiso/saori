from __future__ import annotations

import logging
import os
import secrets
import sys
import tempfile
import traceback
from contextlib import contextmanager

from pathlibfs import Path

if sys.version_info < (3, 9):
    from typing import Generator
else:
    from collections.abc import Generator


LOGGER = logging.getLogger(__name__)


def generate_s3_tempdir(prefix: Path, **options) -> Path:
    """Generate a new temporary directory path"""
    while True:
        tempdir = Path(prefix / f"tmp{secrets.token_hex(8)}", **options)
        if not tempdir.exists():
            return tempdir


def _wrap_prefix(prefix: Path | str | None) -> str | Path | None:
    if prefix is None:
        return os.environ.get("SAORI_TMPDIR")
    return prefix


@contextmanager
def mktempdir(
    prefix: Path | str | None = None, **options
) -> Generator[Path, None, None]:
    """Create temporary directories

    With referential transparency to local filesystem or S3

    Args:
        prefix: Prefix for temporary directory path. Defaults to None.
        **options: Extra options for fsspec.

    Yields:
        Generator[Path, None, None]: temporary directory path object
    """
    if tempdir_prefix := _wrap_prefix(prefix):
        tempdir_prefix_path = Path(tempdir_prefix, **options)
        if tempdir_prefix_path.protocol == "file":
            with tempfile.TemporaryDirectory(
                dir=str(tempdir_prefix_path.path)
            ) as tempdir:
                yield Path(tempdir, **options)
        else:
            s3_tempdir = generate_s3_tempdir(tempdir_prefix_path, **options)
            try:
                yield s3_tempdir
            finally:
                try:
                    s3_tempdir.rm(recursive=True)
                except OSError:
                    LOGGER.debug(traceback.format_stack())
    else:
        with tempfile.TemporaryDirectory() as tempdir:
            yield Path(tempdir, **options)
