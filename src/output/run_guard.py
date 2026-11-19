"""Shared local-output safeguards for CLI and browser jobs."""

from contextlib import contextmanager
import fcntl
import shutil


class OutputSpaceError(OSError):
    pass


def check_space(directory, minimum_mib):
    if shutil.disk_usage(directory).free < minimum_mib * 1024**2:
        raise OutputSpaceError(f"Output filesystem has less than the configured {minimum_mib:,} MiB of free space.")


@contextmanager
def output_run(directory, minimum_mib):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / ".anpr-run.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OSError("Another ANPR job is using this output directory.") from exc
        check_space(directory, minimum_mib)
        yield
