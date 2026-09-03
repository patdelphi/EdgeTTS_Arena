import os
import signal
import time


def add(a: int, b: int) -> int:
    return a + b


def fail() -> None:
    raise RuntimeError("expected worker failure")


def memory_error() -> None:
    raise MemoryError("expected worker memory error")


def sleep_for(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


def hard_exit(code: int) -> None:
    os._exit(code)


def sigkill_self() -> None:
    os.kill(os.getpid(), signal.SIGKILL)
