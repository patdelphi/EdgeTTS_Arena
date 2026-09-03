import time


def add(a: int, b: int) -> int:
    return a + b


def fail() -> None:
    raise RuntimeError("expected worker failure")


def sleep_for(seconds: float) -> str:
    time.sleep(seconds)
    return "done"
