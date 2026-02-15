"""Simple worker process entrypoint."""

import signal
import time


_running = True


def _shutdown_handler(signum: int, _frame: object) -> None:
    global _running
    _running = False
    print(f"Worker received signal {signum}, shutting down.")


def main() -> None:
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    print("Worker started. Waiting for jobs...")
    while _running:
        time.sleep(5)

    print("Worker stopped.")


if __name__ == "__main__":
    main()
