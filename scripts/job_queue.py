"""Run shell-free python module jobs with a fixed number of parallel slots."""

import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    slots = int(sys.argv[1])
    log_dir = Path(sys.argv[2])
    jobs = [arg.split("|") for arg in sys.argv[3:]]
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, str]] = []
    failures = 0
    while jobs or running:
        while jobs and len(running) < slots:
            args = jobs.pop(0)
            name = "_".join(a.strip("-") for a in args[2:])[:80]
            out = open(log_dir / f"{name}.log", "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-m", *args], stdout=out,
                                    stderr=subprocess.STDOUT)
            running.append((proc, name))
            print(f"{time.strftime('%H:%M:%S')} start {name}", flush=True)
        time.sleep(5)
        for proc, name in list(running):
            if proc.poll() is not None:
                running.remove((proc, name))
                failures += proc.returncode != 0
                print(f"{time.strftime('%H:%M:%S')} done {name} rc={proc.returncode}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
