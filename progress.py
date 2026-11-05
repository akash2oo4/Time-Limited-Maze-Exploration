"""
progress.py -- a one-line progress bar for long parallel runs.

    rows = run_with_progress(pool, func, jobs)   # same result and order as pool.map

Prints, refreshing in place:
    [##########----------]  852/1704 runs   50%   elapsed 6m10s   left ~6m10s
"""

import sys
import time


def _fmt(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def run_with_progress(pool, func, jobs, chunksize=2, width=20):
    n = len(jobs)
    t0 = time.time()
    out = []
    for k, r in enumerate(pool.imap(func, jobs, chunksize=chunksize), 1):
        out.append(r)
        el = time.time() - t0
        left = el / k * (n - k)
        done = int(width * k / n)
        sys.stdout.write(f"\r  [{'#' * done}{'-' * (width - done)}] {k:>5}/{n} runs "
                         f"{100 * k / n:4.0f}%   elapsed {_fmt(el)}   left ~{_fmt(left)} ")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return out
