"""Day split used by all 2026 experiments, selected with the environment variable PEMS_SPLIT.

interleave (default, earlier runs): months 4-6; every third usable weekday is a test day; no separate tuning days.
chrono: months 1-6; train = January-April, tune = May, test = June (one-shot final test).
"""
import os

MODE = os.environ.get("PEMS_SPLIT", "interleave")


def months():
    return (1, 2, 3, 4, 5, 6) if MODE == "chrono" else (4, 5, 6)


def indices(days):
    """Return (train, tune, test) index lists into the ordered list of usable weekdays."""
    if MODE == "chrono":
        tr = [i for i, d in enumerate(days) if d.month <= 4]
        tu = [i for i, d in enumerate(days) if d.month == 5]
        te = [i for i, d in enumerate(days) if d.month == 6]
        return tr, tu, te
    te = list(range(2, len(days), 3))
    return [i for i in range(len(days)) if i not in te], [], te


def tag():
    return "_chrono" if MODE == "chrono" else ""
