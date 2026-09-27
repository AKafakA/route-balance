"""bench.arrivals reproduces every arrival-schedule digest recorded by the v3 burst campaign."""
import hashlib

from bench.arrivals import arrival_times, schedule_digest
from tests.traces import constant_trace

N_PROMPTS = 3534
BURSTINESS = {"pois": 1.0, "gam025": 0.25}


def _records(oracle):
    text = (oracle.cells_dir("s2_q1burst_PROD") / "ARRIVAL_DIGESTS.txt").read_text()
    return [line.split() for line in text.split("\n") if line.strip()]


def test_all_73_banked_digests_are_reproduced(oracle):
    records = _records(oracle)
    assert len(records) == 73 and len({tuple(r[:3]) for r in records}) == 36
    wrong = [r for r in records
             if schedule_digest(arrival_times(constant_trace(N_PROMPTS, float(r[1]), BURSTINESS[r[2]]),
                                              int(r[0]))) != r[3]]
    assert wrong == [], f"{len(wrong)}/73 digests differ: {wrong[:5]}"
    print(f"arrival digests reproduced: {len(records) - len(wrong)}/{len(records)}")


def test_a_digest_of_the_raw_times_would_not_match(oracle):
    seed, rate, pattern, digest = _records(oracle)[0]
    times = arrival_times(constant_trace(N_PROMPTS, float(rate), BURSTINESS[pattern]), int(seed))
    raw = hashlib.sha256("".join(f"{t:.9f}|" for t in times).encode()).hexdigest()[:16]
    assert raw != digest == schedule_digest(times)
