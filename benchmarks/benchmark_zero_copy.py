import time

import cloudpickle
import numpy as np

from ensemble_launcher.comm import Message, Result, ResultBatch, IResultBatch, TaskUpdate
from ensemble_launcher.ensemble import Task

NITER = 5
PAYLOAD_SIZE = 1_000_000  # floats per Result/Task (~8MB each)
BATCH_SIZE = 100


def bench(name, serialize_fn, deserialize_fn, niter=NITER):
    ser_times = []
    deser_times = []
    for _ in range(niter):
        tic = time.perf_counter()
        blob = serialize_fn()
        ser_times.append(time.perf_counter() - tic)

        tic = time.perf_counter()
        deserialize_fn(blob)
        deser_times.append(time.perf_counter() - tic)

    ser_med = sorted(ser_times)[niter // 2]
    deser_med = sorted(deser_times)[niter // 2]
    return ser_med, deser_med


def print_comparison(label, ba_ser, ba_deser, cp_ser, cp_deser):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  {'':20s} {'serialize':>12s} {'deserialize':>12s}")
    print(f"  {'to/from_byte_array':20s} {ba_ser*1000:>10.3f}ms {ba_deser*1000:>10.3f}ms")
    print(f"  {'cloudpickle':20s} {cp_ser*1000:>10.3f}ms {cp_deser*1000:>10.3f}ms")
    ser_speedup = cp_ser / ba_ser if ba_ser > 0 else float("inf")
    deser_speedup = cp_deser / ba_deser if ba_deser > 0 else float("inf")
    print(f"  {'speedup':20s} {ser_speedup:>10.1f}x {deser_speedup:>10.1f}x")


# ------------------------------------------------------------------ #
#  Result (single, large payload)
# ------------------------------------------------------------------ #
r = Result(data=np.random.rand(PAYLOAD_SIZE))
r_frames = r.to_byte_array()
r_cp = cloudpickle.dumps(r)

ba_ser, ba_deser = bench(
    "Result",
    lambda: r.to_byte_array(),
    lambda b: Message.from_byte_array(b),
)
cp_ser, cp_deser = bench(
    "Result (cp)",
    lambda: cloudpickle.dumps(r),
    lambda b: cloudpickle.loads(b),
)
print_comparison(f"Result (1x {PAYLOAD_SIZE} floats = ~{PAYLOAD_SIZE*8//1_000_000}MB)", ba_ser, ba_deser, cp_ser, cp_deser)


# ------------------------------------------------------------------ #
#  ResultBatch (100 Results, large payloads)
# ------------------------------------------------------------------ #
rb = ResultBatch(data=[Result(data=np.random.rand(PAYLOAD_SIZE)) for _ in range(BATCH_SIZE)])
rb_frames = rb.to_byte_array()

ba_ser, ba_deser = bench(
    "ResultBatch",
    lambda: rb.to_byte_array(),
    lambda b: ResultBatch.from_byte_array(b),
)
cp_ser, cp_deser = bench(
    "ResultBatch (cp)",
    lambda: cloudpickle.dumps(rb),
    lambda b: cloudpickle.loads(b),
)
print_comparison(f"ResultBatch ({BATCH_SIZE}x Results, ~{BATCH_SIZE*PAYLOAD_SIZE*8//1_000_000}MB total)", ba_ser, ba_deser, cp_ser, cp_deser)


# ------------------------------------------------------------------ #
#  IResultBatch (100 Results, large payloads)
# ------------------------------------------------------------------ #
irb = IResultBatch(data=[Result(data=np.random.rand(PAYLOAD_SIZE)) for _ in range(BATCH_SIZE)])
irb_frames = irb.to_byte_array()

ba_ser, ba_deser = bench(
    "IResultBatch",
    lambda: irb.to_byte_array(),
    lambda b: IResultBatch.from_byte_array(b),
)
cp_ser, cp_deser = bench(
    "IResultBatch (cp)",
    lambda: cloudpickle.dumps(irb),
    lambda b: cloudpickle.loads(b),
)
print_comparison(f"IResultBatch ({BATCH_SIZE}x Results, ~{BATCH_SIZE*PAYLOAD_SIZE*8//1_000_000}MB total)", ba_ser, ba_deser, cp_ser, cp_deser)


# ------------------------------------------------------------------ #
#  TaskUpdate (100 Tasks, large payloads)
# ------------------------------------------------------------------ #
tasks = [
    Task(
        task_id=f"t{i}",
        nnodes=1,
        ppn=1,
        executable=lambda x: x ** 2,
        args=(np.random.rand(PAYLOAD_SIZE),),
    )
    for i in range(BATCH_SIZE)
]
tu = TaskUpdate(added_tasks=tasks, deleted_tasks=[])
tu_frames = tu.to_byte_array()

ba_ser, ba_deser = bench(
    "TaskUpdate",
    lambda: tu.to_byte_array(),
    lambda b: TaskUpdate.from_byte_array(b),
)
cp_ser, cp_deser = bench(
    "TaskUpdate (cp)",
    lambda: cloudpickle.dumps(tu),
    lambda b: cloudpickle.loads(b),
)
print_comparison(f"TaskUpdate ({BATCH_SIZE}x Tasks, ~{BATCH_SIZE*PAYLOAD_SIZE*8//1_000_000}MB total)", ba_ser, ba_deser, cp_ser, cp_deser)
