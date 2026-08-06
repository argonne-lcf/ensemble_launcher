import time
import asyncio
import uvloop

N = 1_000_000_0

async def noop():
    pass

async def bench():
    start = time.perf_counter()
    for _ in range(N):
        await noop()
    dt = time.perf_counter() - start
    print(f"{N/dt:.0f} awaits/sec  ({dt:.3f}s)")


print("="*50)
print("await dispatch")
print("="*50)
# run with asyncio
print("asyncio")
asyncio.run(bench())

print("uv")
# run with uvloop
uvloop.run(bench())

async def bench_tasks():
    start = time.perf_counter()
    tasks = [asyncio.create_task(noop()) for _ in range(N)]
    await asyncio.gather(*tasks)
    dt = time.perf_counter() - start
    print(f"{N/dt:.0f} tasks/sec  ({dt:.3f}s)")


print("="*50)
print("throughput")
print("="*50)

print("asyncio")
# run with asyncio
asyncio.run(bench_tasks())

print("uv")
# run with uvloop
uvloop.run(bench_tasks())
