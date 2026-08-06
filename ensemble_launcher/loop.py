try:
    from uvloop import run
except ImportError:
    from asyncio import run
