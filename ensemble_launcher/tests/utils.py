import time


def echo(task_id: str):
    return f"Hello from task {task_id}"


def echo_stdout(task_id: str):
    print(f"Hello from task {task_id}")


def echo_sleep(task_id: str, sleep_time: float = 0.0):
    time.sleep(sleep_time)
    return f"Hello from task {task_id}"
