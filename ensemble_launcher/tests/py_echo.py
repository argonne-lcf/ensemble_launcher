def echo(task_id: str):
    print(f"Hello from task {task_id}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        task_id = f"task-{sys.argv[1]}"
    else:
        task_id = "default"
    
    result = echo(task_id)
    