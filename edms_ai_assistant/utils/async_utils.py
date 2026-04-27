import asyncio

_background_tasks: set[asyncio.Task] = set()

def spawn_background_task(coro) -> asyncio.Task:
    """Создает фоновую задачу и сохраняет ссылку до её завершения."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task