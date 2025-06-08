# tts_manager.py
import asyncio
from typing import List, Set
import logging

logger = logging.getLogger(__name__)

class TTSManager:
    def __init__(self, servers: list):
        self.servers = servers
        self.lock = asyncio.Lock()
        self.available = asyncio.Queue()
        self.allocations = {}  # {task_id: server}

        for s in servers:
            self.available.put_nowait(s)

    async def acquire(self, task_id: int) -> int:
        async with self.lock:
            server = await self.available.get()
            self.allocations[task_id] = server
            return server

    async def release(self, task_id: int):
        async with self.lock:
            server = self.allocations.pop(task_id, None)
            if server:
                await self.available.put(server)