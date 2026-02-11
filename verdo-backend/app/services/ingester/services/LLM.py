import os
import queue
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from typing import Any, Callable, Dict, Sequence, Union

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# --- Setup --------------------------------------------------------------

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

Message = Dict[str, str]
Messages = Sequence[Message]
Result = Union[str, Exception]

# --- Classes ------------------------------------------------------------

class LLM:
	def __init__(self, maxWorkers: int = 30) -> None:
		# Keys
		self.openaiKey = os.getenv("OPENAIKEY")
		if not self.openaiKey:
			raise RuntimeError("OPENAIKEY environment variable is not set.")

		self.maxWorkers = maxWorkers
		self.taskQueue = queue.Queue()

		self.openaiClient = OpenAI(
			api_key=self.openaiKey,
			http_client=httpx.Client(
				limits=httpx.Limits(
					max_keepalive_connections=self.maxWorkers,
					max_connections=self.maxWorkers,
				),
				timeout=60.0,
			),
		)

		# Simple per-model RPM limiting
		self.rateLock = threading.Lock()
		self.requestHistory = defaultdict(deque)  # model -> deque[timestamps]
		self.maxRpmByModel = {
			"gpt-5.1": 500,
			"gpt-5-mini": 500,
			"gpt-5-nano": 500,
			"gpt-4.1": 500,
			"gpt-4.1-mini": 500,
			"gpt-4.1-nano": 500,
			"o3": 500,
			"o4-mini": 500,
			"gpt-4o": 500,
			"gpt-4o-mini": 500,
			"gpt-4o-realtime-preview": 200,
		}
		self.defaultRpm = 500

		# Persistent thread pool with daemon threads
		self.workers = []
		for _ in range(self.maxWorkers):
			worker = threading.Thread(target=self.workerLoop, daemon=True)
			worker.start()
			self.workers.append(worker)

	# --- Internal Helpers ---------------------------------------------------

	# Block if we're at the per-model RPM limit (simple 60s sliding window)
	def rateLimit(self, model: str) -> None:
		window = 60.0
		now = time.time()
		rpm = self.maxRpmByModel.get(model, self.defaultRpm)

		with self.rateLock:
			hist = self.requestHistory[model]
			# drop entries older than 60s
			while hist and now - hist[0] >= window:
				hist.popleft()

			if len(hist) >= rpm:
				sleepFor = window - (now - hist[0])
			else:
				sleepFor = 0.0

		if sleepFor > 0:
			time.sleep(sleepFor)

		with self.rateLock:
			self.requestHistory[model].append(time.time())

	# Continuously consumes tasks from the queue
	def workerLoop(self) -> None:
		while True:
			func, args, kwargs, future = self.taskQueue.get()
			try:
				result = func(*args, **kwargs)
				future.set_result(result)
			except Exception as e:
				future.set_exception(e)
			finally:
				self.taskQueue.task_done()

	# Submit a callable task for execution by the worker pool
	def submit(self, func: Callable, *args: Any, **kwargs: Any) -> Future:
		future = Future()
		self.taskQueue.put((func, args, kwargs, future))
		return future

	# --- Base Methods -------------------------------------------------------

	def respond(
		self,
		prompt: str,
		model: str = "gpt-5-nano",
		stream: bool = False,
		**options: Any,
	) -> str:
		response = self.chat(
			messages=[{"role": "user", "content": prompt}],
			model=model,
			stream=stream,
			**options,
		)
		return response

	def chat(
		self,
		messages: Messages,
		model: str = "gpt-5-nano",
		stream: bool = False,
		**options: Any,
	) -> str:
		self.rateLimit(model)

		payload = {
			"model": model,
			"messages": list(messages),
			"stream": stream,
		}
		payload.update(options)

		response = self.openaiClient.chat.completions.create(**payload)
		choices = getattr(response, "choices", None) or []
		if not choices:
			raise RuntimeError("Empty response from OpenAI chat completion.")
		return choices[0].message.content

	# --- Parallel Entrypoints -----------------------------------------------

	# Immediately enqueue a respond() call; returns a Future
	def respondAsync(
		self,
		prompt: str,
		model: str = "gpt-5-nano",
		**options: Any,
	) -> Future:
		return self.submit(self.respond, prompt, model, **options)

	# Immediately enqueue a chat() call; returns a Future
	def chatAsync(
		self,
		messages: Messages,
		model: str = "glm-4-flash-250414",
		**options: Any,
	) -> Future:
		return self.submit(self.chat, messages, model, **options)
