#!/usr/bin/env python3
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 5555

# Upgraded complex solutions to return for the new coding tasks
SOLUTIONS = {
    # Task 1: API Integration
    "apiclient": """```python
import time
import requests

class APIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")

class APIClient:
    def __init__(self, base_url: str, api_key: str, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.max_retries = max_retries

    def fetch_page(self, endpoint: str, page: int) -> dict:
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"page": page}
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        wait_sec = float(retry_after)
                    else:
                        wait_sec = float(2 ** attempt)
                    time.sleep(wait_sec)
                else:
                    raise APIError(response.status_code, f"Request failed with status {response.status_code}")
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise APIError(500, str(e))
                time.sleep(2 ** attempt)
        raise APIError(429, "Max retries exceeded on 429")

    def fetch_all_records(self, endpoint: str) -> list[dict]:
        all_records = []
        page = 1
        while True:
            data = self.fetch_page(endpoint, page)
            records = data.get("records", [])
            all_records.extend(records)
            if not data.get("has_more", False):
                break
            page += 1
        return all_records
```""",

    # Task 2: Data Refactoring
    "metricsprocessor": """```python
import json
import math

class MetricsProcessor:
    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self.window = []
        self.sum = 0.0
        self.sq_sum = 0.0

    def add_log(self, log_line: str) -> dict:
        try:
            data = json.loads(log_line)
            ts = data["timestamp"]
            lat = data["latency"]
        except Exception:
            return None
            
        self.window.append({"timestamp": ts, "latency": lat})
        self.sum += lat
        self.sq_sum += lat * lat
        
        limit = ts - self.window_seconds
        while self.window and self.window[0]["timestamp"] <= limit:
            old_item = self.window.pop(0)
            old_lat = old_item["latency"]
            self.sum -= old_lat
            self.sq_sum -= old_lat * old_lat
            self.sum = max(0.0, self.sum)
            self.sq_sum = max(0.0, self.sq_sum)

        count = len(self.window)
        if count == 0:
            return {"count": 0, "avg": 0.0, "std": 0.0, "anomalies": 0}
            
        avg = self.sum / count
        
        if count >= 2:
            var = (self.sq_sum - (self.sum * self.sum) / count) / (count - 1)
            var = max(0.0, var)
            std = math.sqrt(var)
        else:
            std = 0.0

        anomalies = 0
        if std > 0.0:
            threshold = 2.0 * std
            for item in self.window:
                if abs(item["latency"] - avg) > threshold:
                    anomalies += 1

        return {
            "count": count,
            "avg": round(avg, 3),
            "std": round(std, 3),
            "anomalies": anomalies
        }
```""",

    # Task 3: Concurrency Race / Transaction Manager
    "transactionmanager": """```python
class TransactionManager:
    def __init__(self, initial_balance: float, overdraft_limit: float = 0.0):
        self.balance = initial_balance
        self.overdraft_limit = overdraft_limit
        self.pending_transactions = []
        self.transaction_history = []

    def add_transaction(self, transaction_id: str, amount: float):
        if amount == 0.0:
            raise ValueError("Transaction amount cannot be zero")
        self.pending_transactions.append((transaction_id, amount))

    def rollback(self, transaction_id: str):
        for idx, (t_id, amt) in enumerate(self.pending_transactions):
            if t_id == transaction_id:
                self.pending_transactions.pop(idx)
                break

    def commit(self):
        proposed_balance = self.balance + sum(amt for _, amt in self.pending_transactions)
        if proposed_balance < -self.overdraft_limit:
            self.pending_transactions = []
            raise ValueError("Commit failed: Overdraft limit exceeded")
            
        self.balance = proposed_balance
        for t_id, amt in self.pending_transactions:
            self.transaction_history.append((t_id, amt))
        self.pending_transactions = []
```""",

    # Task 4: JSON Serializer
    "serialize_custom": """```python
import datetime
import json

def serialize_custom(obj, indent: int = None) -> str:
    visited = set()
    
    def _escape_str(s: str) -> str:
        return json.dumps(s)

    def _serialize(value) -> str:
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return _escape_str(value)
        elif isinstance(value, (datetime.datetime, datetime.date)):
            return _escape_str(value.isoformat())
            
        val_id = id(value)
        if val_id in visited:
            raise ValueError("Circular reference detected")
        visited.add(val_id)

        try:
            if isinstance(value, (list, tuple)):
                parts = [_serialize(item) for item in value]
                return "[" + ", ".join(parts) + "]"
            elif isinstance(value, dict):
                parts = []
                for k, v in value.items():
                    k_str = str(k)
                    parts.append(f"{_escape_str(k_str)}: {_serialize(v)}")
                return "{" + ", ".join(parts) + "}"
            elif hasattr(value, "to_json") and callable(getattr(value, "to_json")):
                return _serialize(value.to_json())
            elif hasattr(value, "__dict__"):
                return _serialize(value.__dict__)
            else:
                return _escape_str(str(value))
        finally:
            visited.remove(val_id)

    res = _serialize(obj)
    if indent is not None:
        return json.dumps(json.loads(res), indent=indent)
    return res
```""",

    # Task 5: Thread Safe Task Worker
    "threadsafetaskworker": """```python
import queue
import threading

class ThreadSafeTaskWorker:
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.threads = []
        self.results = {}
        self.errors = {}
        self.results_lock = threading.Lock()
        self.active_tasks = 0
        self.active_lock = threading.Lock()
        self.completion_event = threading.Event()
        self.running = False

    def submit_task(self, task_id: str, func: callable, *args, **kwargs) -> None:
        with self.active_lock:
            self.active_tasks += 1
            if self.active_tasks > 0:
                self.completion_event.clear()
        self.task_queue.put((task_id, func, args, kwargs))

    def _worker_loop(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            task_id, func, args, kwargs = task
            try:
                res = func(*args, **kwargs)
                with self.results_lock:
                    self.results[task_id] = res
            except Exception as e:
                with self.results_lock:
                    self.errors[task_id] = e
            finally:
                self.task_queue.task_done()
                with self.active_lock:
                    self.active_tasks -= 1
                    if self.active_tasks == 0:
                        self.completion_event.set()

    def start(self) -> None:
        self.running = True
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.threads.append(t)

    def wait_completion(self, timeout: float = None) -> bool:
        with self.active_lock:
            if self.active_tasks == 0:
                return True
        return self.completion_event.wait(timeout=timeout)

    def get_results(self) -> dict[str, any]:
        with self.results_lock:
            return dict(self.results)

    def get_errors(self) -> dict[str, Exception]:
        with self.results_lock:
            return dict(self.errors)

    def stop(self) -> None:
        self.running = False
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except queue.Empty:
                break
        for t in self.threads:
            t.join(timeout=0.1)
        self.threads = []
```"""
}

class MockLMSRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            req_body = json.loads(post_data)
        except Exception:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")
            return

        messages = req_body.get("messages", [])
        prompt_text = messages[0].get("content", "") if messages else ""
        
        # Match prompt text with new tasks
        matched_solution = SOLUTIONS["threadsafetaskworker"] # default fallback
        for key, code in SOLUTIONS.items():
            if key in prompt_text.lower():
                matched_solution = code
                break

        is_anthropic = "/messages" in self.path
        is_openai = "/chat/completions" in self.path or "/api/v1/chat" in self.path

        if not (is_openai or is_anthropic):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()

        # Split solution code into words to simulate streaming
        chunks = matched_solution.split(" ")
        for i, chunk in enumerate(chunks):
            val = chunk + (" " if i < len(chunks) - 1 else "")
            
            if is_openai:
                chunk_data = {
                    "choices": [{
                        "delta": {
                            "content": val
                        }
                    }]
                }
                self.wfile.write(f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8'))
            elif is_anthropic:
                chunk_data = {
                    "delta": {
                        "text": val
                    }
                }
                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(chunk_data)}\n\n".encode('utf-8'))

            self.wfile.flush()
            time.sleep(0.002) # 2ms delay between chunks

        if is_openai:
            usage_data = {
                "usage": {
                    "completion_tokens": len(matched_solution) // 4,
                    "prompt_tokens": 150,
                    "total_tokens": (len(matched_solution) // 4) + 150
                }
            }
            self.wfile.write(f"data: {json.dumps(usage_data)}\n\n".encode('utf-8'))
            self.wfile.write(b"data: [DONE]\n\n")
        elif is_anthropic:
            usage_data = {
                "usage": {
                    "output_tokens": len(matched_solution) // 4,
                    "input_tokens": 150
                }
            }
            self.wfile.write(f"event: message_delta\ndata: {json.dumps(usage_data)}\n\n".encode('utf-8'))

        self.wfile.flush()

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            models_data = {
                "data": [
                    {"id": "qwen/qwen3.6-27b", "object": "model"},
                    {"id": "qwen3.5-0.8b", "object": "model"}
                ]
            }
            self.wfile.write(json.dumps(models_data).encode('utf-8'))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Mock LMS Server Running")

def run_server():
    server_address = ('', PORT)
    httpd = ThreadingHTTPServer(server_address, MockLMSRequestHandler)
    print(f"Mock LMS server listening on port {PORT}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Mock LMS server stopped.")

if __name__ == "__main__":
    run_server()
