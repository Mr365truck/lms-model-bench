#!/usr/bin/env python3
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse

PORT = 5555

# Correct solutions to return for the benchmark tasks
SOLUTIONS = {
    "LRUCache": """```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```""",

    "merge": """```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(current)
    return merged
```""",

    "Trie": """```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```""",

    "isValid": """```python
def isValid(s: str) -> bool:
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack
```""",

    "searchRange": """```python
def searchRange(nums: list[int], target: int) -> list[int]:
    def findBound(isFirst):
        left, right = 0, len(nums) - 1
        bound = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                bound = mid
                if isFirst:
                    right = mid - 1
                else:
                    left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return bound
    return [findBound(True), findBound(False)]
```"""
}

class MockLMSRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress server logging to keep terminal output clean
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

        # Find which prompt is requested
        messages = req_body.get("messages", [])
        prompt_text = messages[0].get("content", "") if messages else ""
        
        # Match prompt with a solution
        matched_solution = SOLUTIONS["isValid"] # default fallback
        for key, code in SOLUTIONS.items():
            if key.lower() in prompt_text.lower():
                matched_solution = code
                break

        # Check endpoints and format streaming response
        # OpenAI style: POST /v1/chat/completions
        # Anthropic style: POST /v1/messages
        # LM Studio style: POST /api/v1/chat
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

        # Split solution code into words to simulate streaming chunks
        chunks = matched_solution.split(" ")
        for i, chunk in enumerate(chunks):
            # Add back space removed by split
            val = chunk + (" " if i < len(chunks) - 1 else "")
            
            # OpenAI / LM Studio format
            if is_openai:
                chunk_data = {
                    "choices": [{
                        "delta": {
                            "content": val
                        }
                    }]
                }
                self.wfile.write(f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8'))
            
            # Anthropic format
            elif is_anthropic:
                chunk_data = {
                    "delta": {
                        "text": val
                    }
                }
                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(chunk_data)}\n\n".encode('utf-8'))

            self.wfile.flush()
            time.sleep(0.002) # 2ms delay between chunks

        # Send usage info in final chunk
        if is_openai:
            usage_data = {
                "usage": {
                    "completion_tokens": len(matched_solution) // 4,
                    "prompt_tokens": 100,
                    "total_tokens": (len(matched_solution) // 4) + 100
                }
            }
            self.wfile.write(f"data: {json.dumps(usage_data)}\n\n".encode('utf-8'))
            self.wfile.write(b"data: [DONE]\n\n")
        elif is_anthropic:
            usage_data = {
                "usage": {
                    "output_tokens": len(matched_solution) // 4,
                    "input_tokens": 100
                }
            }
            self.wfile.write(f"event: message_delta\ndata: {json.dumps(usage_data)}\n\n".encode('utf-8'))

        self.wfile.flush()

    def do_GET(self):
        # Health check or list models
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
