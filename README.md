# LMS Model Benchmarking Suite

A robust evaluation suite for local language models (LMS/LM Studio, Ollama, etc.) that measures both **speed** (latency, throughput) and **coding quality** (unit test correctness).

This benchmarking suite implements the workflow discussed in [ChatGPT Model Benchmarking](https://chatgpt.com/share/6a4eb55a-4100-83e8-9d57-d99a830c8717).

## Features
- **3 Endpoint Protocols**: Out-of-the-box support for OpenAI, Anthropic, and LM Studio native APIs.
- **6 Standard Coding Tasks**:
  1. `task_01_api_integration`: Implement a paginated REST API client with rate-limit retries.
  2. `task_02_data_refactoring`: Refactor log processing into rolling window metrics.
  3. `task_03_bugfix_transaction`: Fix atomic transaction commit and rollback behavior.
  4. `task_04_json_serializer`: Implement custom JSON serialization with circular-reference detection.
  5. `task_05_concurrency_race`: Implement a thread-safe worker pool with defined stop semantics.
  6. `task_06_file_backed_analytics`: Compute LMS analytics from bundled JSON and CSV fixture files.
- **Rich Metric Tracking**:
  - **TTFT**: Time to First Token (calculated via streaming response).
  - **Speed**: Output Tokens per Second (TPS).
  - **Wall Time**: Total request round-trip time.
  - **Pass Rate**: Pytest validation of the generated code.
- **Isolated Temp Runner**: Each generated solution is tested in a copied temporary task directory with a minimal environment. The runner sets `PYTHONPATH`, `PATH`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD`, and `PYTHONDONTWRITEBYTECODE`, and preserves a few Windows process keys when present. This protects benchmark hygiene, but it is not a security sandbox for hostile generated code.
- **HTML Reports**: Automatically compiles results into `benchmark_report.html` with summary metrics, charts, per-task pass rates, and expandable run logs.

---

## Setup & Requirements

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Load your model in LM Studio**:
   Make sure you start the Local Server in LM Studio (usually serving on port `1234` or configured to port `5555`).

---

## Running the Benchmark

Run the benchmarking harness using Python:

```bash
python3 benchmark.py --api-base http://localhost:5555 --model qwen/qwen3.6-27b --api-type openai --runs 1
```

### Command Line Arguments
- `--api-base`: The base URL of the model endpoint (default: `http://localhost:5555`).
- `--model`: Model name / identifier matching the endpoint's model list (default: `qwen/qwen3.6-27b`).
- `--api-type`: Protocol endpoint type (`openai`, `anthropic`, or `lm_studio`).
- `--runs`: Number of repetitions per task (default: `1`).
- `--warmup`: Send one warmup request before timed benchmark runs.
- `--quant`: Quantization label to store in the report (for example `Q4_K_M`, `Q8_0`, `IQ4_NL`).
- `--context-len`: Context window label to store in the report (for example `16k`, `32k`).
- `--mtp`: Multi-token prediction setting label.
- `--draft`: Draft model or draft-size label for speculative decoding.
- `--gpu-layers`: Number of layers offloaded to GPU.
- `--flash-attention`: Flash Attention setting label.
- `--samplers`: Sampler settings label (for example `temp=0.1, top_p=0.9`).
- `--seed`: Random seed label.
- `--lms-version`: LM Studio version label.

---

## Customizing Configuration in Code

At the top of [`benchmark.py`](benchmark.py), you can directly change the default configuration variables:

```python
# Easily changeable variables matching the endpoint and model identifiers
API_BASE = "http://localhost:5555"
MODEL_NAME = "qwen/qwen3.6-27b"  # Default model identifier
API_TYPE = "openai"              # Options: "openai", "anthropic", "lm_studio"
TEMPERATURE = 0.1
NUM_RUNS_PER_TASK = 1            # Run each task this many times for averaging
TIMEOUT = 60                     # Timeout for API requests in seconds
TEST_TIMEOUT = 10                # Timeout for unit test execution
```

---

## Local Testing with Mock Server

We include a mock LMS server (`mock_lms.py`) to test the benchmarking runner and generate a full report without calling a real LLM:

1. **Start the Mock Server**:
   ```bash
   python3 mock_lms.py
   ```
   *(Listens on port 5555, serving mock OpenAI and Anthropic compatible streaming endpoints)*

2. **Run the Benchmark**:
   ```bash
   python3 benchmark.py
   ```

3. **Check Output**:
   - `benchmark_results.json`: Detailed raw JSON records of the run.
   - `benchmark_report.html`: Beautiful interactive report dashboard. Open this in your browser to view charts, speeds, and test logs.
