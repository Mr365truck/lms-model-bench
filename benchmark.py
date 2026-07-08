#!/usr/bin/env python3
import os
import sys
import time
import json
import re
import subprocess
import shutil
import tempfile
import requests

# ==============================================================================
# CONFIGURATION DEFAULTS
# ==============================================================================
API_BASE = "http://localhost:5555"
MODEL_NAME = "qwen/qwen3.6-27b"  # Default model identifier
API_TYPE = "openai"              # Options: "openai", "anthropic", "lm_studio"
TEMPERATURE = 0.1
NUM_RUNS_PER_TASK = 1            # Run each task this many times for averaging
TIMEOUT = 60                     # Timeout for API requests in seconds
TEST_TIMEOUT = 10                # Timeout for unit test execution (prevent hangs)
# ==============================================================================

def extract_python_code(text):
    """Extracts python code blocks from markdown response."""
    # Look for ```python ... ```
    match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    # Look for ``` ... ```
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback to entire text
    return text

def parse_openai_stream(response):
    """Parses OpenAI stream and yields chunks of text and token counts if available."""
    full_text = ""
    ttft = None
    start_time = time.time()
    usage = None

    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8').strip()
        if line_str.startswith("data: "):
            data_content = line_str[6:]
            if data_content == "[DONE]":
                break
            try:
                chunk = json.loads(data_content)
                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if ttft is None:
                            ttft = time.time() - start_time
                        full_text += content
                        yield content, ttft, usage
            except Exception:
                pass

def parse_anthropic_stream(response):
    """Parses Anthropic stream and yields chunks of text and token counts if available."""
    full_text = ""
    ttft = None
    start_time = time.time()
    usage = None

    current_event = None
    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8').strip()
        if line_str.startswith("event: "):
            current_event = line_str[7:]
        elif line_str.startswith("data: "):
            data_content = line_str[6:]
            try:
                chunk = json.loads(data_content)
                if current_event == "content_block_delta" or "delta" in chunk:
                    delta = chunk.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        if ttft is None:
                            ttft = time.time() - start_time
                        full_text += text
                        yield text, ttft, usage
                elif current_event == "message_delta" or "usage" in chunk:
                    if "usage" in chunk:
                        usage = chunk["usage"]
                    elif "delta" in chunk and "usage" in chunk["delta"]:
                        usage = chunk["delta"]["usage"]
            except Exception:
                pass

def parse_lm_studio_stream(response):
    yield from parse_openai_stream(response)

def query_model(api_base, api_type, model_name, prompt):
    """Queries the model and returns (response_text, ttft, total_time, token_count, is_estimated)."""
    headers = {"Content-Type": "application/json"}
    
    if api_type == "openai":
        url = f"{api_base.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        stream_parser = parse_openai_stream
    elif api_type == "anthropic":
        url = f"{api_base.rstrip('/')}/v1/messages"
        headers["x-api-key"] = "local"
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": TEMPERATURE,
            "stream": True
        }
        stream_parser = parse_anthropic_stream
    elif api_type == "lm_studio":
        url = f"{api_base.rstrip('/')}/api/v1/chat"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "stream": True
        }
        stream_parser = parse_lm_studio_stream
    else:
        raise ValueError(f"Unknown API type: {api_type}")

    start_time = time.time()
    
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=TIMEOUT)
    response.raise_for_status()

    full_text = ""
    ttft = None
    usage = None

    for content, current_ttft, current_usage in stream_parser(response):
        full_text += content
        if ttft is None:
            ttft = current_ttft
        if current_usage:
            usage = current_usage

    total_time = time.time() - start_time
    if ttft is None:
        ttft = total_time

    token_count = None
    if usage:
        token_count = usage.get("completion_tokens") or usage.get("output_tokens")
    
    is_estimated = False
    if token_count is None:
        token_count = max(1, int(len(full_text) / 4))
        is_estimated = True

    return full_text, ttft, total_time, token_count, is_estimated

def run_task_tests(task_dir, extracted_code, base_dir, timeout_sec=TEST_TIMEOUT):
    """Executes the test suite inside an isolated temporary sandbox directory to prevent workspace leaks/damage."""
    sandbox_base = os.path.join(base_dir, "sandbox_tmp")
    os.makedirs(sandbox_base, exist_ok=True)
    
    # Create temporary directory inside workspace sandbox_base
    with tempfile.TemporaryDirectory(dir=sandbox_base) as temp_dir:
        # 1. Write the extracted code to solution.py
        solution_file = os.path.join(temp_dir, "solution.py")
        with open(solution_file, "w", encoding="utf-8") as f:
            f.write(extracted_code)
            
        # 2. Copy the task's tests.py to the sandbox
        src_tests = os.path.join(task_dir, "tests.py")
        dst_tests = os.path.join(temp_dir, "tests.py")
        if os.path.exists(src_tests):
            shutil.copy(src_tests, dst_tests)
            
        # 3. Create empty __init__.py to facilitate imports
        with open(os.path.join(temp_dir, "__init__.py"), "w") as f:
            pass
            
        # 4. Prepare execution command
        cmd = [sys.executable, "-m", "pytest", "-v", "tests.py"]
        
        # Build clean environment with PYTHONPATH pointing to the isolated directory
        clean_env = os.environ.copy()
        clean_env["PYTHONPATH"] = temp_dir
        
        try:
            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                env=clean_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec
            )
            passed = (result.returncode == 0)
            output = result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            passed = False
            output = f"TIMEOUT: Test execution exceeded the limit of {timeout_sec} seconds (possible infinite loop or hang in generated code)."
            
    return passed, output

def generate_html_report(results, config):
    """Generates a premium HTML report summary of the benchmark runs."""
    total_runs = len(results)
    passed_runs = sum(1 for r in results if r["passed"])
    pass_rate = (passed_runs / total_runs * 100) if total_runs > 0 else 0
    
    successful_speed_runs = [r for r in results if r.get("decode_tps") is not None and r["decode_tps"] > 0]
    avg_decode_tps = (sum(r["decode_tps"] for r in successful_speed_runs) / len(successful_speed_runs)) if successful_speed_runs else 0
    avg_e2e_tps = (sum(r["e2e_tps"] for r in successful_speed_runs) / len(successful_speed_runs)) if successful_speed_runs else 0
    avg_ttft = (sum(r["ttft"] for r in results) / total_runs) if total_runs > 0 else 0
    avg_wall_time = (sum(r["wall_time"] for r in results) / total_runs) if total_runs > 0 else 0
    
    # Group results by task
    tasks_summary = {}
    for r in results:
        task = r["task"]
        if task not in tasks_summary:
            tasks_summary[task] = {
                "runs": [],
                "passed_count": 0,
                "decode_tps_sum": 0,
                "e2e_tps_sum": 0,
                "ttft_sum": 0,
                "wall_time_sum": 0,
                "metadata": r.get("metadata", {})
            }
        tasks_summary[task]["runs"].append(r)
        if r["passed"]:
            tasks_summary[task]["passed_count"] += 1
        tasks_summary[task]["decode_tps_sum"] += r["decode_tps"]
        tasks_summary[task]["e2e_tps_sum"] += r["e2e_tps"]
        tasks_summary[task]["ttft_sum"] += r["ttft"]
        tasks_summary[task]["wall_time_sum"] += r["wall_time"]

    # Calculate task-specific averages
    task_rows = []
    for task_name, info in tasks_summary.items():
        n = len(info["runs"])
        task_pass_rate = (info["passed_count"] / n * 100)
        task_avg_decode_tps = info["decode_tps_sum"] / n
        task_avg_e2e_tps = info["e2e_tps_sum"] / n
        task_avg_ttft = info["ttft_sum"] / n
        task_avg_wall_time = info["wall_time_sum"] / n
        
        meta = info["metadata"]
        disp_name = meta.get("name", task_name.replace("task_", "").replace("_", " ").title())
        category = meta.get("category", "General")
        difficulty = meta.get("difficulty", "Medium")
        
        task_rows.append({
            "name": disp_name,
            "category": category,
            "difficulty": difficulty,
            "runs_count": n,
            "pass_rate": f"{task_pass_rate:.1f}%",
            "passed_count": info["passed_count"],
            "avg_decode_tps": f"{task_avg_decode_tps:.1f}",
            "avg_e2e_tps": f"{task_avg_e2e_tps:.1f}",
            "avg_ttft": f"{task_avg_ttft:.3f}s",
            "avg_wall_time": f"{task_avg_wall_time:.2f}s"
        })

    results_json = json.dumps(results, indent=2)

    task_rows_html = []
    for row in task_rows:
        pass_rate_val = float(row['pass_rate'].rstrip('%'))
        badge_class = 'badge-success' if pass_rate_val > 50 else 'badge-error'
        diff_class = 'badge-success' if row['difficulty'] == 'Easy' else ('badge-warning' if row['difficulty'] == 'Medium' else 'badge-danger')
        
        row_html = f"""
                    <tr>
                        <td>
                            <div style="font-weight: 600;">{row['name']}</div>
                            <div style="font-size: 0.75rem; color: var(--text-muted);">{row['category']}</div>
                        </td>
                        <td><span class="badge {diff_class}">{row['difficulty']}</span></td>
                        <td>{row['runs_count']}</td>
                        <td>
                            <span class="badge {badge_class}">
                                {row['pass_rate']}
                            </span>
                        </td>
                        <td style="font-family: 'JetBrains Mono', monospace;">{row['avg_decode_tps']} tok/s</td>
                        <td style="font-family: 'JetBrains Mono', monospace; color: var(--text-muted); font-size: 0.85rem;">{row['avg_e2e_tps']} tok/s</td>
                        <td style="font-family: 'JetBrains Mono', monospace;">{row['avg_ttft']}</td>
                        <td style="font-family: 'JetBrains Mono', monospace;">{row['avg_wall_time']}</td>
                    </tr>
        """
        task_rows_html.append(row_html)
    task_rows_html_str = "".join(task_rows_html)

    detailed_runs_html = []
    for idx, run in enumerate(results):
        badge_class = 'badge-success' if run['passed'] else 'badge-error'
        status_text = 'PASS' if run['passed'] else 'FAIL'
        test_output_class = 'test-output-pass' if run['passed'] else 'test-output-fail'
        
        est_suffix = " (est)" if run.get("is_estimated") else ""
        decode_speed = f"{run['decode_tps']:.1f} tok/s{est_suffix}"
        e2e_speed = f"{run['e2e_tps']:.1f} tok/s{est_suffix}"
        
        meta = run.get("metadata", {})
        disp_name = meta.get("name", run['task'].replace("task_", "").replace("_", " ").title())
        
        run_html = f"""
            <div class="run-accordion" id="run-container-{idx}">
                <div class="run-header" onclick="toggleAccordion({idx})">
                    <div class="run-title-group">
                        <span class="badge {badge_class}">
                            {status_text}
                        </span>
                        <span style="font-weight: 600;">{disp_name}</span>
                        <span style="color: var(--text-muted); font-size: 0.85rem;">Run #{run['run_index']}</span>
                    </div>
                    <div class="run-meta-group">
                        <span>Decode Speed: <strong>{decode_speed}</strong></span>
                        <span>E2E Speed: <strong style="color: var(--text-muted);">{e2e_speed}</strong></span>
                        <span>TTFT: <strong>{run['ttft']:.3f}s</strong></span>
                        <span>Wall Time: <strong>{run['wall_time']:.2f}s</strong></span>
                        <span id="accordion-arrow-{idx}">▼</span>
                    </div>
                </div>
                <div class="run-content" id="run-content-{idx}">
                    <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
                        <div class="code-container">
                            <div class="code-title">Extracted Python Code (solution.py)</div>
                            <pre><code>{html.escape(run['extracted_code'])}</code></pre>
                        </div>
                        <div class="code-container">
                            <div class="code-title">Pytest Results Output</div>
                            <pre class="{test_output_class}"><code>{html.escape(run['test_output'])}</code></pre>
                        </div>
                    </div>
                </div>
            </div>
        """
        detailed_runs_html.append(run_html)
    detailed_runs_html_str = "".join(detailed_runs_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LMS Model Benchmarking Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-secondary: #161f30;
            --bg-tertiary: #1f2d44;
            --accent: #3b82f6;
            --accent-success: #10b981;
            --accent-warning: #fbbf24;
            --accent-danger: #ef4444;
            --accent-error: #ef4444;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border-color: rgba(255, 255, 255, 0.08);
            --card-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            line-height: 1.6;
            padding: 2rem 1.5rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            background: linear-gradient(135deg, rgba(22, 31, 48, 0.8) 0%, rgba(11, 15, 25, 0.8) 100%);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2.5rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(12px);
            box-shadow: var(--card-shadow);
            position: relative;
            overflow: hidden;
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to right, #ffffff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
            border-top: 1px solid var(--border-color);
            padding-top: 1.5rem;
        }}

        .config-item {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.8rem 1.2rem;
        }}

        .config-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.2rem;
        }}

        .config-value {{
            font-weight: 600;
            font-size: 0.95rem;
            font-family: 'JetBrains Mono', monospace;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.8rem;
            box-shadow: var(--card-shadow);
        }}

        .metric-title {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .metric-value {{
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }}

        .metric-desc {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .metric-pass {{ color: var(--accent-success); }}
        .metric-tps {{ color: var(--accent); }}

        .chart-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .chart-container {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: var(--card-shadow);
            height: 320px;
        }}

        .chart-title {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}

        .table-section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.8rem;
            box-shadow: var(--card-shadow);
            margin-bottom: 2rem;
            overflow-x: auto;
        }}

        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}

        th {{
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 1rem;
            border-bottom: 2px solid var(--border-color);
        }}

        td {{
            padding: 1.2rem 1rem;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.95rem;
        }}

        .badge {{
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }}

        .badge-success {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .badge-warning {{
            background: rgba(251, 191, 36, 0.15);
            color: var(--accent-warning);
            border: 1px solid rgba(251, 191, 36, 0.3);
        }}

        .badge-danger {{
            background: rgba(239, 68, 68, 0.15);
            color: var(--accent-danger);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        .badge-error {{
            background: rgba(239, 68, 68, 0.15);
            color: var(--accent-error);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        .run-accordion {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 0.8rem;
            box-shadow: var(--card-shadow);
        }}

        .run-header {{
            padding: 1.2rem 1.5rem;
            background: rgba(255, 255, 255, 0.02);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }}

        .run-header:hover {{ background: rgba(255, 255, 255, 0.04); }}

        .run-title-group {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .run-meta-group {{
            display: flex;
            align-items: center;
            gap: 1.5rem;
            font-size: 0.9rem;
            color: var(--text-muted);
        }}

        .run-content {{
            padding: 1.5rem;
            border-top: 1px solid var(--border-color);
            display: none;
            background: rgba(11, 15, 25, 0.4);
        }}

        .code-container {{ margin-top: 1rem; }}
        .code-title {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }}

        pre {{
            background: #07090e;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 1rem;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #e5e7eb;
        }}

        .test-output-pass {{ border-left: 4px solid var(--accent-success); }}
        .test-output-fail {{ border-left: 4px solid var(--accent-error); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LMS Model Benchmarking</h1>
            <p style="color: var(--text-muted);">Evaluation report containing latency, throughput, and code-generation correctiveness.</p>
            
            <div class="config-grid">
                <div class="config-item">
                    <div class="config-label">Model Name</div>
                    <div class="config-value" style="color: var(--accent);">{config.get('model')}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Quantization</div>
                    <div class="config-value">{config.get('quant') or 'Unknown'}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Context Length</div>
                    <div class="config-value">{config.get('context_len') or 'Unknown'}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">MTP Setting</div>
                    <div class="config-value">{config.get('mtp') or 'Unknown'}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Draft Size / Model</div>
                    <div class="config-value">{config.get('draft') or 'None'}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">GPU Layers</div>
                    <div class="config-value">{config.get('gpu_layers') or 'Unknown'}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Flash Attention</div>
                    <div class="config-value">{config.get('flash_attention') or 'Unknown'}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">API protocol</div>
                    <div class="config-value" style="text-transform: uppercase;">{config.get('api_type')}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Sampler Settings</div>
                    <div class="config-value">{config.get('samplers') or 'Default'}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Seed</div>
                    <div class="config-value">{config.get('seed') or 'Random'}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">LM Studio Version</div>
                    <div class="config-value">{config.get('lms_version') or 'Unknown'}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Generated At</div>
                    <div class="config-value">{time.strftime('%Y-%m-%d %H:%M:%S local')}</div>
                </div>
            </div>
        </header>

        <div class="metrics-grid">
            <div class="card">
                <div class="metric-title">Unit Test Pass Rate</div>
                <div class="metric-value metric-pass">{pass_rate:.1f}%</div>
                <div class="metric-desc">{passed_runs} of {total_runs} runs passed</div>
            </div>
            <div class="card">
                <div class="metric-title">Avg Decode Speed</div>
                <div class="metric-value metric-tps">{avg_decode_tps:.1f} <span style="font-size: 1.2rem; font-weight: 500;">tok/s</span></div>
                <div class="metric-desc">Token generation rate (excluding TTFT)</div>
            </div>
            <div class="card">
                <div class="metric-title">Avg E2E Throughput</div>
                <div class="metric-value" style="color: #6366f1;">{avg_e2e_tps:.1f} <span style="font-size: 1.2rem; font-weight: 500;">tok/s</span></div>
                <div class="metric-desc">End-to-End throughput (including TTFT)</div>
            </div>
            <div class="card">
                <div class="metric-title">Avg Latency (TTFT) / Wall Time</div>
                <div class="metric-value" style="color: #fbbf24;">{avg_ttft:.3f}s <span style="font-size: 1.2rem; font-weight: 500; color: var(--text-muted);">/ {avg_wall_time:.1f}s</span></div>
                <div class="metric-desc">TTFT / Total Wall Time average</div>
            </div>
        </div>

        <div class="chart-section">
            <div class="chart-container">
                <div class="chart-title">Pass Rate per Coding Task</div>
                <canvas id="passRateChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Token Speed & Latency by Run</div>
                <canvas id="metricsChart"></canvas>
            </div>
        </div>

        <div class="table-section">
            <div class="section-title">
                <span style="color: var(--accent);">📊</span> Task-Specific Performance
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Coding Task</th>
                        <th>Difficulty</th>
                        <th>Runs</th>
                        <th>Pass Rate</th>
                        <th>Avg Decode</th>
                        <th>Avg E2E</th>
                        <th>Avg Latency</th>
                        <th>Avg Wall Time</th>
                    </tr>
                </thead>
                <tbody>
                    {task_rows_html_str}
                </tbody>
            </table>
        </div>

        <div class="section-title" style="margin-bottom: 1rem;">
            <span style="color: var(--accent);">🔍</span> Detailed Run Logs
        </div>
        <div class="runs-section">
            {detailed_runs_html_str}
        </div>
    </div>

    <script>
        function toggleAccordion(idx) {{
            const content = document.getElementById('run-content-' + idx);
            const arrow = document.getElementById('accordion-arrow-' + idx);
            if (content.style.display === 'block') {{
                content.style.display = 'none';
                arrow.textContent = '▼';
            }} else {{
                content.style.display = 'block';
                arrow.textContent = '▲';
            }}
        }}

        const rawResults = {results_json};
        
        const tasksSummary = {{}};
        rawResults.forEach(r => {{
            const name = r.metadata && r.metadata.name ? r.metadata.name : r.task.replace("task_", "").replace("_", " ").title();
            if (!tasksSummary[name]) {{
                tasksSummary[name] = {{ passed: 0, total: 0 }};
            }}
            tasksSummary[name].total++;
            if (r.passed) tasksSummary[name].passed++;
        }});

        const taskLabels = Object.keys(tasksSummary);
        const passRates = taskLabels.map(t => (tasksSummary[t].passed / tasksSummary[t].total) * 100);

        const ctxPass = document.getElementById('passRateChart').getContext('2d');
        new Chart(ctxPass, {{
            type: 'bar',
            data: {{
                labels: taskLabels,
                datasets: [{{
                    label: 'Pass Rate (%)',
                    data: passRates,
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderColor: '#10b981',
                    borderWidth: 1,
                    borderRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }}
            }}
        }});

        const runIndices = rawResults.map((r, i) => {{
            const name = r.metadata && r.metadata.name ? r.metadata.name : r.task.replace("task_", "").replace("_", " ");
            return `#${{i+1}} (${{name.substring(0, 10)}}...)`;
        }});
        const decodeTpsValues = rawResults.map(r => r.decode_tps);
        const e2eTpsValues = rawResults.map(r => r.e2e_tps);
        const ttftValues = rawResults.map(r => r.ttft);

        const ctxMetrics = document.getElementById('metricsChart').getContext('2d');
        new Chart(ctxMetrics, {{
            type: 'line',
            data: {{
                labels: runIndices,
                datasets: [
                    {{
                        label: 'Decode Speed (tok/s)',
                        data: decodeTpsValues,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'E2E Throughput (tok/s)',
                        data: e2eTpsValues,
                        borderColor: '#6366f1',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.3,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'TTFT (s)',
                        data: ttftValues,
                        borderColor: '#fbbf24',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        tension: 0.3,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{ display: true, text: 'Tokens / Second', color: '#3b82f6' }},
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#3b82f6' }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{ display: true, text: 'TTFT (Seconds)', color: '#fbbf24' }},
                        grid: {{ drawOnChartArea: false }},
                        ticks: {{ color: '#fbbf24' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#f3f4f6' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    with open("benchmark_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LMS Model Benchmarking Harness")
    parser.add_argument("--api-base", default=API_BASE, help="LMS Endpoint Base URL")
    parser.add_argument("--model", default=MODEL_NAME, help="Model identifier matching endpoint ID")
    parser.add_argument("--api-type", default=API_TYPE, choices=["openai", "anthropic", "lm_studio"], help="Protocol type")
    parser.add_argument("--runs", type=int, default=NUM_RUNS_PER_TASK, help="Number of runs per task")
    parser.add_argument("--warmup", action="store_true", help="Perform a warmup query before benchmarking")
    
    parser.add_argument("--quant", default="Unknown", help="Quantization (e.g. Q4_K_M, Q8_0, IQ4_NL)")
    parser.add_argument("--context-len", default="Unknown", help="Context window length (e.g. 16k, 32k)")
    parser.add_argument("--mtp", default="Unknown", help="Multi-token prediction settings (e.g. off, 1, 2, 4)")
    parser.add_argument("--draft", default="None", help="Draft model used for speculative decoding")
    parser.add_argument("--gpu-layers", default="Unknown", help="Number of layers offloaded to GPU")
    parser.add_argument("--flash-attention", default="Unknown", help="Flash Attention setting (e.g. Enabled, Disabled)")
    parser.add_argument("--samplers", default="Default", help="Custom sampler settings (e.g. temp=0.1, top_p=0.9)")
    parser.add_argument("--seed", default="Random", help="Random seed used")
    parser.add_argument("--lms-version", default="Unknown", help="LM Studio application version")

    args = parser.parse_args()

    config_dict = {
        "api_base": args.api_base,
        "model": args.model,
        "api_type": args.api_type,
        "runs": args.runs,
        "quant": args.quant,
        "context_len": args.context_len,
        "mtp": args.mtp,
        "draft": args.draft,
        "gpu_layers": args.gpu_layers,
        "flash_attention": args.flash_attention,
        "samplers": args.samplers,
        "seed": args.seed,
        "lms_version": args.lms_version
    }

    print("=" * 60)
    print(" LMS MODEL BENCHMARK HARNESS (SANDBOXED RUNNER)")
    print("=" * 60)
    print(f"Target URL:    {args.api_base}")
    print(f"Model Name:    {args.model}")
    print(f"Protocol Type: {args.api_type}")
    print(f"Runs/Task:     {args.runs}")
    print(f"Quantization:  {args.quant}")
    print(f"MTP Setting:   {args.mtp}")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    bench_dir = os.path.join(base_dir, "bench")
    if not os.path.exists(bench_dir):
        print(f"Error: bench/ directory not found in {base_dir}")
        sys.exit(1)

    tasks = sorted([d for d in os.listdir(bench_dir) if os.path.isdir(os.path.join(bench_dir, d))])
    if not tasks:
        print("No tasks found in bench/ directory.")
        sys.exit(0)

    print(f"Discovered {len(tasks)} tasks to evaluate.")
    
    if args.warmup:
        print("\nPerforming warmup request to wake model...")
        try:
            query_model(args.api_base, args.api_type, args.model, "Return 'Warmed' and nothing else.")
            print("Warmup complete.")
        except Exception as e:
            print(f"Warmup failed (skipping): {e}")

    results = []

    global html
    import html

    for task in tasks:
        task_path = os.path.join(bench_dir, task)
        prompt_file = os.path.join(task_path, "prompt.txt")
        metadata_file = os.path.join(task_path, "metadata.json")
        
        if not os.path.exists(prompt_file):
            print(f"Skipping task '{task}' - prompt.txt is missing.")
            continue

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()

        # Load task metadata
        task_meta = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    task_meta = json.load(f)
            except Exception:
                pass

        for run_idx in range(1, args.runs + 1):
            print(f"\nEvaluating task '{task}' (Run {run_idx}/{args.runs})...")
            
            # 1. Query LLM
            try:
                response_text, ttft, wall_time, token_count, is_estimated = query_model(
                    args.api_base, args.api_type, args.model, prompt
                )
                
                # Split TPS into End-to-End and Pure Decode
                e2e_tps = token_count / wall_time if wall_time > 0 else 0
                decode_time = max(wall_time - ttft, 1e-6)
                decode_tps = token_count / decode_time
                
                est_suffix = " (est)" if is_estimated else ""
                print(f"  API Response: TTFT={ttft:.3f}s, Time={wall_time:.2f}s, Tokens={token_count}")
                print(f"  Speed: Decode={decode_tps:.1f} tok/s{est_suffix}, E2E={e2e_tps:.1f} tok/s{est_suffix}")
            except Exception as e:
                print(f"  [ERROR] API Query failed: {e}")
                results.append({
                    "task": task,
                    "metadata": task_meta,
                    "run_index": run_idx,
                    "passed": False,
                    "ttft": 0.0,
                    "wall_time": 0.0,
                    "e2e_tps": 0.0,
                    "decode_tps": 0.0,
                    "response_text": "",
                    "extracted_code": "",
                    "test_output": f"API Request Failed: {e}",
                    "is_estimated": False
                })
                continue

            # 2. Extract code
            extracted_code = extract_python_code(response_text)

            # 3. Run unit tests in isolated temporary sandbox directory
            print(f"  Executing unit tests via pytest in isolated sandbox...")
            passed, test_output = run_task_tests(task_path, extracted_code, base_dir, timeout_sec=TEST_TIMEOUT)
            
            status_str = "PASS" if passed else "FAIL"
            print(f"  Test Result: {status_str}")
            
            # Store results, clean local path references to protect repo hygiene
            results.append({
                "task": task,
                "metadata": task_meta,
                "run_index": run_idx,
                "passed": passed,
                "ttft": ttft,
                "wall_time": wall_time,
                "e2e_tps": e2e_tps,
                "decode_tps": decode_tps,
                "response_text": response_text,
                "extracted_code": extracted_code,
                "test_output": test_output.replace(base_dir, ""),
                "is_estimated": is_estimated
            })

    # Save raw results JSON
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate HTML report
    generate_html_report(results, config_dict)
    
    print("\n" + "=" * 60)
    print(" BENCHMARK COMPLETE")
    print("=" * 60)
    passed_total = sum(1 for r in results if r["passed"])
    overall_rate = (passed_total / len(results) * 100) if results else 0
    print(f"Overall Pass Rate: {overall_rate:.1f}% ({passed_total}/{len(results)} runs)")
    print("Generated files (git-ignored):")
    print(" - benchmark_results.json (raw logs)")
    print(" - benchmark_report.html   (visual report summary)")
    print("=" * 60)

if __name__ == "__main__":
    main()
