#!/usr/bin/env python3
import os
import sys
import time
import json
import re
import subprocess
import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Easily changeable variables matching the endpoint and model identifiers
API_BASE = "http://localhost:5555"
MODEL_NAME = "qwen/qwen3.6-27b"  # Default model identifier
API_TYPE = "openai"              # Options: "openai", "anthropic", "lm_studio"
TEMPERATURE = 0.1
NUM_RUNS_PER_TASK = 1            # Run each task this many times for averaging
TIMEOUT = 60                     # Timeout for API requests in seconds
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
                # Check for usage
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
                    # Anthropic sends usage in message_delta event
                    if "usage" in chunk:
                        usage = chunk["usage"]
                    elif "delta" in chunk and "usage" in chunk["delta"]:
                        usage = chunk["delta"]["usage"]
            except Exception:
                pass

def parse_lm_studio_stream(response):
    """Parses LM Studio native stream or falls back to OpenAI parser."""
    # LM Studio's native endpoint /api/v1/chat uses a streaming format
    # very similar to OpenAI. We can reuse the OpenAI parser.
    yield from parse_openai_stream(response)

def query_model(api_base, api_type, model_name, prompt):
    """Queries the model and returns (response_text, ttft, total_time, token_count)."""
    headers = {"Content-Type": "application/json"}
    
    if api_type == "openai":
        url = f"{api_base.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "stream": True
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
        # Native LM Studio chat completion
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
    
    # Send the request
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
        ttft = total_time # Fallback if no streaming chunks detected

    # Calculate token count
    # Check usage response
    token_count = None
    if usage:
        # OpenAI style: completion_tokens, Anthropic style: output_tokens
        token_count = usage.get("completion_tokens") or usage.get("output_tokens")
    
    # If API didn't return usage stats, estimate using standard heuristic
    if token_count is None:
        # Heuristic: 1 token ≈ 4 characters
        token_count = max(1, int(len(full_text) / 4))

    return full_text, ttft, total_time, token_count

def run_task_tests(task_dir):
    """Executes the test suite in the task folder and returns (passed, output)."""
    # Overwrite/__init__.py to make sure it is seen as package if needed
    init_file = os.path.join(task_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            pass

    test_file = os.path.join(task_dir, "tests.py")
    
    # Run pytest on the specific task directory tests.py
    cmd = [sys.executable, "-m", "pytest", "-v", test_file]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    passed = (result.returncode == 0)
    output = result.stdout + "\n" + result.stderr
    return passed, output

def generate_html_report(results, api_base, model_name, api_type):
    """Generates a premium HTML report summary of the benchmark runs."""
    
    # Aggregate statistics
    total_runs = len(results)
    passed_runs = sum(1 for r in results if r["passed"])
    pass_rate = (passed_runs / total_runs * 100) if total_runs > 0 else 0
    
    successful_speed_runs = [r for r in results if r.get("tps") is not None and r["tps"] > 0]
    avg_tps = (sum(r["tps"] for r in successful_speed_runs) / len(successful_speed_runs)) if successful_speed_runs else 0
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
                "tps_sum": 0,
                "ttft_sum": 0,
                "wall_time_sum": 0,
            }
        tasks_summary[task]["runs"].append(r)
        if r["passed"]:
            tasks_summary[task]["passed_count"] += 1
        tasks_summary[task]["tps_sum"] += r["tps"]
        tasks_summary[task]["ttft_sum"] += r["ttft"]
        tasks_summary[task]["wall_time_sum"] += r["wall_time"]

    # Calculate task-specific averages
    task_rows = []
    for task_name, info in tasks_summary.items():
        n = len(info["runs"])
        task_pass_rate = (info["passed_count"] / n * 100)
        task_avg_tps = info["tps_sum"] / n
        task_avg_ttft = info["ttft_sum"] / n
        task_avg_wall_time = info["wall_time_sum"] / n
        
        task_rows.append({
            "name": task_name,
            "runs_count": n,
            "pass_rate": f"{task_pass_rate:.1f}%",
            "passed_count": info["passed_count"],
            "avg_tps": f"{task_avg_tps:.1f}",
            "avg_ttft": f"{task_avg_ttft:.3f}s",
            "avg_wall_time": f"{task_avg_wall_time:.2f}s"
        })

    # Prepare JSON dump of results for HTML charts
    results_json = json.dumps(results, indent=2)

    # Pre-render rows to avoid f-string nesting syntax errors
    task_rows_html = []
    for row in task_rows:
        pass_rate_val = float(row['pass_rate'].rstrip('%'))
        badge_class = 'badge-success' if pass_rate_val > 50 else 'badge-error'
        row_html = f"""
                    <tr>
                        <td style="font-weight: 600;">{row['name']}</td>
                        <td>{row['runs_count']}</td>
                        <td>
                            <span class="badge {badge_class}">
                                {row['pass_rate']}
                            </span>
                        </td>
                        <td style="font-family: 'JetBrains Mono', monospace;">{row['avg_tps']} tok/s</td>
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
        
        run_html = f"""
            <div class="run-accordion" id="run-container-{idx}">
                <div class="run-header" onclick="toggleAccordion({idx})">
                    <div class="run-title-group">
                        <span class="badge {badge_class}">
                            {status_text}
                        </span>
                        <span style="font-weight: 600;">{run['task']}</span>
                        <span style="color: var(--text-muted); font-size: 0.85rem;">Run #{run['run_index']}</span>
                    </div>
                    <div class="run-meta-group">
                        <span>Speed: <strong>{run['tps']:.1f} tok/s</strong></span>
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
    <!-- Modern typography -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <!-- Chart.js CDN for beautiful graphs -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-secondary: #161f30;
            --bg-tertiary: #1f2d44;
            --accent: #3b82f6;
            --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            --accent-success: #10b981;
            --accent-success-gradient: linear-gradient(135deg, #10b981 0%, #047857 100%);
            --accent-error: #ef4444;
            --accent-error-gradient: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
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

        /* Header design */
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

        header::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, rgba(0,0,0,0) 70%);
            border-radius: 50%;
            z-index: 0;
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to right, #ffffff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
            position: relative;
            z-index: 1;
        }}

        .meta-item {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.8rem 1.2rem;
        }}

        .meta-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.2rem;
        }}

        .meta-value {{
            font-weight: 600;
            font-size: 1rem;
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Metrics grid */
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
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.15);
            border-color: rgba(59, 130, 246, 0.3);
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

        .metric-pass {{
            color: var(--accent-success);
        }}

        .metric-tps {{
            color: var(--accent);
        }}

        /* Visual Charts section */
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
            color: var(--text-main);
        }}

        /* Table styling */
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

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.015);
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

        .badge-error {{
            background: rgba(239, 68, 68, 0.15);
            color: var(--accent-error);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        /* Expandable Accordion for detailed runs */
        .runs-section {{
            display: flex;
            flex-col: column;
            gap: 1rem;
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
            transition: background 0.2s;
        }}

        .run-header:hover {{
            background: rgba(255, 255, 255, 0.04);
        }}

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

        .code-container {{
            margin-top: 1rem;
        }}

        .code-title {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
            font-weight: 500;
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

        .test-output-pass {{
            border-left: 4px solid var(--accent-success);
        }}

        .test-output-fail {{
            border-left: 4px solid var(--accent-error);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LMS Model Benchmarking</h1>
            <p style="color: var(--text-muted);">Evaluation report containing latency, throughput, and code-generation correctiveness.</p>
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-label">Model Identifier</div>
                    <div class="meta-value" style="color: var(--accent);">{model_name}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">API Endpoint</div>
                    <div class="meta-value">{api_base}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Endpoint Protocol</div>
                    <div class="meta-value" style="text-transform: uppercase;">{api_type}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Generated At</div>
                    <div class="meta-value">{time.strftime('%Y-%m-%d %H:%M:%S local')}</div>
                </div>
            </div>
        </header>

        <!-- KPI summary -->
        <div class="metrics-grid">
            <div class="card">
                <div class="metric-title">Unit Test Pass Rate</div>
                <div class="metric-value metric-pass">{pass_rate:.1f}%</div>
                <div class="metric-desc">{passed_runs} of {total_runs} runs passed</div>
            </div>
            <div class="card">
                <div class="metric-title">Avg Generation Speed</div>
                <div class="metric-value metric-tps">{avg_tps:.1f} <span style="font-size: 1.2rem; font-weight: 500;">tok/s</span></div>
                <div class="metric-desc">Output token generation rate</div>
            </div>
            <div class="card">
                <div class="metric-title">Avg Latency (TTFT)</div>
                <div class="metric-value" style="color: #fbbf24;">{avg_ttft:.3f}s</div>
                <div class="metric-desc">Time to first token response</div>
            </div>
            <div class="card">
                <div class="metric-title">Avg Wall Time</div>
                <div class="metric-value" style="color: #a78bfa;">{avg_wall_time:.2f}s</div>
                <div class="metric-desc">Total request roundtrip duration</div>
            </div>
        </div>

        <!-- Visual Charts -->
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

        <!-- Task Summaries -->
        <div class="table-section">
            <div class="section-title">
                <span style="color: var(--accent);">📊</span> Task-Specific Performance
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Coding Task</th>
                        <th>Runs</th>
                        <th>Pass Rate</th>
                        <th>Avg Speed</th>
                        <th>Avg Latency</th>
                        <th>Avg Wall Time</th>
                    </tr>
                </thead>
                <tbody>
                    {task_rows_html_str}
                </tbody>
            </table>
        </div>

        <!-- Detailed Runs -->
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

        // Data injections for Chart.js
        const rawResults = {results_json};
        
        // Process data for Pass Rate Chart
        const tasksSummary = {{}};
        rawResults.forEach(r => {{
            if (!tasksSummary[r.task]) {{
                tasksSummary[r.task] = {{ passed: 0, total: 0 }};
            }}
            tasksSummary[r.task].total++;
            if (r.passed) tasksSummary[r.task].passed++;
        }});

        const taskLabels = Object.keys(tasksSummary);
        const passRates = taskLabels.map(t => (tasksSummary[t].passed / tasksSummary[t].total) * 100);

        // Render Pass Rate Chart
        const ctxPass = document.getElementById('passRateChart').getContext('2d');
        new Chart(ctxPass, {{
            type: 'bar',
            data: {{
                labels: taskLabels.map(l => l.replace('task_', '')),
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

        // Process data for Metrics Chart (Run speed + TTFT)
        const runIndices = rawResults.map((r, i) => `#${{i+1}} (${{r.task.substring(0, 10)}}...)`);
        const tpsValues = rawResults.map(r => r.tps);
        const ttftValues = rawResults.map(r => r.ttft);

        const ctxMetrics = document.getElementById('metricsChart').getContext('2d');
        new Chart(ctxMetrics, {{
            type: 'line',
            data: {{
                labels: runIndices,
                datasets: [
                    {{
                        label: 'Speed (tok/s)',
                        data: tpsValues,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
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
    args = parser.parse_args()

    print("=" * 60)
    print(" LMS MODEL BENCHMARK HARNESS")
    print("=" * 60)
    print(f"Target URL:    {args.api_base}")
    print(f"Model Name:    {args.model}")
    print(f"Protocol Type: {args.api_type}")
    print(f"Runs/Task:     {args.runs}")
    print("=" * 60)

    # Resolve bench directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bench_dir = os.path.join(base_dir, "bench")
    if not os.path.exists(bench_dir):
        print(f"Error: bench/ directory not found in {base_dir}")
        sys.exit(1)

    # Discover tasks
    tasks = sorted([d for d in os.listdir(bench_dir) if os.path.isdir(os.path.join(bench_dir, d))])
    if not tasks:
        print("No tasks found in bench/ directory.")
        sys.exit(0)

    print(f"Discovered {len(tasks)} tasks to evaluate.")
    
    results = []

    # Import html module locally for report escaping
    global html
    import html

    for task in tasks:
        task_path = os.path.join(bench_dir, task)
        prompt_file = os.path.join(task_path, "prompt.txt")
        if not os.path.exists(prompt_file):
            print(f"Skipping task '{task}' - prompt.txt is missing.")
            continue

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()

        for run_idx in range(1, args.runs + 1):
            print(f"\nEvaluating task '{task}' (Run {run_idx}/{args.runs})...")
            
            # 1. Query LLM
            try:
                response_text, ttft, wall_time, token_count = query_model(
                    args.api_base, args.api_type, args.model, prompt
                )
                tps = token_count / wall_time if wall_time > 0 else 0
                print(f"  API Response Received: TTFT={ttft:.3f}s, Time={wall_time:.2f}s, Token Count={token_count}, Speed={tps:.1f} tok/s")
            except Exception as e:
                print(f"  [ERROR] API Query failed: {e}")
                # Save failed run status
                results.append({
                    "task": task,
                    "run_index": run_idx,
                    "passed": False,
                    "ttft": 0.0,
                    "wall_time": 0.0,
                    "tps": 0.0,
                    "response_text": "",
                    "extracted_code": "",
                    "test_output": f"API Request Failed: {e}"
                })
                continue

            # 2. Extract code
            extracted_code = extract_python_code(response_text)
            
            # Save solution.py in the task directory
            solution_file = os.path.join(task_path, "solution.py")
            with open(solution_file, "w", encoding="utf-8") as f:
                f.write(extracted_code)

            # 3. Run unit tests
            print(f"  Executing unit tests via pytest...")
            passed, test_output = run_task_tests(task_path)
            
            status_str = "PASS" if passed else "FAIL"
            print(f"  Test Result: {status_str}")
            
            # Store results
            results.append({
                "task": task,
                "run_index": run_idx,
                "passed": passed,
                "ttft": ttft,
                "wall_time": wall_time,
                "tps": tps,
                "response_text": response_text,
                "extracted_code": extracted_code,
                "test_output": test_output
            })

    # Save raw results JSON
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate HTML report
    generate_html_report(results, args.api_base, args.model, args.api_type)
    
    print("\n" + "=" * 60)
    print(" BENCHMARK COMPLETE")
    print("=" * 60)
    passed_total = sum(1 for r in results if r["passed"])
    overall_rate = (passed_total / len(results) * 100) if results else 0
    print(f"Overall Pass Rate: {overall_rate:.1f}% ({passed_total}/{len(results)} runs)")
    print("Generated files:")
    print(" - benchmark_results.json (raw logs)")
    print(" - benchmark_report.html   (visual report summary)")
    print("=" * 60)

if __name__ == "__main__":
    main()
