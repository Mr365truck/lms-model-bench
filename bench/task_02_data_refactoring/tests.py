import pytest
import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))

def test_metrics_processor_sliding():
    from solution import MetricsProcessor
    
    proc = MetricsProcessor(10)
    
    r1 = proc.add_log('{"timestamp": 100, "latency": 10.0}')
    assert r1["count"] == 1
    assert r1["avg"] == 10.0
    assert r1["std"] == 0.0
    assert r1["anomalies"] == 0
    
    r2 = proc.add_log('{"timestamp": 102, "latency": 20.0}')
    assert r2["count"] == 2
    assert r2["avg"] == 15.0
    # std of [10, 20] is 7.071 (sample std) or 5.0 (population std)
    assert math.isclose(r2["std"], 7.071, abs_tol=1e-3) or math.isclose(r2["std"], 5.0, abs_tol=1e-3)
    
    r3 = proc.add_log('{"timestamp": 111, "latency": 30.0}')
    # window has 102 and 111 (count=2). 100 is evicted.
    assert r3["count"] == 2
    assert r3["avg"] == 25.0
    assert math.isclose(r3["std"], 7.071, abs_tol=1e-3) or math.isclose(r3["std"], 5.0, abs_tol=1e-3)

def test_metrics_processor_anomalies():
    from solution import MetricsProcessor
    
    proc = MetricsProcessor(100)
    
    # Add baseline values
    proc.add_log('{"timestamp": 10, "latency": 10.0}')
    proc.add_log('{"timestamp": 11, "latency": 10.0}')
    proc.add_log('{"timestamp": 12, "latency": 10.0}')
    proc.add_log('{"timestamp": 13, "latency": 10.0}')
    
    # Add a moderate outlier
    r = proc.add_log('{"timestamp": 14, "latency": 100.0}')
    # latencies: [10, 10, 10, 10, 100]. count = 5. avg = 28.0.
    # sample std = 40.249. 2 * std = 80.498.
    # 100 is at distance 72.0 from 28.0, which is < 80.498.
    # So anomaly count must be exactly 0.
    assert r["anomalies"] == 0
    
    # Add a massive outlier
    r2 = proc.add_log('{"timestamp": 15, "latency": 1000.0}')
    # latencies: [10, 10, 10, 10, 100, 1000]. count = 6. avg = 190.0.
    # sample std = 398.447. 2 * std = 796.894.
    # population std = 363.786. 2 * std = 727.572.
    # 1000 is at distance 810.0 from 190.0, which is > 2 * std (in both sample and population cases).
    # 100 is at distance 90.0 from 190.0, which is < 2 * std.
    # 10 is at distance 180.0 from 190.0, which is < 2 * std.
    # So anomaly count must be exactly 1.
    assert r2["anomalies"] == 1
