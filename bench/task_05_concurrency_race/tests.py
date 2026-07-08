import pytest
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))

def test_task_worker_basic():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=3)
    
    results = []
    lock = threading.Lock()
    
    def task_func(x):
        time.sleep(0.1)
        with lock:
            results.append(x)
        return x * 2

    def failing_func():
        time.sleep(0.1)
        raise ValueError("Failed intentionally")

    # Submit tasks
    worker.submit_task("t1", task_func, 10)
    worker.submit_task("t2", task_func, 20)
    worker.submit_task("t3", failing_func)
    
    # Start and wait
    worker.start()
    completed = worker.wait_completion(timeout=2.0)
    
    assert completed is True
    
    # Check results
    res_dict = worker.get_results()
    assert res_dict["t1"] == 20
    assert res_dict["t2"] == 40
    assert "t3" not in res_dict
    
    # Check errors
    err_dict = worker.get_errors()
    assert "t3" in err_dict
    assert isinstance(err_dict["t3"], ValueError)
    
    # Stop workers
    worker.stop()
