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
        time.sleep(0.05)
        with lock:
            results.append(x)
        return x * 2

    def failing_func():
        time.sleep(0.05)
        raise ValueError("Failed intentionally")

    worker.submit_task("t1", task_func, 10)
    worker.submit_task("t2", task_func, 20)
    worker.submit_task("t3", failing_func)
    
    worker.start()
    completed = worker.wait_completion(timeout=2.0)
    assert completed is True
    
    res_dict = worker.get_results()
    assert res_dict["t1"] == 20
    assert res_dict["t2"] == 40
    assert "t3" not in res_dict
    
    err_dict = worker.get_errors()
    assert "t3" in err_dict
    assert isinstance(err_dict["t3"], ValueError)
    
    worker.stop()

def test_submit_after_start():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=2)
    worker.start()
    
    # Submit after start is called
    worker.submit_task("t1", lambda: 42)
    
    completed = worker.wait_completion(timeout=1.0)
    assert completed is True
    assert worker.get_results()["t1"] == 42
    
    worker.stop()

def test_start_twice():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=2)
    worker.start()
    
    # Starting twice should not crash or double spawn threads (leak check)
    try:
        worker.start()
    except Exception as e:
        pytest.fail(f"Calling start() twice raised an exception: {e}")
        
    worker.submit_task("t1", lambda: 1)
    worker.wait_completion(timeout=1.0)
    assert worker.get_results()["t1"] == 1
    
    worker.stop()

def test_wait_before_tasks():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=2)
    # wait_completion before starting or submitting anything should return True immediately
    assert worker.wait_completion(timeout=1.0) is True
    
    worker.start()
    assert worker.wait_completion(timeout=1.0) is True
    
    worker.stop()

def test_stop_with_pending_work():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=2)
    
    # Submit several slow tasks
    def slow_task():
        time.sleep(1.0)
        return 1
        
    for i in range(10):
        worker.submit_task(f"slow_{i}", slow_task)
        
    worker.start()
    # Sleep slightly so workers pick up a couple of tasks, then stop immediately
    time.sleep(0.05)
    
    # stop() should terminate threads quickly without hanging, even though tasks are in queue
    start_stop = time.time()
    worker.stop()
    stop_duration = time.time() - start_stop
    
    # Should stop in under 0.5s (should not block for 10 seconds!)
    assert stop_duration < 0.5

def test_timeout_expired():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=1)
    
    def slow_task():
        time.sleep(0.5)
        return 1
        
    worker.submit_task("slow", slow_task)
    worker.start()
    
    # Wait with a short timeout. Since task takes 0.5s, wait_completion(0.05) must return False
    completed = worker.wait_completion(timeout=0.05)
    assert completed is False
    
    worker.wait_completion(timeout=1.0)
    worker.stop()

def test_duplicate_task_ids():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=2)
    
    # Submit tasks with duplicate IDs. The later one should replace or overwrite, or execute.
    # What's important is the results dict handles it cleanly (either runs both and overrides key, or logs it).
    worker.submit_task("dup", lambda: "first")
    worker.submit_task("dup", lambda: "second")
    
    worker.start()
    worker.wait_completion(timeout=1.0)
    
    # The final value in results should be either "first" or "second" (typically "second")
    assert worker.get_results()["dup"] in ["first", "second"]
    
    worker.stop()

def test_stress_many_tasks():
    from solution import ThreadSafeTaskWorker
    
    num_tasks = 150
    worker = ThreadSafeTaskWorker(num_workers=10)
    
    def add_one(x):
        return x + 1
        
    for i in range(num_tasks):
        worker.submit_task(f"t_{i}", add_one, i)
        
    worker.start()
    completed = worker.wait_completion(timeout=3.0)
    assert completed is True
    
    results = worker.get_results()
    assert len(results) == num_tasks
    for i in range(num_tasks):
        assert results[f"t_{i}"] == i + 1
        
    worker.stop()
