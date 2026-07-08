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

def test_duplicate_task_id():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=2)
    worker.submit_task("dup", lambda: 1)
    
    # Submitting duplicate ID should raise ValueError
    with pytest.raises(ValueError):
        worker.submit_task("dup", lambda: 2)
        
    worker.start()
    worker.wait_completion(timeout=1.0)
    
    # Check that subsequent submit_task of completed task ID also raises ValueError
    with pytest.raises(ValueError):
        worker.submit_task("dup", lambda: 3)
        
    worker.stop()

def test_submit_after_stop():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=2)
    worker.start()
    worker.stop()
    
    # Submitting task after stop() must raise RuntimeError
    with pytest.raises(RuntimeError):
        worker.submit_task("post_stop", lambda: 1)

def test_start_twice():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=2)
    worker.start()
    
    # Starting twice should be a no-op
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
    assert worker.wait_completion(timeout=1.0) is True
    
    worker.start()
    assert worker.wait_completion(timeout=1.0) is True
    
    worker.stop()

def test_stop_semantics():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=2)

    started = [threading.Event(), threading.Event()]
    queued_executed = []
    queued_lock = threading.Lock()

    # 2 workers. Submit 2 slow tasks and wait until both are actually active
    # before calling stop(), so this test does not depend on scheduler timing.
    def slow_task(index):
        started[index].set()
        time.sleep(0.3)
        return 100
        
    def normal_task():
        with queued_lock:
            queued_executed.append(True)
        return 200
        
    worker.submit_task("active_1", slow_task, 0)
    worker.submit_task("active_2", slow_task, 1)
    
    for i in range(8):
        worker.submit_task(f"queued_{i}", normal_task)
        
    worker.start()
    assert started[0].wait(timeout=1.0)
    assert started[1].wait(timeout=1.0)
    
    # stop() should clear/cancel the 8 queued tasks and wait for the 2 active tasks to finish
    start_time = time.time()
    worker.stop()
    stop_duration = time.time() - start_time
    
    # stop_duration should reflect the active tasks and should NOT wait for the
    # other 8 queued tasks to execute.
    assert stop_duration < 0.5
    
    # Verify results: active_1 and active_2 have results, the others are cancelled
    results = worker.get_results()
    assert results["active_1"] == 100
    assert results["active_2"] == 100
    assert queued_executed == []
    
    for i in range(8):
        assert f"queued_{i}" not in results

def test_timeout_expired():
    from solution import ThreadSafeTaskWorker
    
    worker = ThreadSafeTaskWorker(num_workers=1)
    
    def slow_task():
        time.sleep(0.4)
        return 1
        
    worker.submit_task("slow", slow_task)
    worker.start()
    
    completed = worker.wait_completion(timeout=0.05)
    assert completed is False
    
    worker.wait_completion(timeout=1.0)
    worker.stop()

def test_stress_many_tasks():
    from solution import ThreadSafeTaskWorker
    
    num_tasks = 100
    worker = ThreadSafeTaskWorker(num_workers=5)
    
    def add_one(x):
        return x + 1
        
    for i in range(num_tasks):
        worker.submit_task(f"t_{i}", add_one, i)
        
    worker.start()
    completed = worker.wait_completion(timeout=2.0)
    assert completed is True
    
    results = worker.get_results()
    assert len(results) == num_tasks
    for i in range(num_tasks):
        assert results[f"t_{i}"] == i + 1
        
    worker.stop()
