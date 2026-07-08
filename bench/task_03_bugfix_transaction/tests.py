import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def test_transaction_manager_deferred_balance():
    from solution import TransactionManager
    
    tm = TransactionManager(100.0, overdraft_limit=50.0)
    
    # Check invalid amount
    with pytest.raises(ValueError):
        tm.add_transaction("t0", 0.0)

    # Add transaction: should NOT mutate balance immediately
    tm.add_transaction("t1", 50.0)
    tm.add_transaction("t2", -30.0)
    assert tm.balance == 100.0  # Deferred balance: still initial balance!

    # Rollback t2 (pending)
    tm.rollback("t2")
    assert tm.balance == 100.0
    
    # Commit: should apply t1 (50.0) -> balance becomes 150.0
    tm.commit()
    assert tm.balance == 150.0
    assert len(tm.transaction_history) == 1
    assert tm.transaction_history[0] == ("t1", 50.0)

def test_transaction_manager_atomic_commit():
    from solution import TransactionManager
    
    tm = TransactionManager(150.0, overdraft_limit=50.0)
    
    # Add transactions that would drop the balance below the overdraft limit:
    # 150.0 + (-220.0) = -70.0 (overdraft limit is 50.0, i.e., min balance is -50.0)
    tm.add_transaction("t3", -220.0)
    assert tm.balance == 150.0  # Still 150.0 before commit

    # Commit must fail with ValueError, and discard/revert all pending transactions
    with pytest.raises(ValueError):
        tm.commit()
        
    # Reverted state: balance remains 150.0, pending transactions discarded
    assert tm.balance == 150.0
    assert len(tm.pending_transactions) == 0
    assert len(tm.transaction_history) == 0
