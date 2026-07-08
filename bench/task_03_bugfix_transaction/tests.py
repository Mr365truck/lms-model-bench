import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def test_transaction_manager_basic():
    from solution import TransactionManager
    
    tm = TransactionManager(100.0, overdraft_limit=50.0)
    
    # Check invalid amount
    with pytest.raises(ValueError):
        tm.add_transaction("t0", 0.0)

    # Add valid transactions (deposits and withdrawals)
    tm.add_transaction("t1", 50.0)    # balance = 150
    tm.add_transaction("t2", -30.0)   # balance = 120
    assert tm.balance == 120.0

    # Rollback t2 (withdrawal)
    tm.rollback("t2")
    assert tm.balance == 150.0
    
    tm.commit()
    assert tm.balance == 150.0
    assert len(tm.transaction_history) == 1
    assert tm.transaction_history[0][0] == "t1"

def test_transaction_manager_atomic_commit():
    from solution import TransactionManager
    
    tm = TransactionManager(100.0, overdraft_limit=50.0)
    
    # Start: 100. Add t1: -120 (balance = -20). Add t2: -40 (balance = -60)
    # The final balance would be -60, which violates the overdraft limit of 50 (min balance is -50).
    tm.add_transaction("t1", -120.0)
    tm.add_transaction("t2", -40.0)
    assert tm.balance == -60.0

    # Commits should fail, raising ValueError, and it should revert the balance and pending transactions!
    with pytest.raises(ValueError):
        tm.commit()
        
    # Reverted state: balance should go back to 100.0, and pending transactions should be cleared
    assert tm.balance == 100.0
    assert len(tm.pending_transactions) == 0
    assert len(tm.transaction_history) == 0
