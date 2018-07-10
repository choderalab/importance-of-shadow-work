"""
Unit and regression test for the shadow_work package.
"""

# Import package, test suite, and other packages as needed
import shadow_work
import pytest
import sys

def test_shadow_work_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "shadow_work" in sys.modules
