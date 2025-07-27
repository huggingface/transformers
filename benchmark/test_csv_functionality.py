#!/usr/bin/env python3
"""
Test script to verify that CSV export works correctly both with and without database connections.
"""

import logging
import sys
import os
from datetime import datetime

# Add the benchmark directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks_entrypoint import MetricsRecorder

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_csv_without_database():
    """Test CSV functionality without database connection"""
    logger.info("Testing CSV export without database connection...")
    
    # Create metrics recorder without database
    recorder = MetricsRecorder(None, logger, "test-repo", "test-branch", "abc123", "Test commit")
    
    # Add some test data
    benchmark_id = recorder.initialise_benchmark({"test_model": "test-model-v1", "gpu_name": "Test GPU"})
    
    # Add some device measurements
    for i in range(5):
        recorder.collect_device_measurements(benchmark_id, 50.0 + i, 1000.0 + i*10, 80.0 + i, 2000.0 + i*20)
    
    # Add model measurements
    recorder.collect_model_measurements(benchmark_id, {
        "model_load_time": 2.5,
        "inference_time": 0.1,
        "throughput": 100.5
    })
    
    # Export to CSV
    recorder.export_to_csv("test_results_no_db")
    recorder.close()
    
    logger.info("✓ CSV export without database connection completed successfully")

def test_csv_with_mock_database():
    """Test CSV functionality with a mock database connection (simulation)"""
    logger.info("Testing CSV export with database connection (simulated)...")
    
    # For this test, we'll create a recorder that thinks it has a database connection
    # but we'll mock it to avoid needing an actual database
    class MockConnection:
        def __init__(self):
            self.autocommit = True
        
        def cursor(self):
            return MockCursor()
        
        def close(self):
            pass
    
    class MockCursor:
        def __init__(self):
            self._counter = 1
        
        def execute(self, query, params=None):
            pass
        
        def fetchone(self):
            result = (self._counter,)
            self._counter += 1
            return result
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    # Create metrics recorder with mock database
    mock_conn = MockConnection()
    recorder = MetricsRecorder(mock_conn, logger, "test-repo", "test-branch", "def456", "Test commit with DB")
    
    # Add some test data
    benchmark_id = recorder.initialise_benchmark({"test_model": "test-model-v2", "gpu_name": "Test GPU Pro"})
    
    # Add some device measurements
    for i in range(3):
        recorder.collect_device_measurements(benchmark_id, 60.0 + i, 1200.0 + i*15, 85.0 + i, 2500.0 + i*25)
    
    # Add model measurements
    recorder.collect_model_measurements(benchmark_id, {
        "model_load_time": 3.2,
        "inference_time": 0.08,
        "throughput": 125.7
    })
    
    # Export to CSV
    recorder.export_to_csv("test_results_with_db")
    recorder.close()
    
    logger.info("✓ CSV export with database connection (simulated) completed successfully")

def main():
    logger.info("Starting CSV functionality tests...")
    
    try:
        test_csv_without_database()
        test_csv_with_mock_database()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 All tests completed successfully!")
        logger.info("Check the following directories for CSV output:")
        logger.info("  - test_results_no_db/")
        logger.info("  - test_results_with_db/")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 