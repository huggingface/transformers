#!/usr/bin/env python3
"""
Test script for GPU monitoring functionality.
This tests GPU monitoring behavior when GPUs are not available.
"""

import logging
import sys
import time

# Add current directory to path
sys.path.insert(0, '.')

from framework import GPUMonitor

def test_gpu_monitoring():
    """Test GPU monitoring with proper logging."""
    # Setup logging to see debug messages
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(name)s: %(message)s'
    )
    logger = logging.getLogger("test_gpu")
    
    print("=" * 60)
    print("TESTING GPU MONITORING")
    print("=" * 60)
    
    # Test 1: Create GPU monitor
    print("\n1. Creating GPU Monitor...")
    monitor = GPUMonitor(sample_interval=0.1, logger=logger)
    print(f"   GPU available: {monitor.gpu_available}")
    
    # Test 2: Start monitoring
    print("\n2. Starting GPU monitoring...")
    monitor.start()
    print("   Monitoring started")
    
    # Test 3: Let it run for a short time
    print("\n3. Running for 2 seconds...")
    time.sleep(2)
    
    # Test 4: Stop monitoring and get results
    print("\n4. Stopping monitoring and collecting results...")
    metrics = monitor.stop()
    print("   Monitoring stopped")
    
    # Test 5: Display results
    print("\n5. GPU Monitoring Results:")
    print("   " + "-" * 40)
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    print("   " + "-" * 40)
    
    # Test 6: Verify expected behavior
    print("\n6. Verification:")
    if monitor.gpu_available:
        print("   ✓ GPUs detected - monitoring should have collected data")
        expected_keys = ["gpu_utilization_mean", "sample_count", "gpu_monitoring_status"]
        if all(key in metrics for key in expected_keys):
            print("   ✓ All expected metrics present")
        else:
            print("   ✗ Missing expected metrics")
    else:
        print("   ✓ No GPUs detected - this is expected in this environment")
        if metrics.get("gpu_monitoring_status") == "disabled":
            print("   ✓ GPU monitoring correctly disabled")
        else:
            print(f"   ✗ Unexpected status: {metrics.get('gpu_monitoring_status')}")
    
    return metrics

def test_manual_gpustat():
    """Test gpustat directly to understand the failure mode."""
    print("\n" + "=" * 60)
    print("TESTING GPUSTAT DIRECTLY")
    print("=" * 60)
    
    try:
        import gpustat
        print("✓ gpustat imported successfully")
        
        print("\nTrying to query GPU stats...")
        gpu_stats = gpustat.GPUStatCollection.new_query()
        
        if gpu_stats:
            print(f"✓ GPU stats returned: {len(gpu_stats)} GPU(s)")
            for i, gpu in enumerate(gpu_stats):
                print(f"  GPU {i}: {gpu}")
        else:
            print("⚠ GPU stats returned None")
            
    except Exception as e:
        print(f"✗ gpustat failed: {e}")
        print(f"  Error type: {type(e).__name__}")

if __name__ == "__main__":
    print("Testing GPU monitoring functionality...")
    
    # Test direct gpustat usage
    test_manual_gpustat()
    
    # Test our GPU monitoring wrapper
    metrics = test_gpu_monitoring()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if "gpu_monitoring_status" in metrics:
        status = metrics["gpu_monitoring_status"]
        print(f"GPU Monitoring Status: {status}")
        
        if status == "disabled":
            print("✓ GPU monitoring correctly disabled (no GPUs available)")
        elif status == "success":
            print("✓ GPU monitoring worked successfully")
        elif status == "failed":
            reason = metrics.get("gpu_monitoring_reason", "unknown")
            print(f"⚠ GPU monitoring failed: {reason}")
        else:
            print(f"? Unknown status: {status}")
    else:
        print("✗ No GPU monitoring status returned")
    
    print("\nThis test demonstrates how GPU monitoring behaves")
    print("when GPUs are not available in the environment.") 