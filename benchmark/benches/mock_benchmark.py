#!/usr/bin/env python3
"""
Mock benchmark for testing CSV export functionality.
This benchmark simulates realistic data collection without requiring GPU or other dependencies.
"""

import time
import random
import math
from logging import Logger
from threading import Event, Thread
import sys
import os

# Add the parent directory to Python path to import benchmarks_entrypoint
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks_entrypoint import MetricsRecorder

def simulate_device_monitoring(benchmark_id, continue_monitoring, metrics_recorder, duration_seconds=10):
    """
    Simulate device monitoring by generating realistic CPU/GPU usage patterns
    """
    start_time = time.time()
    measurement_count = 0
    
    while not continue_monitoring.is_set() and (time.time() - start_time) < duration_seconds:
        # Simulate realistic but varying system metrics
        base_cpu = 30.0
        base_memory = 2048.0
        base_gpu = 60.0
        base_gpu_memory = 4096.0
        
        # Add some realistic variation and trends
        elapsed = time.time() - start_time
        cpu_util = base_cpu + random.uniform(-10, 20) + (elapsed * 2)  # CPU increases over time
        mem_megabytes = base_memory + random.uniform(-200, 500) + (measurement_count * 10)
        gpu_util = base_gpu + random.uniform(-15, 25) + (5 * math.sin(elapsed))  # Oscillating GPU usage
        gpu_mem_megabytes = base_gpu_memory + random.uniform(-500, 1000) + (measurement_count * 20)
        
        # Clamp values to realistic ranges
        cpu_util = max(0, min(100, cpu_util))
        gpu_util = max(0, min(100, gpu_util))
        mem_megabytes = max(500, mem_megabytes)
        gpu_mem_megabytes = max(1000, gpu_mem_megabytes)
        
        metrics_recorder.collect_device_measurements(
            benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes
        )
        
        measurement_count += 1
        time.sleep(0.1)  # Collect measurements every 100ms

def run_benchmark(
    logger: Logger, repository: str, branch: str, commit_id: str, commit_msg: str, 
    metrics_recorder=None, simulation_duration=15
):
    """
    Mock benchmark that simulates model loading, inference, and performance measurement.
    
    Args:
        logger: Logger instance
        repository: Repository name
        branch: Branch name  
        commit_id: Commit ID
        commit_msg: Commit message
        metrics_recorder: Optional pre-created metrics recorder
        simulation_duration: How long to simulate benchmark (seconds)
    """
    continue_monitoring = Event()
    monitoring_thread = None
    model_name = "mock-transformer-7b"
    
    # If no metrics_recorder is provided, create one for backward compatibility
    if metrics_recorder is None:
        try:
            import psycopg2
            connection = psycopg2.connect("dbname=metrics")
            metrics_recorder = MetricsRecorder(connection, logger, repository, branch, commit_id, commit_msg, True)
            should_close_recorder = True
        except Exception as e:
            logger.warning(f"Could not connect to database: {e}. Using CSV-only mode.")
            metrics_recorder = MetricsRecorder(None, logger, repository, branch, commit_id, commit_msg, True)
            should_close_recorder = True
    else:
        should_close_recorder = False

    try:
        logger.info(f"Starting mock benchmark for {model_name}")
        
        # Initialize benchmark
        benchmark_id = metrics_recorder.initialise_benchmark({
            "model_name": model_name,
            "gpu_name": "Mock GPU RTX 4090",
            "benchmark_type": "mock_inference_test",
            "simulation_duration_seconds": simulation_duration
        })
        
        logger.info(f"Mock benchmark #{benchmark_id} initialized")
        
        # Start device monitoring in background
        monitoring_thread = Thread(
            target=simulate_device_monitoring,
            args=[benchmark_id, continue_monitoring, metrics_recorder, simulation_duration]
        )
        monitoring_thread.start()
        logger.info("Started background device monitoring")
        
        # Simulate model loading
        logger.info("Simulating model loading...")
        time.sleep(1.0)  # Simulate download/load time
        model_load_start = time.time()
        time.sleep(random.uniform(1.5, 3.0))  # Variable load time
        model_load_time = time.time() - model_load_start
        logger.info(f"Mock model loaded in {model_load_time:.2f}s")
        
        # Simulate first eager forward pass
        logger.info("Simulating first eager forward pass...")
        start = time.time()
        time.sleep(random.uniform(0.3, 0.7))
        first_eager_fwd_pass_time = time.time() - start
        logger.info(f"First eager forward pass: {first_eager_fwd_pass_time:.3f}s")
        
        # Simulate second eager forward pass (should be faster)
        logger.info("Simulating second eager forward pass...")
        start = time.time()
        time.sleep(random.uniform(0.2, 0.5))
        second_eager_fwd_pass_time = time.time() - start
        logger.info(f"Second eager forward pass: {second_eager_fwd_pass_time:.3f}s")
        
        # Simulate first eager generation
        logger.info("Simulating first eager generation...")
        start = time.time()
        time.sleep(random.uniform(1.0, 2.0))
        first_eager_generate_time = time.time() - start
        logger.info(f"First eager generation: {first_eager_generate_time:.3f}s")
        
        # Simulate second eager generation (should be faster)
        logger.info("Simulating second eager generation...")
        start = time.time()
        time.sleep(random.uniform(0.8, 1.5))
        second_eager_generate_time = time.time() - start
        logger.info(f"Second eager generation: {second_eager_generate_time:.3f}s")
        
        # Simulate token generation timings
        logger.info("Simulating token generation timings...")
        time_to_first_token = random.uniform(0.1, 0.3)
        time_to_second_token = random.uniform(0.05, 0.15)
        time_to_third_token = random.uniform(0.04, 0.12)
        mean_time_to_next_token = random.uniform(0.03, 0.08)
        
        # Simulate compile generation runs
        logger.info("Simulating compile generation runs...")
        
        # First compile run (slowest)
        start = time.time()
        time.sleep(random.uniform(2.0, 4.0))
        first_compile_generate_time = time.time() - start
        logger.info(f"First compile generation: {first_compile_generate_time:.3f}s")
        
        # Second compile run (faster)
        start = time.time()
        time.sleep(random.uniform(1.5, 2.5))
        second_compile_generate_time = time.time() - start
        logger.info(f"Second compile generation: {second_compile_generate_time:.3f}s")
        
        # Third compile run (even faster)
        start = time.time()
        time.sleep(random.uniform(1.0, 2.0))
        third_compile_generate_time = time.time() - start
        logger.info(f"Third compile generation: {third_compile_generate_time:.3f}s")
        
        # Fourth compile run (fastest)
        start = time.time()
        time.sleep(random.uniform(0.8, 1.5))
        fourth_compile_generate_time = time.time() - start
        logger.info(f"Fourth compile generation: {fourth_compile_generate_time:.3f}s")
        
        logger.info("Benchmark completed, collecting final measurements...")
        
        # Collect model measurements using the same schema as llama.py
        metrics_recorder.collect_model_measurements(benchmark_id, {
            "model_load_time": model_load_time,
            "first_eager_forward_pass_time_secs": first_eager_fwd_pass_time,
            "second_eager_forward_pass_time_secs": second_eager_fwd_pass_time,
            "first_eager_generate_time_secs": first_eager_generate_time,
            "second_eager_generate_time_secs": second_eager_generate_time,
            "time_to_first_token_secs": time_to_first_token,
            "time_to_second_token_secs": time_to_second_token,
            "time_to_third_token_secs": time_to_third_token,
            "time_to_next_token_mean_secs": mean_time_to_next_token,
            "first_compile_generate_time_secs": first_compile_generate_time,
            "second_compile_generate_time_secs": second_compile_generate_time,
            "third_compile_generate_time_secs": third_compile_generate_time,
            "fourth_compile_generate_time_secs": fourth_compile_generate_time,
        })
        
        logger.info(f"Mock benchmark completed successfully!")
        logger.info(f"Results summary:")
        logger.info(f"  - Model load time: {model_load_time:.2f}s")
        logger.info(f"  - First eager forward pass: {first_eager_fwd_pass_time:.3f}s")
        logger.info(f"  - First eager generation: {first_eager_generate_time:.3f}s")
        logger.info(f"  - Mean time to next token: {mean_time_to_next_token:.3f}s")
        
    except Exception as e:
        logger.error(f"Mock benchmark failed: {e}")
    finally:
        # Stop monitoring
        continue_monitoring.set()
        if monitoring_thread is not None:
            monitoring_thread.join()
            logger.info("Device monitoring stopped")
        
        # Only close the recorder if we created it locally
        if should_close_recorder:
            metrics_recorder.close()

if __name__ == "__main__":
    # Allow running the mock benchmark directly for testing
    import logging
    
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    
    print("Running mock benchmark directly...")
    run_benchmark(logger, "test-repo", "main", "mock123", "Mock benchmark test")
    print("Mock benchmark completed!") 