import argparse
import importlib.util
import logging
import os
import sys
import csv
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List

try:
    from psycopg2.extensions import register_adapter
    from psycopg2.extras import Json
    register_adapter(dict, Json)
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


register_adapter(dict, Json)


class ImportModuleException(Exception):
    pass


class MetricsRecorder:
    def __init__(
        self, connection, logger: logging.Logger, repository: str, branch: str, commit_id: str, commit_msg: str
    ):
        self.conn = connection
        self.use_database = connection is not None
        if self.use_database:
            self.conn.autocommit = True
        self.logger = logger
        self.repository = repository
        self.branch = branch
        self.commit_id = commit_id
        self.commit_msg = commit_msg
        self.current_benchmark_id = 0
        
        # For CSV export - store all data in memory
        self.csv_data = {
            'benchmarks': [],
            'device_measurements': [],
            'model_measurements': []
        }

    def initialise_benchmark(self, metadata: dict[str, str]) -> int:
        """
        Creates a new benchmark, returns the benchmark id
        """
        if self.use_database:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO benchmarks (repository, branch, commit_id, commit_message, metadata) VALUES (%s, %s, %s, %s, %s) RETURNING benchmark_id",
                    (self.repository, self.branch, self.commit_id, self.commit_msg, metadata),
                )
                benchmark_id = cur.fetchone()[0]
                self.logger.debug(f"initialised benchmark #{benchmark_id}")
                return benchmark_id
        else:
            # Generate a unique benchmark ID for CSV mode
            self.current_benchmark_id += 1
            benchmark_id = self.current_benchmark_id
            
            # Store benchmark data for CSV export
            self.csv_data['benchmarks'].append({
                'benchmark_id': benchmark_id,
                'repository': self.repository,
                'branch': self.branch,
                'commit_id': self.commit_id,
                'commit_message': self.commit_msg,
                'metadata': json.dumps(metadata),
                'created_at': datetime.utcnow().isoformat()
            })
            
            self.logger.debug(f"initialised benchmark #{benchmark_id} (CSV mode)")
            return benchmark_id

    def collect_device_measurements(self, benchmark_id: int, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes):
        """
        Collect device metrics, such as CPU & GPU usage. These are "static", as in you cannot pass arbitrary arguments to the function.
        """
        # Always store device measurements for CSV export
        self.csv_data['device_measurements'].append({
            'benchmark_id': benchmark_id,
            'cpu_util': cpu_util,
            'mem_megabytes': mem_megabytes,
            'gpu_util': gpu_util,
            'gpu_mem_megabytes': gpu_mem_megabytes,
            'time': datetime.utcnow().isoformat()
        })
        
        # Also store in database if available
        if self.use_database:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO device_measurements (benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes) VALUES (%s, %s, %s, %s, %s)",
                    (benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes),
                )
            
        self.logger.debug(
            f"collected device measurements for benchmark #{benchmark_id} [CPU util: {cpu_util}, mem MBs: {mem_megabytes}, GPU util: {gpu_util}, GPU mem MBs: {gpu_mem_megabytes}]"
        )

    def collect_model_measurements(self, benchmark_id: int, measurements: dict[str, float]):
        # Always store model measurements for CSV export
        self.csv_data['model_measurements'].append({
            'benchmark_id': benchmark_id,
            'measurements': json.dumps(measurements),
            'time': datetime.utcnow().isoformat()
        })
        
        # Also store in database if available
        if self.use_database:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_measurements (
                        benchmark_id,
                        measurements
                    ) VALUES (%s, %s)
                    """,
                    (
                        benchmark_id,
                        measurements,
                    ),
                )
            
        self.logger.debug(f"collected model measurements for benchmark #{benchmark_id}: {measurements}")

    def export_to_csv(self, output_dir: str = "benchmark_results"):
        """
        Export all collected data to CSV files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export benchmarks
        if self.csv_data['benchmarks']:
            benchmarks_file = os.path.join(output_dir, f"benchmarks_{timestamp}.csv")
            with open(benchmarks_file, 'w', newline='') as csvfile:
                fieldnames = ['benchmark_id', 'repository', 'branch', 'commit_id', 'commit_message', 'metadata', 'created_at']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.csv_data['benchmarks'])
            self.logger.info(f"Exported benchmarks to {benchmarks_file}")
        
        # Export device measurements
        if self.csv_data['device_measurements']:
            device_file = os.path.join(output_dir, f"device_measurements_{timestamp}.csv")
            with open(device_file, 'w', newline='') as csvfile:
                fieldnames = ['benchmark_id', 'cpu_util', 'mem_megabytes', 'gpu_util', 'gpu_mem_megabytes', 'time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.csv_data['device_measurements'])
            self.logger.info(f"Exported device measurements to {device_file}")
        
        # Export model measurements (flattened)
        if self.csv_data['model_measurements']:
            model_file = os.path.join(output_dir, f"model_measurements_{timestamp}.csv")
            with open(model_file, 'w', newline='') as csvfile:
                # Flatten the measurements JSON for easier CSV reading
                flattened_data = []
                for record in self.csv_data['model_measurements']:
                    measurements = json.loads(record['measurements'])
                    row = {
                        'benchmark_id': record['benchmark_id'],
                        'time': record['time']
                    }
                    row.update(measurements)
                    flattened_data.append(row)
                
                if flattened_data:
                    fieldnames = list(flattened_data[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flattened_data)
                self.logger.info(f"Exported model measurements to {model_file}")
        
        # Create a summary file
        summary_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
        self._create_summary_csv(summary_file)
        
    def _create_summary_csv(self, summary_file: str):
        """
        Create a summary CSV that combines all benchmark data into a single comprehensive view
        """
        summary_data = []
        
        for benchmark in self.csv_data['benchmarks']:
            benchmark_id = benchmark['benchmark_id']
            
            # Get model measurements for this benchmark
            model_measurements = {}
            for model_record in self.csv_data['model_measurements']:
                if model_record['benchmark_id'] == benchmark_id:
                    model_measurements.update(json.loads(model_record['measurements']))
            
            # Calculate device measurement aggregates
            device_stats = {
                'avg_cpu_util': 0,
                'max_cpu_util': 0,
                'avg_mem_megabytes': 0,
                'max_mem_megabytes': 0,
                'avg_gpu_util': 0,
                'max_gpu_util': 0,
                'avg_gpu_mem_megabytes': 0,
                'max_gpu_mem_megabytes': 0,
                'device_measurement_count': 0
            }
            
            device_measurements = [d for d in self.csv_data['device_measurements'] if d['benchmark_id'] == benchmark_id]
            if device_measurements:
                cpu_utils = [d['cpu_util'] for d in device_measurements if d['cpu_util'] is not None]
                mem_values = [d['mem_megabytes'] for d in device_measurements if d['mem_megabytes'] is not None]
                gpu_utils = [d['gpu_util'] for d in device_measurements if d['gpu_util'] is not None]
                gpu_mems = [d['gpu_mem_megabytes'] for d in device_measurements if d['gpu_mem_megabytes'] is not None]
                
                device_stats.update({
                    'avg_cpu_util': sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0,
                    'max_cpu_util': max(cpu_utils) if cpu_utils else 0,
                    'avg_mem_megabytes': sum(mem_values) / len(mem_values) if mem_values else 0,
                    'max_mem_megabytes': max(mem_values) if mem_values else 0,
                    'avg_gpu_util': sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                    'max_gpu_util': max(gpu_utils) if gpu_utils else 0,
                    'avg_gpu_mem_megabytes': sum(gpu_mems) / len(gpu_mems) if gpu_mems else 0,
                    'max_gpu_mem_megabytes': max(gpu_mems) if gpu_mems else 0,
                    'device_measurement_count': len(device_measurements)
                })
            
            # Combine all data into summary row
            summary_row = {**benchmark, **model_measurements, **device_stats}
            summary_data.append(summary_row)
        
        if summary_data:
            with open(summary_file, 'w', newline='') as csvfile:
                fieldnames = list(summary_data[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
            self.logger.info(f"Created benchmark summary at {summary_file}")

    def close(self):
        if self.use_database and self.conn:
            self.conn.close()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s - %(asctime)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments() -> tuple[str, str, str, str]:
    """
    Parse command line arguments for the benchmarking CLI.
    """
    parser = argparse.ArgumentParser(description="CLI for benchmarking the huggingface/transformers.")

    parser.add_argument(
        "repository",
        type=str,
        help="The repository name on which the benchmarking is performed.",
    )

    parser.add_argument(
        "branch",
        type=str,
        help="The branch name on which the benchmarking is performed.",
    )

    parser.add_argument(
        "commit_id",
        type=str,
        help="The commit hash on which the benchmarking is performed.",
    )

    parser.add_argument(
        "commit_msg",
        type=str,
        help="The commit message associated with the commit, truncated to 70 characters.",
    )

    args = parser.parse_args()

    return args.repository, args.branch, args.commit_id, args.commit_msg


def import_from_path(module_name, file_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise ImportModuleException(f"failed to load python module: {e}")


def create_database_connection():
    """
    Try to create a database connection. Returns None if connection fails.
    """
    if not PSYCOPG2_AVAILABLE:
        logger.warning("psycopg2 not available - running in CSV-only mode")
        return None
        
    try:
        import psycopg2
        conn = psycopg2.connect("dbname=metrics")
        logger.info("Successfully connected to database")
        return conn
    except Exception as e:
        logger.warning(f"Failed to connect to database: {e}. Running in CSV-only mode")
        return None


def create_global_metrics_recorder(repository: str, branch: str, commit_id: str, commit_msg: str) -> MetricsRecorder:
    """
    Create a global metrics recorder that will be used across all benchmarks.
    """
    connection = create_database_connection()
    recorder = MetricsRecorder(connection, logger, repository, branch, commit_id, commit_msg)
    
    if connection is None:
        logger.info("Running benchmarks in CSV-only mode (no database connection)")
    else:
        logger.info("Running benchmarks with both database storage and CSV export")
    
    return recorder


if __name__ == "__main__":
    benchmarks_folder_path = os.path.dirname(os.path.realpath(__file__))

    repository, branch, commit_id, commit_msg = parse_arguments()
    
    # Create a global metrics recorder
    global_metrics_recorder = create_global_metrics_recorder(repository, branch, commit_id, commit_msg)
    
    successful_benchmarks = 0
    failed_benchmarks = 0

    for entry in os.scandir(benchmarks_folder_path):
        try:
            if not entry.name.endswith(".py"):
                continue
            if entry.path == __file__:
                continue
            logger.debug(f"loading: {entry.name}")
            module = import_from_path(entry.name.split(".")[0], entry.path)
            logger.info(f"running benchmarks in: {entry.name}")
            
            # Check if the module has an updated run_benchmark function that accepts metrics_recorder
            try:
                # Try the new signature first
                module.run_benchmark(logger, repository, branch, commit_id, commit_msg, global_metrics_recorder)
            except TypeError:
                # Fall back to the old signature for backward compatibility
                logger.warning(f"Module {entry.name} using old run_benchmark signature - database connection will be created per module")
                module.run_benchmark(logger, repository, branch, commit_id, commit_msg)
            
            successful_benchmarks += 1
        except ImportModuleException as e:
            logger.error(e)
            failed_benchmarks += 1
        except Exception as e:
            logger.error(f"error running benchmarks for {entry.name}: {e}")
            failed_benchmarks += 1

    # Export CSV results at the end (always generated regardless of database connection)
    try:
        global_metrics_recorder.export_to_csv()
        logger.info(f"Benchmark run completed. Successful: {successful_benchmarks}, Failed: {failed_benchmarks}")
        logger.info("CSV reports have been generated and saved to the benchmark_results directory")
    except Exception as e:
        logger.error(f"Failed to export CSV results: {e}")
    finally:
        global_metrics_recorder.close()
