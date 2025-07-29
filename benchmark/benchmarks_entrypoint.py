import argparse
import importlib.util
import logging
import os
import sys
import csv
import json
import uuid
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
        self, connection, logger: logging.Logger, repository: str, branch: str, commit_id: str, commit_msg: str, 
        collect_csv_data: bool = True
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
        self.collect_csv_data = collect_csv_data
        
        # For CSV export - store all data in memory (only if CSV collection is enabled)
        if self.collect_csv_data:
            self.csv_data = {
                'benchmarks': [],
                'device_measurements': [],
                'model_measurements': []
            }
        else:
            self.csv_data = None

    def initialise_benchmark(self, metadata: dict[str, str]) -> str:
        """
        Creates a new benchmark, returns the benchmark id (UUID)
        """
        # Generate a unique UUID for this benchmark
        benchmark_id = str(uuid.uuid4())
        
        if self.use_database:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO benchmarks (benchmark_id, repository, branch, commit_id, commit_message, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                    (benchmark_id, self.repository, self.branch, self.commit_id, self.commit_msg, metadata),
                )
                self.logger.debug(f"initialised benchmark #{benchmark_id}")
        
        # Store benchmark data for CSV export (if enabled)
        if self.collect_csv_data:
            self.csv_data['benchmarks'].append({
                'benchmark_id': benchmark_id,
                'repository': self.repository,
                'branch': self.branch,
                'commit_id': self.commit_id,
                'commit_message': self.commit_msg,
                'metadata': json.dumps(metadata),
                'created_at': datetime.utcnow().isoformat()
            })
            
        mode_info = []
        if self.use_database:
            mode_info.append("database")
        if self.collect_csv_data:
            mode_info.append("CSV")
        mode_str = " + ".join(mode_info) if mode_info else "no storage"
        
        self.logger.debug(f"initialised benchmark #{benchmark_id} ({mode_str} mode)")
        return benchmark_id

    def collect_device_measurements(self, benchmark_id: str, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes):
        """
        Collect device metrics, such as CPU & GPU usage. These are "static", as in you cannot pass arbitrary arguments to the function.
        """
        # Store device measurements for CSV export (if enabled)
        if self.collect_csv_data:
            self.csv_data['device_measurements'].append({
                'benchmark_id': benchmark_id,
                'cpu_util': cpu_util,
                'mem_megabytes': mem_megabytes,
                'gpu_util': gpu_util,
                'gpu_mem_megabytes': gpu_mem_megabytes,
                'time': datetime.utcnow().isoformat()
            })
        
        # Store in database if available
        if self.use_database:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO device_measurements (benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes) VALUES (%s, %s, %s, %s, %s)",
                    (benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes),
                )
            
        self.logger.debug(
            f"collected device measurements for benchmark #{benchmark_id} [CPU util: {cpu_util}, mem MBs: {mem_megabytes}, GPU util: {gpu_util}, GPU mem MBs: {gpu_mem_megabytes}]"
        )

    def collect_model_measurements(self, benchmark_id: str, measurements: dict[str, float]):
        # Store model measurements for CSV export (if enabled)
        if self.collect_csv_data:
            self.csv_data['model_measurements'].append({
                'benchmark_id': benchmark_id,
                'measurements': json.dumps(measurements),
                'time': datetime.utcnow().isoformat()
            })
        
        # Store in database if available
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
        if not self.collect_csv_data:
            self.logger.warning("CSV data collection is disabled - no CSV files will be generated")
            return
            
        if self.csv_data is None:
            self.logger.error("No CSV data available for export")
            return
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_created = []
        
        # Export benchmarks (always create file, even if empty)
        benchmarks_file = os.path.join(output_dir, f"benchmarks_{timestamp}.csv")
        with open(benchmarks_file, 'w', newline='') as csvfile:
            fieldnames = ['benchmark_id', 'repository', 'branch', 'commit_id', 'commit_message', 'metadata', 'created_at']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            if self.csv_data['benchmarks']:
                writer.writerows(self.csv_data['benchmarks'])
        files_created.append(benchmarks_file)
        self.logger.info(f"Exported {len(self.csv_data['benchmarks'])} benchmark records to {benchmarks_file}")
        
        # Export device measurements (always create file, even if empty)
        device_file = os.path.join(output_dir, f"device_measurements_{timestamp}.csv")
        with open(device_file, 'w', newline='') as csvfile:
            fieldnames = ['benchmark_id', 'cpu_util', 'mem_megabytes', 'gpu_util', 'gpu_mem_megabytes', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            if self.csv_data['device_measurements']:
                writer.writerows(self.csv_data['device_measurements'])
        files_created.append(device_file)
        self.logger.info(f"Exported {len(self.csv_data['device_measurements'])} device measurement records to {device_file}")
        
        # Export model measurements (flattened, always create file)
        model_file = os.path.join(output_dir, f"model_measurements_{timestamp}.csv")
        with open(model_file, 'w', newline='') as csvfile:
            flattened_data = []
            if self.csv_data['model_measurements']:
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
            else:
                # Create empty file with basic structure
                fieldnames = ['benchmark_id', 'time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        files_created.append(model_file)
        self.logger.info(f"Exported {len(self.csv_data['model_measurements'])} model measurement records to {model_file}")
        
        # Create a summary file
        summary_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
        self._create_summary_csv(summary_file)
        files_created.append(summary_file)
        
        self.logger.info(f"CSV export complete! Created {len(files_created)} files in {output_dir}")
        
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


def parse_arguments() -> tuple[str, str, str, str, bool, str]:
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
    
    parser.add_argument(
        "--csv",
        action="store_true",
        default=False,
        help="Enable CSV output files generation."
    )
    
    parser.add_argument(
        "--csv-output-dir",
        type=str,
        default="benchmark_results",
        help="Directory for CSV output files (default: benchmark_results)."
    )

    args = parser.parse_args()
    
    # CSV is disabled by default, only enabled when --csv is used
    generate_csv = args.csv

    return args.repository, args.branch, args.commit_id, args.commit_msg, generate_csv, args.csv_output_dir


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


def create_global_metrics_recorder(repository: str, branch: str, commit_id: str, commit_msg: str, 
                                   generate_csv: bool = False) -> MetricsRecorder:
    """
    Create a global metrics recorder that will be used across all benchmarks.
    """
    connection = create_database_connection()
    recorder = MetricsRecorder(connection, logger, repository, branch, commit_id, commit_msg, generate_csv)
    
    # Log the storage mode
    storage_modes = []
    if connection is not None:
        storage_modes.append("database")
    if generate_csv:
        storage_modes.append("CSV")
    
    if not storage_modes:
        logger.warning("Running benchmarks with NO data storage (no database connection, CSV disabled)")
        logger.warning("Use --csv flag to enable CSV output when database is unavailable")
    else:
        logger.info(f"Running benchmarks with: {' + '.join(storage_modes)} storage")
    
    return recorder


if __name__ == "__main__":
    benchmarks_folder_path = os.path.dirname(os.path.realpath(__file__))
    benches_folder_path = os.path.join(benchmarks_folder_path, "benches")

    repository, branch, commit_id, commit_msg, generate_csv, csv_output_dir = parse_arguments()
    
    # Create a global metrics recorder
    global_metrics_recorder = create_global_metrics_recorder(repository, branch, commit_id, commit_msg, generate_csv)
    
    successful_benchmarks = 0
    failed_benchmarks = 0
    
    # Automatically discover all benchmark modules in benches/ folder
    benchmark_modules = []
    
    if os.path.exists(benches_folder_path):
        logger.debug(f"Scanning for benchmarks in: {benches_folder_path}")
        for entry in os.scandir(benches_folder_path):
            if not entry.name.endswith(".py"):
                continue
            if entry.name.startswith("__"):  # Skip __init__.py, __pycache__, etc.
                continue
                
            # Check if the file has a run_benchmark function
            try:
                logger.debug(f"checking if benches/{entry.name} has run_benchmark function")
                module = import_from_path(entry.name.split(".")[0], entry.path)
                if hasattr(module, 'run_benchmark'):
                    benchmark_modules.append(entry.name)
                    logger.debug(f"discovered benchmark: {entry.name}")
                else:
                    logger.debug(f"skipping {entry.name} - no run_benchmark function found")
            except Exception as e:
                logger.debug(f"failed to check benches/{entry.name}: {e}")
    else:
        logger.warning(f"Benches directory not found: {benches_folder_path}")

    if benchmark_modules:
        logger.info(f"Discovered {len(benchmark_modules)} benchmark(s): {benchmark_modules}")
    else:
        logger.warning("No benchmark modules found in benches/ directory")

    for module_name in benchmark_modules:
        module_path = os.path.join(benches_folder_path, module_name)
        try:
            logger.debug(f"loading: {module_name}")
            module = import_from_path(module_name.split(".")[0], module_path)
            logger.info(f"running benchmarks in: {module_name}")
            
            # Check if the module has an updated run_benchmark function that accepts metrics_recorder
            try:
                # Try the new signature first
                module.run_benchmark(logger, repository, branch, commit_id, commit_msg, global_metrics_recorder)
            except TypeError:
                # Fall back to the old signature for backward compatibility
                logger.warning(f"Module {module_name} using old run_benchmark signature - database connection will be created per module")
                module.run_benchmark(logger, repository, branch, commit_id, commit_msg)
            
            successful_benchmarks += 1
        except ImportModuleException as e:
            logger.error(e)
            failed_benchmarks += 1
        except Exception as e:
            logger.error(f"error running benchmarks for {module_name}: {e}")
            failed_benchmarks += 1

    # Add some sample data if no data was actually collected (for testing purposes)
    if generate_csv and global_metrics_recorder.csv_data is not None:
        total_data_points = (len(global_metrics_recorder.csv_data['benchmarks']) + 
                            len(global_metrics_recorder.csv_data['device_measurements']) + 
                            len(global_metrics_recorder.csv_data['model_measurements']))
    else:
        total_data_points = 1  # Assume data exists if CSV is disabled
    
    if generate_csv and total_data_points == 0:
        logger.warning("CSV is enabled but no data was collected from any benchmarks.")
        logger.warning("Adding sample data for CSV export demonstration...")
        sample_benchmark_id = global_metrics_recorder.initialise_benchmark({
            "note": "Sample data - no actual data was collected",
            "reason": "All benchmark modules failed or collected no data",
            "model_name": "sample-model",
            "gpu_name": "Sample GPU"
        })
        
        # Add a few sample measurements
        logger.info("Adding sample device measurements...")
        for i in range(5):
            global_metrics_recorder.collect_device_measurements(
                sample_benchmark_id, 
                25.0 + i*5, 
                1024.0 + i*100, 
                50.0 + i*10, 
                2048.0 + i*200
            )
        
        logger.info("Adding sample model measurements...")
        global_metrics_recorder.collect_model_measurements(sample_benchmark_id, {
            "sample_load_time": 2.5,
            "sample_inference_time": 0.1,
            "sample_throughput": 100.0,
            "note": "This is sample data since no real benchmarks collected data"
        })
        
        logger.info("Sample data added successfully.")

    # Export CSV results at the end (if enabled)
    try:
        if generate_csv:
            global_metrics_recorder.export_to_csv(csv_output_dir)
            logger.info(f"CSV reports have been generated and saved to the {csv_output_dir} directory")
        else:
            logger.info("CSV generation disabled - no CSV files created (use --csv to enable)")
        
        logger.info(f"Benchmark run completed. Successful: {successful_benchmarks}, Failed: {failed_benchmarks}")
    except Exception as e:
        logger.error(f"Failed to export CSV results: {e}")
    finally:
        global_metrics_recorder.close()
