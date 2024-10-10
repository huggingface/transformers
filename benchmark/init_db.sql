CREATE TABLE IF NOT EXISTS benchmarks (
  benchmark_id SERIAL PRIMARY KEY,
  branch VARCHAR(255),
  commit_id VARCHAR(72),
  commit_message VARCHAR(70),
  gpu_name VARCHAR(255),
  created_at timestamp without time zone NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'UTC')
);

CREATE TABLE IF NOT EXISTS device_measurements (
  measurement_id SERIAL PRIMARY KEY,
  benchmark_id int REFERENCES benchmarks (benchmark_id),
  cpu_util double precision,
  mem_megabytes double precision,
  gpu_util double precision,
  gpu_mem_megabytes double precision,
  time timestamp without time zone NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'UTC')
);

CREATE TABLE IF NOT EXISTS model_measurements (
  measurement_id SERIAL PRIMARY KEY,
  benchmark_id int REFERENCES benchmarks (benchmark_id),
  measurements jsonb,
  time timestamp without time zone NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'UTC')
);

