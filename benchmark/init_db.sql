\c metrics;

CREATE TABLE benchmarks (
  benchmark_id SERIAL PRIMARY KEY,
  branch VARCHAR(255),
  commit_id VARCHAR(72),
  commit_message VARCHAR(70),
  gpu_name VARCHAR(255),
  created_at timestamp without time zone NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'UTC')
);

CREATE TABLE device_measurements (
  measurement_id SERIAL PRIMARY KEY,
  benchmark_id int REFERENCES benchmarks (benchmark_id),
  cpu_util double precision,
  mem_megabytes double precision,
  gpu_util double precision,
  gpu_mem_megabytes double precision,
  time timestamp without time zone NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'UTC')
);

CREATE TABLE model_measurements (
  measurement_id SERIAL PRIMARY KEY,
  benchmark_id int REFERENCES benchmarks (benchmark_id),
  model_load_time double precision,
  first_eager_forward_pass_time_secs double precision,
  second_eager_forward_pass_time_secs double precision,
  first_compile_forward_pass_time_secs double precision,
  second_compile_forward_pass_time_secs double precision,
  third_compile_forward_pass_time_secs double precision,
  fourth_compile_forward_pass_time_secs double precision,
  time timestamp without time zone NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'UTC')
);

