#!/bin/bash

# cpu small
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 1 --avg-seqlen 32 --max-seqlen 32 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask False --benchmarks hf bt 
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 1 --avg-seqlen 32 --max-seqlen 32 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 1 --avg-seqlen 30 --max-seqlen 32 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 1 --avg-seqlen 24 --max-seqlen 32 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 1 --avg-seqlen 16 --max-seqlen 32 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 1 --avg-seqlen 8 --max-seqlen 32 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt

# cpu medium
# python3 benchmark_bettertransformer.py --num-batches 10 --batch-size 8 --avg-seqlen 64 --max-seqlen 64 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask False --benchmarks hf bt 
# python3 benchmark_bettertransformer.py --num-batches 10 --batch-size 8 --avg-seqlen 64 --max-seqlen 64 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 10 --batch-size 8 --avg-seqlen 61 --max-seqlen 64 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 10 --batch-size 8 --avg-seqlen 48 --max-seqlen 64 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 10 --batch-size 8 --avg-seqlen 32 --max-seqlen 64 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 10 --batch-size 8 --avg-seqlen 16 --max-seqlen 64 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt

# cpu large
# python3 benchmark_bettertransformer.py --num-batches 5 --batch-size 64 --avg-seqlen 256 --max-seqlen 256 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask False --benchmarks hf bt 
# python3 benchmark_bettertransformer.py --num-batches 5 --batch-size 64 --avg-seqlen 256 --max-seqlen 256 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 5 --batch-size 64 --avg-seqlen 235 --max-seqlen 256 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 5 --batch-size 64 --avg-seqlen 192 --max-seqlen 256 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 5 --batch-size 64 --avg-seqlen 128 --max-seqlen 256 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt
# python3 benchmark_bettertransformer.py --num-batches 5 --batch-size 64 --avg-seqlen 64 --max-seqlen 256 --seqlen-stdev 0 --is-cuda False --is-large False --is-half False --use-mask True --benchmarks hf bt

# For profiling
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 245 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 256 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf

# Single layer
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 256 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask False --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 256 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 254 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 250 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 245 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 235 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 225 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 215 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 200 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 182 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 158 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 128 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 100 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 88 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 64 --avg-seqlen 64 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers 1 --is-half True --use-mask True --benchmarks bt hf

# tiny config
# python3 benchmark_bettertransformer.py --num-batches 100 --batch-size 1 --avg-seqlen 25 --max-seqlen 25 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers -1 --is-half True --use-mask False --benchmarks bt hf

# small config
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 1 --avg-seqlen 8 --max-seqlen 32 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers -1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 1 --avg-seqlen 16 --max-seqlen 32 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers -1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 1 --avg-seqlen 25 --max-seqlen 32 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers -1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 1 --avg-seqlen 28 --max-seqlen 32 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers -1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 1 --avg-seqlen 30 --max-seqlen 32 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers -1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 1 --avg-seqlen 31 --max-seqlen 32 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers -1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 1 --avg-seqlen 32 --max-seqlen 32 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers -1 --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 1 --avg-seqlen 32 --max-seqlen 32 --seqlen-stdev 0 --is-cuda True --is-large False --num-layers -1 --is-half True --use-mask False --benchmarks bt hf

# # medium config
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 8 --avg-seqlen 64 --max-seqlen 64 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask False --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 8 --avg-seqlen 64 --max-seqlen 64 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 8 --avg-seqlen 63 --max-seqlen 64 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 8 --avg-seqlen 61 --max-seqlen 64 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 8 --avg-seqlen 57 --max-seqlen 64 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 8 --avg-seqlen 50 --max-seqlen 64 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 8 --avg-seqlen 32 --max-seqlen 64 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 300 --batch-size 8 --avg-seqlen 16 --max-seqlen 64 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf


# large config
#python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 256 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask False --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 256 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 254 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 250 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 245 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 235 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 225 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 215 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 200 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 128 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 64 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf

# large config with varying seqlen
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 235 --max-seqlen 256 --seqlen-stdev 0 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 235 --max-seqlen 256 --seqlen-stdev 1 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 235 --max-seqlen 256 --seqlen-stdev 5 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 235 --max-seqlen 256 --seqlen-stdev 10 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf
# python3 benchmark_bettertransformer.py --num-batches 50 --batch-size 64 --avg-seqlen 235 --max-seqlen 256 --seqlen-stdev 20 --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf

# DEBUG - assumes setting inputs
# python3 benchmark_bettertransformer.py --is-cuda True --is-large False --is-half True --use-mask True --benchmarks bt hf

# summarize
# python3 summarize_results.py log.csv --n-repeat 5









