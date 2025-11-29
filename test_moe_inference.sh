#!/bin/bash

# Script to test vLLM inference with transformers backend for MoE models
# Each model is tested and results are reported

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Define models to test (architecture -> HuggingFace checkpoint)
declare -A MODELS=(
    ["olmoe"]="allenai/OLMoE-1B-7B-0924"
    ["mixtral"]="TitanML/tiny-mixtral"
    ["qwen2_moe"]="Qwen/Qwen1.5-MoE-A2.7B-Chat"
    ["qwen3_moe"]="tiny-random/qwen3-moe"
    ["gpt_oss"]="tiny-random/gpt-oss"
)

# Results tracking
declare -A RESULTS

echo "=========================================="
echo "  MoE Models Inference Test Script"
echo "=========================================="
echo ""

# Function to run inference test
run_test() {
    local model_name=$1
    local model_checkpoint=$2
    
    echo -e "${YELLOW}Testing: ${model_name} (${model_checkpoint})${NC}"
    echo "-------------------------------------------"
    
    # Set environment and run the command
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python vllm/examples/offline_inference/basic/generate.py \
        --model "$model_checkpoint" \
        --model-impl transformers \
        --enforce-eager \
        --no-enable-prefix-caching \
        2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        RESULTS[$model_name]="SUCCESS"
        echo -e "${GREEN}✓ ${model_name}: SUCCESS${NC}"
    else
        RESULTS[$model_name]="FAILED (exit code: $exit_code)"
        echo -e "${RED}✗ ${model_name}: FAILED (exit code: $exit_code)${NC}"
    fi
    
    echo ""
    return $exit_code
}

# Run tests for each model
for model_name in "${!MODELS[@]}"; do
    run_test "$model_name" "${MODELS[$model_name]}"
done

# Print summary
echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo ""

success_count=0
fail_count=0

for model_name in "${!RESULTS[@]}"; do
    result="${RESULTS[$model_name]}"
    if [[ "$result" == "SUCCESS" ]]; then
        echo -e "${GREEN}✓ ${model_name}: ${result}${NC}"
        ((success_count++))
    else
        echo -e "${RED}✗ ${model_name}: ${result}${NC}"
        ((fail_count++))
    fi
done

echo ""
echo "-------------------------------------------"
echo -e "Total: ${GREEN}${success_count} passed${NC}, ${RED}${fail_count} failed${NC}"
echo "=========================================="
