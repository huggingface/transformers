#!/bin/bash

# Script to run all FSDP mixin tests for dense models.
# Usage: ./run_fsdp_mixin_dense_tests.sh [/path/to/results]
#        ./run_fsdp_mixin_dense_tests.sh --model <model_name> [/path/to/results]
#        ./run_fsdp_mixin_dense_tests.sh --report /path/to/results

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREY='\033[0;90m'
DIM='\033[0;90m'
NC='\033[0m'

# All FSDP mixin test methods
TEST_METHODS=(
    "test_get_transformer_block_classes"
    "test_fsdp2_sharding_structure_untied"
    "test_fsdp2_sharding_structure_tied"
    "test_fsdp2_auto_plan_vs_ddp_float32_untied"
    "test_fsdp2_auto_plan_vs_ddp_bfloat16_untied"
    "test_fsdp2_auto_plan_vs_ddp_float32_tied"
    "test_fsdp2_auto_plan_vs_ddp_bfloat16_tied"
    "test_fsdp2_manual_plan_vs_ddp_float32_untied"
    "test_fsdp2_manual_plan_vs_ddp_bfloat16_untied"
    "test_fsdp2_manual_plan_vs_ddp_float32_tied"
    "test_fsdp2_manual_plan_vs_ddp_bfloat16_tied"
    "test_fsdp2_save_load"
)

# Dense models that inherit from CausalLMModelTest
TEST_ENTRIES=(
    "tests/models/apertus/test_modeling_apertus.py::ApertusModelTest"
    "tests/models/arcee/test_modeling_arcee.py::ArceeModelTest"
    "tests/models/bloom/test_modeling_bloom.py::BloomModelTest"
    "tests/models/blt/test_modeling_blt.py::BltModelTest"
    "tests/models/cwm/test_modeling_cwm.py::CwmModelTest"
    "tests/models/ernie4_5/test_modeling_ernie4_5.py::Ernie4_5ModelTest"
    "tests/models/exaone4/test_modeling_exaone4.py::Exaone4ModelTest"
    "tests/models/falcon/test_modeling_falcon.py::FalconModelTest"
    "tests/models/gemma/test_modeling_gemma.py::GemmaModelTest"
    "tests/models/gemma2/test_modeling_gemma2.py::Gemma2ModelTest"
    "tests/models/gemma3/test_modeling_gemma3.py::Gemma3TextModelTest"
    "tests/models/gemma3n/test_modeling_gemma3n.py::Gemma3nTextModelTest"
    "tests/models/glm/test_modeling_glm.py::GlmModelTest"
    "tests/models/glm4/test_modeling_glm4.py::Glm4ModelTest"
    "tests/models/gpt2/test_modeling_gpt2.py::GPT2ModelTest"
    "tests/models/helium/test_modeling_helium.py::HeliumModelTest"
    "tests/models/hunyuan_v1_dense/test_modeling_hunyuan_v1_dense.py::HunYuanDenseV1ModelTest"
    "tests/models/jais2/test_modeling_jais2.py::Jais2ModelTest"
    "tests/models/lfm2/test_modeling_lfm2.py::Lfm2ModelTest"
    "tests/models/llama/test_modeling_llama.py::LlamaModelTest"
    "tests/models/ministral/test_modeling_ministral.py::MinistralModelTest"
    "tests/models/ministral3/test_modeling_ministral3.py::Ministral3ModelTest"
    "tests/models/mistral/test_modeling_mistral.py::MistralModelTest"
    "tests/models/modernbert_decoder/test_modeling_modernbert_decoder.py::ModernBertDecoderModelTest"
    "tests/models/nanochat/test_modeling_nanochat.py::NanoChatModelTest"
    "tests/models/nemotron/test_modeling_nemotron.py::NemotronModelTest"
    "tests/models/olmo3/test_modeling_olmo3.py::Olmo3ModelTest"
    "tests/models/persimmon/test_modeling_persimmon.py::PersimmonModelTest"
    "tests/models/phi/test_modeling_phi.py::PhiModelTest"
    "tests/models/phi3/test_modeling_phi3.py::Phi3ModelTest"
    "tests/models/qwen2/test_modeling_qwen2.py::Qwen2ModelTest"
    "tests/models/qwen3/test_modeling_qwen3.py::Qwen3ModelTest"
    "tests/models/qwen3_5/test_modeling_qwen3_5.py::Qwen3_5TextModelTest"
    "tests/models/recurrent_gemma/test_modeling_recurrent_gemma.py::RecurrentGemmaModelTest"
    "tests/models/seed_oss/test_modeling_seed_oss.py::SeedOssModelTest"
    "tests/models/smollm3/test_modeling_smollm3.py::SmolLM3ModelTest"
    "tests/models/stablelm/test_modeling_stablelm.py::StableLmModelTest"
    "tests/models/starcoder2/test_modeling_starcoder2.py::Starcoder2ModelTest"
    "tests/models/vaultgemma/test_modeling_vaultgemma.py::VaultGemmaModelTest"
    "tests/models/youtu/test_modeling_youtu.py::YoutuModelTest"
)

# ── Helpers ─────────────────────────────────────────────────────────────────
extract_model_name() {
    echo "$1" | sed 's|tests/models/\([^/]*\)/.*|\1|'
}

# ── Report ──────────────────────────────────────────────────────────────────
print_report() {
    local results_dir=$1
    results_dir=$(cd "$results_dir" && pwd)

    if [ ! -d "$results_dir" ]; then
        echo "Error: Results directory '$results_dir' does not exist"
        exit 1
    fi

    echo "=========================================="
    echo "  FSDP Mixin Dense Test Report"
    echo "  Results directory: $results_dir"
    echo "=========================================="
    echo ""

    local success_count=0
    local fail_count=0
    local skip_count=0

    for entry in "${TEST_ENTRIES[@]}"; do
        local model_name
        model_name=$(extract_model_name "$entry")
        local result_file="$results_dir/${model_name}.result"
        if [ -f "$result_file" ]; then
            local result=$(cat "$result_file")
            if [[ "$result" == "SUCCESS" ]]; then
                echo -e "${GREEN}✓ ${model_name}: ${result}${NC}"
                ((success_count++))
            elif [[ "$result" == "SKIPPED" ]]; then
                echo -e "${GREY}○ ${model_name}: ${result}${NC}"
                ((skip_count++))
            else
                echo -e "${RED}✗ ${model_name}: ${result}${NC}"
                if [ -f "$results_dir/${model_name}.log" ]; then
                    echo -e "${DIM}  Error snippet:"
                    tail -n 5 "$results_dir/${model_name}.log" | while read -r line; do echo -e "    ${DIM}${line}${NC}"; done
                fi
                ((fail_count++))
            fi
        else
            echo -e "${YELLOW}? ${model_name}: NOT RUN${NC}"
        fi
    done

    echo ""
    echo "-------------------------------------------"
    echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}"
    echo "=========================================="

    if [ $fail_count -gt 0 ]; then
        exit 1
    fi
}

# ── Argument parsing ────────────────────────────────────────────────────────
if [ "$1" == "--report" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --report /path/to/results"
        exit 1
    fi
    print_report "$2"
    exit 0
fi

if [ "$1" == "--model" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --model <model_name> [/path/to/results]"
        echo "Available models:"
        for entry in "${TEST_ENTRIES[@]}"; do
            printf '  %s\n' "$(extract_model_name "$entry")"
        done
        exit 1
    fi
    SINGLE_MODEL="$2"
    shift 2
    MATCHED_ENTRY=""
    for entry in "${TEST_ENTRIES[@]}"; do
        if [ "$(extract_model_name "$entry")" == "$SINGLE_MODEL" ]; then
            MATCHED_ENTRY="$entry"
            break
        fi
    done
    if [ -z "$MATCHED_ENTRY" ]; then
        echo "Error: Unknown model '$SINGLE_MODEL'"
        echo "Available models:"
        for entry in "${TEST_ENTRIES[@]}"; do
            printf '  %s\n' "$(extract_model_name "$entry")"
        done
        exit 1
    fi
    TEST_ENTRIES=("$MATCHED_ENTRY")
fi

# ── Results directory ───────────────────────────────────────────────────────
if [ -n "$1" ]; then
    RESULTS_DIR="$1"
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
else
    RESULTS_DIR=$(mktemp -d)
    CLEANUP_RESULTS=true
fi
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)

if [ "$CLEANUP_RESULTS" = true ]; then
    trap "rm -rf $RESULTS_DIR" EXIT
fi

echo "=========================================="
echo "  FSDP Mixin Dense Tests"
echo "  (${#TEST_ENTRIES[@]} models, ${#TEST_METHODS[@]} tests each)"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo ""

# ── Run tests ───────────────────────────────────────────────────────────────
success_count=0
fail_count=0
skip_count=0

for entry in "${TEST_ENTRIES[@]}"; do
    model_name=$(extract_model_name "$entry")

    # Build pytest node IDs for all test methods
    test_ids=()
    for test_method in "${TEST_METHODS[@]}"; do
        test_ids+=("${entry}::${test_method}")
    done

    # Single pytest invocation per model -> one log file
    python -m pytest -vs "${test_ids[@]}" \
        > "$RESULTS_DIR/${model_name}.log" 2>&1
    exit_code=$?

    if [ $exit_code -eq 5 ]; then
        echo "SKIPPED" > "$RESULTS_DIR/${model_name}.result"
        echo -e "${GREY}○ ${model_name}: SKIPPED${NC}"
        ((skip_count++))
    elif [ $exit_code -eq 0 ]; then
        if grep -q "passed" "$RESULTS_DIR/${model_name}.log"; then
            echo "SUCCESS" > "$RESULTS_DIR/${model_name}.result"
            echo -e "${GREEN}✓ ${model_name}${NC}"
            ((success_count++))
        else
            echo "SKIPPED" > "$RESULTS_DIR/${model_name}.result"
            echo -e "${GREY}○ ${model_name}: SKIPPED${NC}"
            ((skip_count++))
        fi
    else
        echo "FAILED (exit code: $exit_code)" > "$RESULTS_DIR/${model_name}.result"
        echo -e "${RED}✗ ${model_name}: FAILED${NC}"
        echo -e "${DIM}  $(tail -n 3 "$RESULTS_DIR/${model_name}.log")${NC}"
        ((fail_count++))
    fi
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo ""
echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}"
echo "=========================================="

if [ $fail_count -gt 0 ]; then
    echo ""
    echo "Failed test logs:"
    for entry in "${TEST_ENTRIES[@]}"; do
        model_name=$(extract_model_name "$entry")
        result_file="$RESULTS_DIR/${model_name}.result"
        if [ -f "$result_file" ] && [[ "$(cat "$result_file")" != "SUCCESS" ]] && [[ "$(cat "$result_file")" != "SKIPPED" ]]; then
            echo "  $RESULTS_DIR/${model_name}.log"
        fi
    done
    exit 1
fi
