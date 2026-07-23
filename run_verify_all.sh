#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[0;90m'
NC='\033[0m'

SCRIPT="verify_loading.py"
LOGDIR="$(dirname "$0")/verify_logs"
mkdir -p "$LOGDIR"

NUM_GPUS=$(nvidia-smi -L | wc -l)

# Job definitions: "mode nproc_per_node"
declare -a JOBS=(
    "single_gpu 1"
    "fsdp 2"
    "tp 2"
    "tp_sp 2"
    "tp_fsdp 4"
    "tp_sp_fsdp 4"
)
MODE_NAMES=(single_gpu fsdp tp tp_sp tp_fsdp tp_sp_fsdp)

echo -e "${BOLD}=========================================="
echo -e "  Verify Loading (${NUM_GPUS} GPUs available)"
echo -e "  Modes: ${MODE_NAMES[*]}"
echo -e "  Logs:  $LOGDIR/"
echo -e "==========================================${NC}"
echo ""

# ============================================================
# Round-robin GPU scheduler
# ============================================================
NEXT_GPU=0
MASTER_PORT=29500
PIDS=()
PID_MODES=()

for job in "${JOBS[@]}"; do
    mode=${job% *}
    nproc=${job#* }

    # Wait if not enough GPUs left in this round
    if [ $((NEXT_GPU + nproc)) -gt "$NUM_GPUS" ]; then
        echo -e "${DIM}  (waiting for current round to finish...)${NC}"
        for pid in "${PIDS[@]}"; do
            wait "$pid" 2>/dev/null
        done
        PIDS=()
        NEXT_GPU=0
    fi

    # Build CUDA_VISIBLE_DEVICES range
    GPU_END=$((NEXT_GPU + nproc - 1))
    GPUS=""
    for g in $(seq "$NEXT_GPU" "$GPU_END"); do
        [ -n "$GPUS" ] && GPUS="${GPUS},"
        GPUS="${GPUS}${g}"
    done

    echo -e "  ${CYAN}[${mode}]${NC} GPUs ${NEXT_GPU}-${GPU_END} (nproc=${nproc})"

    if [ "$nproc" -eq 1 ]; then
        CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT" --mode "$mode" \
            > "$LOGDIR/${mode}.log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
            --nproc_per_node="$nproc" --master_port="$MASTER_PORT" \
            "$SCRIPT" --mode "$mode" \
            > "$LOGDIR/${mode}.log" 2>&1 &
        ((MASTER_PORT++))
    fi

    PIDS+=($!)
    PID_MODES+=("$mode")
    NEXT_GPU=$((GPU_END + 1))
done

# Wait for remaining jobs
echo ""
echo -e "${BOLD}Waiting for all jobs to finish...${NC}"
for i in "${!PIDS[@]}"; do
    mode="${PID_MODES[$i]}"
    if wait "${PIDS[$i]}"; then
        echo -e "  ${GREEN}✓${NC} ${mode}"
    else
        echo -e "  ${RED}✗${NC} ${mode} (exit $?)"
    fi
done

# ============================================================
# Results
# ============================================================
echo ""
echo -e "${BOLD}=== Results ===${NC}"
for mode in "${MODE_NAMES[@]}"; do
    log="$LOGDIR/$mode.log"
    loss_before=$(grep -oP 'loss_before = \K[0-9.]+' "$log" 2>/dev/null)
    loss_after=$(grep -oP 'loss_after  = \K[0-9.]+' "$log" 2>/dev/null)
    if grep -q '^PASS' "$log" 2>/dev/null; then
        printf "  ${GREEN}%-12s PASS  (before=%-10s after=%s)${NC}\n" "$mode" "$loss_before" "$loss_after"
    elif [ -n "$loss_before" ]; then
        diff=$(grep -oP 'diff = \K[0-9.e+-]+' "$log" 2>/dev/null)
        printf "  ${RED}%-12s FAIL  (before=%-10s after=%-10s diff=%s)${NC}\n" "$mode" "$loss_before" "$loss_after" "$diff"
    else
        printf "  ${RED}%-12s ERROR (see log)${NC}\n" "$mode"
    fi
done

# ============================================================
# Cross-mode loss comparison
# ============================================================
echo ""
echo -e "${BOLD}=== Cross-mode loss comparison (PASS modes only) ===${NC}"
REF_LOSS=""
ALL_MATCH=1
for mode in "${MODE_NAMES[@]}"; do
    log="$LOGDIR/$mode.log"
    # Only include modes where save/load roundtrip passed
    if ! grep -q '^PASS' "$log" 2>/dev/null; then
        continue
    fi
    loss=$(grep -oP 'loss_before = \K[0-9.]+' "$log" 2>/dev/null)
    if [ -z "$loss" ]; then
        continue
    fi
    if [ -z "$REF_LOSS" ]; then
        REF_LOSS="$loss"
        printf "  ${GREEN}%-12s %s (reference)${NC}\n" "$mode" "$loss"
    elif [ "$loss" = "$REF_LOSS" ]; then
        printf "  ${GREEN}%-12s %s${NC}\n" "$mode" "$loss"
    else
        printf "  ${YELLOW}%-12s %s (differs from %s)${NC}\n" "$mode" "$loss" "$REF_LOSS"
        ALL_MATCH=0
    fi
done
if [ "$ALL_MATCH" -eq 1 ] && [ -n "$REF_LOSS" ]; then
    echo -e "  ${GREEN}All modes produce the same loss.${NC}"
fi

# Hints for failures
HAS_FAIL=0
for mode in "${MODE_NAMES[@]}"; do
    if ! grep -q '^PASS' "$LOGDIR/$mode.log" 2>/dev/null; then
        HAS_FAIL=1
    fi
done
if [ "$HAS_FAIL" -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}Some modes failed. Check logs:${NC}"
    for mode in "${MODE_NAMES[@]}"; do
        if ! grep -q '^PASS' "$LOGDIR/$mode.log" 2>/dev/null; then
            echo -e "  ${YELLOW}cat $LOGDIR/$mode.log${NC}"
        fi
    done
fi
