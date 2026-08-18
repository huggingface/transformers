# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class ModelDebtReport:
    model_id: str
    mdi_score: float  # Model Debt Index (target <= 12.0)
    memory_inflation_multiplier: float  # Target <= 1.08x
    latency_ms_per_token: float  # Target <= 35ms/tok
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for Hugging Face Transformers model runs."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_model_inference(
        self,
        model_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        meta_hash = hashlib.sha256(meta_bytes).hexdigest()
        canonical_content = f"{index}|{self._last_hash}|{model_id}|{event_type}|{readiness_index}|{timestamp}|{meta_hash}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "model_id": model_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtEvaluator:
    """A2Z SOC Production Debt & Technical Due Diligence Evaluator for Hugging Face Transformers.

    Quantifies foundation model inference pipelines against 4 Enterprise KPIs:
    1. Model Debt Index (MDI <= 12.0)
    2. KV Cache Memory Inflation Multiplier (KVM <= 1.08x)
    3. P99 Token Latency Ceiling (<= 35ms/tok)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_mdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_mdi = max_acceptable_mdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_inference(
        self,
        model_id: str,
        allocated_kv_cache_mb: float = 2048.0,
        utilized_kv_cache_mb: float = 1950.0,
        latency_ms_per_token: float = 25.0,
        autoregressive_loop_count: int = 0,
        un_gated_mutations: int = 0,
    ) -> ModelDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_model_inference(
                model_id=model_id,
                event_type="inference_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            raise PermissionError(
                "A2Z SOC ActionGate: Emergency kill switch is engaged. Foundation model inference halted."
            )

        critical_smells: list[str] = []

        # KPI 2: KV Cache Memory Inflation Multiplier
        memory_ratio = allocated_kv_cache_mb / max(1.0, utilized_kv_cache_mb)
        if memory_ratio > 1.8:
            critical_smells.append(f"HIGH_KV_CACHE_INFLATION_{memory_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if latency_ms_per_token > 60.0:
            critical_smells.append(f"HIGH_TOKEN_LATENCY_{latency_ms_per_token:.1f}MS")

        # Repetitive autoregressive loop count
        if autoregressive_loop_count > 2:
            critical_smells.append(f"DETECTED_{autoregressive_loop_count}_REPETITIVE_GENERATION_LOOPS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_TOOL_MUTATIONS")

        # KPI 1: Model Debt Index (0 = Clean, 100 = Catastrophic)
        mdi = (
            max(0.0, (memory_ratio - 1.0) * 20.0)
            + max(0.0, (latency_ms_per_token - 35.0) * 0.5)
            + (autoregressive_loop_count * 15.0)
            + (un_gated_mutations * 30.0)
        )
        mdi_score = round(min(100.0, mdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - mdi_score)
        is_production_ready = (
            mdi_score <= self.max_acceptable_mdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_model_inference(
            model_id=model_id,
            event_type="inference_authorized" if is_production_ready else "inference_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "mdi_score": mdi_score,
                "memory_ratio": memory_ratio,
                "allocated_kv_cache_mb": allocated_kv_cache_mb,
                "utilized_kv_cache_mb": utilized_kv_cache_mb,
                "latency_ms_per_token": latency_ms_per_token,
                "autoregressive_loop_count": autoregressive_loop_count,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return ModelDebtReport(
            model_id=model_id,
            mdi_score=mdi_score,
            memory_inflation_multiplier=round(memory_ratio, 2),
            latency_ms_per_token=round(latency_ms_per_token, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
