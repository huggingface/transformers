"""CPU memory monitoring utilities for training CI tests."""

import logging
import os
from collections import namedtuple

# Try to import psutil for CPU memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


# Named tuple for passing memory stats for logging
MemoryStats = namedtuple(
    "MemoryStats",
    [
        "rss_gib",           # Resident Set Size in GiB
        "rss_pct",           # RSS as percentage of total memory
        "vms_gib",           # Virtual Memory Size in GiB
        "peak_rss_gib",      # Peak RSS in GiB
        "peak_rss_pct",      # Peak RSS as percentage of total memory
        "available_gib",     # Available system memory in GiB
        "total_gib",         # Total system memory in GiB
    ],
)


class CPUMemoryMonitor:
    """Monitor CPU memory usage for the current process."""

    def __init__(self):
        self.device_name = "CPU"
        self._peak_rss = 0

        if HAS_PSUTIL:
            self._process = psutil.Process(os.getpid())
            mem_info = psutil.virtual_memory()
            self.total_memory = mem_info.total
            self.total_memory_gib = self._to_gib(self.total_memory)
        else:
            self._process = None
            self.total_memory = 0
            self.total_memory_gib = 0

    def _to_gib(self, memory_in_bytes: int) -> float:
        """Convert bytes to GiB."""
        return memory_in_bytes / (1024 * 1024 * 1024)

    def _to_pct(self, memory_in_bytes: int) -> float:
        """Convert bytes to percentage of total memory."""
        if self.total_memory == 0:
            return 0.0
        return 100.0 * memory_in_bytes / self.total_memory

    def _update_peak(self) -> None:
        """Update peak memory tracking."""
        if self._process is not None:
            current_rss = self._process.memory_info().rss
            self._peak_rss = max(self._peak_rss, current_rss)

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if not HAS_PSUTIL:
            logger.warning("psutil not installed, CPU memory tracking disabled")
            return MemoryStats(0, 0, 0, 0, 0, 0, 0)

        self._update_peak()

        mem_info = self._process.memory_info()
        sys_mem = psutil.virtual_memory()

        return MemoryStats(
            rss_gib=self._to_gib(mem_info.rss),
            rss_pct=self._to_pct(mem_info.rss),
            vms_gib=self._to_gib(mem_info.vms),
            peak_rss_gib=self._to_gib(self._peak_rss),
            peak_rss_pct=self._to_pct(self._peak_rss),
            available_gib=self._to_gib(sys_mem.available),
            total_gib=self._to_gib(sys_mem.total),
        )

    def reset_peak_stats(self) -> None:
        """Reset peak memory tracking."""
        if self._process is not None:
            self._peak_rss = self._process.memory_info().rss


def build_cpu_memory_monitor() -> CPUMemoryMonitor:
    """Build and initialize a CPU memory monitor."""
    monitor = CPUMemoryMonitor()
    if HAS_PSUTIL:
        logger.info(
            f"CPU memory monitor initialized: {monitor.total_memory_gib:.2f} GiB total"
        )
    else:
        logger.warning("psutil not available, memory monitoring disabled")
    return monitor

