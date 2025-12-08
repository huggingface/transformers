#!/usr/bin/env python3
"""
Monitor all Python processes and kill them if total memory usage exceeds threshold.
Excludes buff/cache from memory calculations (uses available memory).

Usage:
    python memory_monitor.py [--threshold 90] [--interval 5] [--dry-run]
"""

import psutil
import time
import sys
import signal
import argparse
from datetime import datetime

# Force unbuffered output so logs appear immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def print_both(message, flush=True):
    """Print to both stdout (log file) and all terminals."""
    print(message, flush=flush)
    
    # Try to write to the original terminal (if it exists)
    try:
        with open('/dev/tty', 'w') as tty:
            tty.write(message + '\n')
    except:
        pass
    
    # Also write to a shared file that users can monitor
    try:
        with open('/tmp/memory_monitor_alerts.txt', 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    except:
        pass
    
    # Try to broadcast to all logged-in terminals using wall
    # This works even if the monitor started before SSH
    try:
        import subprocess
        subprocess.run(['wall', message], stderr=subprocess.DEVNULL, timeout=1)
    except:
        pass


def get_memory_info():
    """Get system memory info, excluding buffers/cache."""
    mem = psutil.virtual_memory()
    
    used_excl_cache = mem.used - mem.buffers - mem.cached
    
    # The correct three components that add to 100%:
    # Total = Used (apps) + Available + Non-reclaimable cache
    #
    # Because:
    # Available = Free + Reclaimable cache
    # Total = Used (apps) + Free + Total cache
    # Total = Used (apps) + Free + Reclaimable + Non-reclaimable
    # Total = Used (apps) + Available + Non-reclaimable ✓
    #
    # Therefore:
    # Non-reclaimable = Total - Used (apps) - Available
    
    non_reclaimable_cache = mem.total - used_excl_cache - mem.available
    non_reclaimable_cache = max(0, non_reclaimable_cache)
    
    return {
        'total': mem.total,
        'available': mem.available,
        'used': used_excl_cache,
        'non_reclaimable_cache': non_reclaimable_cache,
        'percent_used': (used_excl_cache / mem.total) * 100,
        'percent_available': (mem.available / mem.total) * 100,
        'percent_non_reclaimable_cache': (non_reclaimable_cache / mem.total) * 100
    }


def get_all_python_processes():
    """Get all running Python processes with their memory usage."""
    python_procs = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'create_time']):
        try:
            # Check if it's a Python process
            name = proc.info['name']
            if name and ('python' in name.lower()):
                mem_mb = proc.info['memory_info'].rss / (1024 * 1024)  # Convert to MB
                
                # Simplified cmdline to avoid hanging on slow processes
                try:
                    cmdline = ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else ''
                except:
                    cmdline = '<unknown>'
                
                python_procs.append({
                    'pid': proc.info['pid'],
                    'name': name,
                    'cmdline': cmdline,
                    'memory_mb': mem_mb,
                    'process': proc
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, Exception):
            pass
    
    return python_procs


def format_size(bytes_size):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def kill_python_processes(python_procs, dry_run=False, use_sigkill=False):
    """Kill all Python processes except this monitor script."""
    current_pid = psutil.Process().pid
    killed_pids = []
    
    for proc_info in python_procs:
        if proc_info['pid'] == current_pid:
            continue  # Don't kill the monitor itself
        
        try:
            if dry_run:
                msg = f"[DRY RUN] Would kill PID {proc_info['pid']}: {proc_info['name']} ({proc_info['memory_mb']:.2f} MB)"
                print_both(msg)
            else:
                # If system is critical (use_sigkill), kill immediately
                if use_sigkill:
                    proc_info['process'].kill()
                    msg = f"Sent SIGKILL to PID {proc_info['pid']}: {proc_info['name']} ({proc_info['memory_mb']:.2f} MB)"
                    print_both(msg)
                else:
                    proc_info['process'].send_signal(signal.SIGTERM)
                    msg = f"Sent SIGTERM to PID {proc_info['pid']}: {proc_info['name']} ({proc_info['memory_mb']:.2f} MB)"
                    print_both(msg)
                killed_pids.append(proc_info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            msg = f"Failed to kill PID {proc_info['pid']}: {e}"
            print_both(msg)
    
    # Wait a bit and force kill if still alive (only if we used SIGTERM)
    if not dry_run and not use_sigkill and killed_pids:
        time.sleep(1)
        for pid in killed_pids:
            try:
                proc = psutil.Process(pid)
                if proc.is_running():
                    proc.kill()
                    msg = f"Sent SIGKILL to PID {pid} (still running after SIGTERM)"
                    print_both(msg)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass  # Process already dead or not accessible
    
    return killed_pids


def monitor_memory(threshold=90, interval=5, dry_run=False, verbose=False):
    """
    Monitor memory usage and kill Python processes if available memory drops below threshold.
    
    Args:
        threshold: Kill when used memory exceeds this % (i.e., available < 100-threshold)
                   For example: threshold=70 means kill when available < 30%
        interval: Check interval in seconds
        dry_run: If True, don't actually kill processes
        verbose: If True, print status every check
    """
    min_available_percent = 100 - threshold
    print(f"Starting memory monitor (kill when available < {min_available_percent:.1f}%, interval: {interval}s, dry_run: {dry_run})", flush=True)
    print(f"Monitoring all Python processes (excluding buff/cache from memory calculation)", flush=True)
    print("Press Ctrl+C to stop\n", flush=True)
    
    check_count = 0
    
    try:
        while True:
            check_count += 1
            mem_info = get_memory_info()
            python_procs = get_all_python_processes()
            
            total_python_mem_mb = sum(p['memory_mb'] for p in python_procs)
            total_python_mem_percent = (total_python_mem_mb * 1024 * 1024 / mem_info['total']) * 100
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if verbose or mem_info['percent_available'] < (100 - threshold) * 1.2:  # Show when approaching threshold
                print(f"[{timestamp}] Check #{check_count}")
                print(f"  System Memory: {mem_info['percent_used']:.1f}% used (apps), {mem_info['percent_available']:.1f}% available, {mem_info['percent_non_reclaimable_cache']:.1f}% non-reclaimable cache")
                print(f"  Total: {format_size(mem_info['total'])}, Used: {format_size(mem_info['used'])}, Available: {format_size(mem_info['available'])}, Non-reclaimable Cache: {format_size(mem_info['non_reclaimable_cache'])}")
                print(f"  Python processes: {len(python_procs)} running, using {total_python_mem_mb:.2f} MB ({total_python_mem_percent:.1f}% of total)")
            
            # Check if available memory has dropped below threshold
            # For threshold=70, this means kill when available < 30%
            min_available_percent = 100 - threshold
            if mem_info['percent_available'] < min_available_percent:
                # Use SIGKILL immediately if critically low available memory (<5%)
                use_sigkill = mem_info['percent_available'] < 5
                
                print_both(f"\n{'='*80}")
                print_both(f"⚠️  MEMORY THRESHOLD EXCEEDED: {mem_info['percent_available']:.1f}% available < {min_available_percent:.1f}% minimum")
                print_both(f"⚠️  (System memory used: {mem_info['percent_used']:.1f}%)")
                if use_sigkill:
                    print_both(f"⚠️  CRITICAL: Using SIGKILL immediately (available < 5%)")
                print_both(f"{'='*80}")
                print_both(f"Available memory: {format_size(mem_info['available'])} ({mem_info['percent_available']:.1f}%)")
                print_both(f"Non-reclaimable Buffers/Cache: {format_size(mem_info['non_reclaimable_cache'])} ({mem_info['percent_non_reclaimable_cache']:.1f}%)")
                print_both(f"\nPython processes ({len(python_procs)} total, {total_python_mem_mb:.2f} MB):")
                
                # Sort by memory usage
                python_procs_sorted = sorted(python_procs, key=lambda x: x['memory_mb'], reverse=True)
                
                # Show fewer processes if system is hanging
                num_to_show = 5 if use_sigkill else 10
                for i, proc in enumerate(python_procs_sorted[:num_to_show], 1):
                    cmdline_short = proc['cmdline'][:80] + '...' if len(proc['cmdline']) > 80 else proc['cmdline']
                    print_both(f"  {i}. PID {proc['pid']}: {proc['memory_mb']:.2f} MB - {cmdline_short}")
                
                if len(python_procs_sorted) > num_to_show:
                    print_both(f"  ... and {len(python_procs_sorted) - num_to_show} more processes")
                
                print_both(f"\n{'='*80}")
                print_both("Killing all Python processes...")
                print_both(f"{'='*80}\n")
                
                killed_pids = kill_python_processes(python_procs, dry_run=dry_run, use_sigkill=use_sigkill)
                
                if not dry_run:
                    print_both(f"\n✓ Killed {len(killed_pids)} Python processes")
                    print_both("Memory monitor will continue running...")
                else:
                    print_both(f"\n[DRY RUN] Would have killed {len([p for p in python_procs if p['pid'] != psutil.Process().pid])} processes")
                
                print_both("")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMemory monitor stopped by user")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Python processes and kill them if memory usage exceeds threshold"
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=90,
        help='Memory usage threshold percentage. Kills processes when available memory drops below (100-threshold)%%. For example: --threshold 70 means kill when available < 30%% (default: 90, meaning kill when available < 10%%)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=5,
        help='Check interval in seconds, supports decimals (default: 5)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be killed without actually killing'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print status on every check (default: only when approaching threshold)'
    )
    
    args = parser.parse_args()
    
    if args.threshold <= 0 or args.threshold > 100:
        print("Error: threshold must be between 0 and 100")
        sys.exit(1)
    
    if args.interval <= 0:
        print("Error: interval must be positive")
        sys.exit(1)
    
    monitor_memory(
        threshold=args.threshold,
        interval=args.interval,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
