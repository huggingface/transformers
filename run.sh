#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1 && export PYTHONUNBUFFERED=1 && export OMP_NUM_THREADS=1 && export TRANSFORMERS_IS_CI=true && export PYTEST_TIMEOUT=120 && export RUN_PIPELINE_TESTS=false && export RUN_FLAKY=true
python3 utils/fetch_hub_objects_for_ci.py

echo $(date "+%Y-%m-%d %H:%M:%S")
timeout 600  ./pytest.sh & TIMEOUT_PID=$!; echo $TIMEOUT_PID; echo $TIMEOUT_PID > TIMEOUT_PID.txt; cat TIMEOUT_PID.txt
echo $(date "+%Y-%m-%d %H:%M:%S")
echo "sleep start"
sleep 240
echo "sleep done"
echo $(date "+%Y-%m-%d %H:%M:%S")

if kill -0 $TIMEOUT_PID 2>/dev/null; then
  echo "Process seems hung ..."

 # Find the pytest process ID
 PYTEST_PID=$(pgrep -f "python3 -m pytest -m not generate -n 8")

 echo "PYTEST_PID 1"
 echo $PYTEST_PID
 echo $PYTEST_PID > PYTEST_PID.txt
 echo "PYTEST_PID 2"
 cat PYTEST_PID.txt
 echo "PYTEST_PID 3"
 echo $(cat PYTEST_PID.txt)

 # Monitor with timeout
 MONITOR_START=$(date +%s)
 MONITOR_TIMEOUT=600  # 10 minutes max

 while true; do
   # Check monitoring timeout
   CURRENT_TIME=$(date +%s)
   if [ $((CURRENT_TIME - MONITOR_START)) -gt $MONITOR_TIMEOUT ]; then
     echo "Monitoring timeout reached, exiting"
     break
   fi

   # Find the pytest process ID
   PYTEST_PID=$(pgrep -f "python3 -m pytest -m not generate -n 8")

   if [ -n "$PYTEST_PID" ]; then
     echo "=== $(date) ==="
     echo "PID: $PYTEST_PID"

     # Add error handling for each command
     if [ -f "/proc/$PYTEST_PID/status" ]; then
       cat /proc/$PYTEST_PID/status | grep -E "State|VmRSS|Threads|voluntary|nonvoluntary"
     else
       echo "Process $PYTEST_PID no longer exists"
       break
     fi

     CPU=$(ps -p $PYTEST_PID -o %cpu --no-headers 2>/dev/null)
     echo "CPU: ${CPU:-N/A}%"

     if [ -d "/proc/$PYTEST_PID/fd" ]; then
       echo "FDs: $(ls /proc/$PYTEST_PID/fd/ 2>/dev/null | wc -l)"
     fi

     # Try to get kernel stack if permissions allow
     if [ -r "/proc/$PYTEST_PID/stack" ]; then
       echo "Kernel stack:"
       cat /proc/$PYTEST_PID/stack
     fi
   else
     echo "No pytest process found"
     break
   fi
   sleep 30
 done

 # Clean up
 echo "Killing timeout process..."
 kill $TIMEOUT_PID 2>/dev/null || true
 exit 1
fi