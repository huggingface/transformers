#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1 && export PYTHONUNBUFFERED=1 && export OMP_NUM_THREADS=1 && export TRANSFORMERS_IS_CI=true && export PYTEST_TIMEOUT=120 && export RUN_PIPELINE_TESTS=false && export RUN_FLAKY=true
apt-get install -y net-tools iproute2
python3 utils/fetch_hub_objects_for_ci.py

echo $(date "+%Y-%m-%d %H:%M:%S")
timeout 600  ./pytest.sh & TIMEOUT_PID=$!; echo $TIMEOUT_PID; echo $TIMEOUT_PID > TIMEOUT_PID.txt; cat TIMEOUT_PID.txt
echo $(date "+%Y-%m-%d %H:%M:%S")
echo "sleep start"
sleep 80
echo "sleep done"
echo $(date "+%Y-%m-%d %H:%M:%S")

if kill -0 $TIMEOUT_PID 2>/dev/null; then
  echo "Process seems hung ..."

 # Find the pytest process ID
 PYTEST_PID=$(pgrep -f "python3 -m pytest -m not generate -n 2")

 echo "PYTEST_PID 1"
 echo $PYTEST_PID
 echo $PYTEST_PID > PYTEST_PID.txt
 echo "PYTEST_PID 2"
 cat PYTEST_PID.txt
 echo "PYTEST_PID 3"
 echo $(cat PYTEST_PID.txt)

 # Monitor with timeout
 MONITOR_START=$(date +%s)
 MONITOR_TIMEOUT=420  # 7 minutes max

 while true; do
   # Check monitoring timeout
   CURRENT_TIME=$(date +%s)
   if [ $((CURRENT_TIME - MONITOR_START)) -gt $MONITOR_TIMEOUT ]; then
     echo "Monitoring timeout reached, exiting"
     break
   fi

   # Find the pytest process ID
   PYTEST_PID=$(pgrep -f "python3 -m pytest -m not generate -n 2")

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
      FD_COUNT=$(ls /proc/$PYTEST_PID/fd/ 2>/dev/null | wc -l)
      echo "FDs: $FD_COUNT"

      # List file descriptors with details
      echo "File descriptors:"
      ls -la /proc/$PYTEST_PID/fd/ 2>/dev/null | head -20
      echo "FD types summary:"
      ls -l /proc/$PYTEST_PID/fd/ 2>/dev/null | awk '{print $NF}' | sort | uniq -c
    fi

    # List all threads
    echo "Thread details:"
    if [ -d "/proc/$PYTEST_PID/task" ]; then
      for tid in /proc/$PYTEST_PID/task/*; do
        if [ -d "$tid" ]; then
          TID_NUM=$(basename "$tid")
          THREAD_NAME=$(cat "$tid/comm" 2>/dev/null || echo "unknown")
          THREAD_STATE=$(cat "$tid/stat" 2>/dev/null | awk '{print $3}' || echo "?")
          echo "  TID $TID_NUM: $THREAD_NAME (state: $THREAD_STATE)"
        fi
      done
    fi

  echo "Monitor socket activity:"
  netstat -p | grep $PYTEST_PID
  ss -p | grep $PYTEST_PID

  echo "See what the sockets are connected to"
  netstat -p 2>/dev/null | grep $PYTEST_PID | head -10

  # Your existing thread details section, but enhanced:
echo "Socket details by file descriptor:"
echo "Socket and Thread Analysis:"
echo ""

# Get all CLOSE_WAIT connections for this PID
ss -anp | grep "$PYTEST_PID" | grep CLOSE-WAIT | while read line; do
  # Extract connection details
  LOCAL_PORT=$(echo "$line" | awk '{print $4}' | cut -d: -f2)
  REMOTE_PORT=$(echo "$line" | awk '{print $5}' | cut -d: -f2)
  FD_NUM=$(echo "$line" | grep -o 'fd=[0-9]*' | cut -d= -f2)

  # Get socket ID
  SOCKET_ID=""
  if [ -n "$FD_NUM" ] && [ -L "/proc/$PYTEST_PID/fd/$FD_NUM" ]; then
    SOCKET_ID=$(readlink "/proc/$PYTEST_PID/fd/$FD_NUM" 2>/dev/null | sed 's/socket:\[\([0-9]*\)\]/\1/')
  fi

  echo "=== Connection Analysis ==="
  echo "Local Port: $LOCAL_PORT, Remote Port: $REMOTE_PORT"
  echo "FD: $FD_NUM, Socket ID: $SOCKET_ID"
  echo "Full line: $line"
  echo ""

  # Try to find if remote port is still active anywhere
  echo "Checking for remote port $REMOTE_PORT usage:"

  # Check if remote port exists as a listening port
  ss -tln | grep ":$REMOTE_PORT " 2>/dev/null | sed 's/^/  Listening: /' || echo "  No listener on port $REMOTE_PORT"

  # Check if any process is using remote port
  ss -anp | grep ":$REMOTE_PORT" 2>/dev/null | grep -v "$line" | sed 's/^/  Other usage: /' || echo "  No other usage of port $REMOTE_PORT"

  # Check if any thread is specifically associated with this port
  lsof -i :$REMOTE_PORT 2>/dev/null | sed 's/^/  lsof: /' || echo "  lsof: No info for port $REMOTE_PORT"

  echo ""

  # Try to find which thread might be handling this FD
  echo "Thread analysis for FD $FD_NUM:"
  if [ -d "/proc/$PYTEST_PID/task" ]; then
    for tid in /proc/$PYTEST_PID/task/*; do
      TID_NUM=$(basename "$tid")
      THREAD_NAME=$(cat "$tid/comm" 2>/dev/null || echo "unknown")
      THREAD_STATE=$(cat "$tid/stat" 2>/dev/null | awk '{print $3}' || echo "?")

      # Try to see if this thread is doing anything with this FD
      THREAD_INFO="TID $TID_NUM: $THREAD_NAME (state: $THREAD_STATE)"

      # Check if thread is blocked on socket operations
      if [ -r "$tid/stack" ]; then
        STACK_TOP=$(cat "$tid/stack" 2>/dev/null | head -1)
        if echo "$STACK_TOP" | grep -q -E "(socket|tcp|net|select|poll)" 2>/dev/null; then
          THREAD_INFO="$THREAD_INFO [Network I/O: $STACK_TOP]"
        fi
      fi

      echo "  $THREAD_INFO"
    done
  fi

  echo ""
  echo "----------------------------------------"
  echo ""
done

echo "Summary of all active connections for PID $PYTEST_PID:"
ss -anp | grep "$PYTEST_PID" 2>/dev/null | sed 's/^/  /'

echo ""
echo "Summary of all threads:"
if [ -d "/proc/$PYTEST_PID/task" ]; then
  for tid in /proc/$PYTEST_PID/task/*; do
    TID_NUM=$(basename "$tid")
    THREAD_NAME=$(cat "$tid/comm" 2>/dev/null || echo "unknown")
    THREAD_STATE=$(cat "$tid/stat" 2>/dev/null | awk '{print $3}' || echo "?")
    echo "  TID $TID_NUM: $THREAD_NAME (state: $THREAD_STATE)"
  done
fi

  echo "Monitor if sockets are changing"
  echo "Socket fingerprint: $(ls /proc/$PYTEST_PID/fd/ | grep socket | wc -l)"

     # Try to get kernel stack if permissions allow
     if [ -r "/proc/$PYTEST_PID/stack" ]; then
       echo "Kernel stack:"
       cat /proc/$PYTEST_PID/stack
     fi
   else
     echo "No pytest process found"
     kill $TIMEOUT_PID 2>/dev/null || true
     exit 0
   fi
   sleep 1
 done

 # Clean up
 echo "Killing timeout process..."
 kill $TIMEOUT_PID 2>/dev/null || true
 exit 1
fi