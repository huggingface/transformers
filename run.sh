#!/bin/bash


# cd project
# git fetch origin
# git pull origin debug_too_long_no_output


export CONTAINER_ID=$(docker run --memory=16g --privileged -d huggingface/transformers-torch-light:dev sleep 3600) && echo $CONTAINER_ID > CONTAINER_ID.txt
cat CONTAINER_ID.txt
echo "" > gdb_output.txt

docker cp pytest_prepare.sh $(cat CONTAINER_ID.txt):/pytest_prepare.sh
docker cp pytest.sh $(cat CONTAINER_ID.txt):/pytest.sh
docker cp gdb_commands.txt $(cat CONTAINER_ID.txt):/gdb_commands.txt
docker exec $(cat CONTAINER_ID.txt) ls -la /pytest_prepare.sh
docker exec $(cat CONTAINER_ID.txt) ls -la /pytest.sh
docker exec $(cat CONTAINER_ID.txt) chmod +x /pytest_prepare.sh
docker exec $(cat CONTAINER_ID.txt) chmod +x /pytest.sh

docker exec $(cat CONTAINER_ID.txt) /pytest_prepare.sh

echo $(date "+%Y-%m-%d %H:%M:%S")
timeout 600 docker exec $(cat CONTAINER_ID.txt) /pytest.sh & PYTEST_PID=$!; echo $PYTEST_PID; echo $PYTEST_PID > PYTEST_PID.txt; cat PYTEST_PID.txt
echo $(date "+%Y-%m-%d %H:%M:%S")
echo "sleep start"
sleep 240
echo "sleep done"
echo $(date "+%Y-%m-%d %H:%M:%S")

# if no more timeout command --> kill fail --> error --> null --> won't enter
# if still has timeout command process --> kill success
if kill -0 $PYTEST_PID 2>/dev/null; then
  echo "Process seems hung, launching GDB debugger..."

  # Install gdb in the container if needed
  docker exec $(cat CONTAINER_ID.txt) bash -c "apt-get update && apt-get install -y gdb && apt-get remove --purge needrestart -y && echo 'deb http://deb.debian.org/debian bullseye main' >> /etc/apt/sources.list && apt-get update && apt-get install -y python3.9-dbg && sed -i '/bullseye/d' /etc/apt/sources.list"

  # Find the pytest process ID inside container
  PYTEST_IN_CONTAINER_PID=$(docker exec $(cat CONTAINER_ID.txt) pgrep -f "python3 -m pytest -m not generate -n 8")
  echo "PYTEST_IN_CONTAINER_PID 1"
  echo $PYTEST_IN_CONTAINER_PID
  echo $PYTEST_IN_CONTAINER_PID > PYTEST_IN_CONTAINER_PID.txt
  echo "PYTEST_IN_CONTAINER_PID 2"
  cat PYTEST_IN_CONTAINER_PID.txt
  echo "PYTEST_IN_CONTAINER_PID 3"
  echo $(cat PYTEST_IN_CONTAINER_PID.txt)

  docker cp PYTEST_IN_CONTAINER_PID.txt $(cat CONTAINER_ID.txt):/PYTEST_IN_CONTAINER_PID.txt
  echo "PYTEST_IN_CONTAINER_PID 4"
  docker exec --privileged $(cat CONTAINER_ID.txt) cat PYTEST_IN_CONTAINER_PID.txt

  # Attach GDB to the hung process
  docker exec --privileged -it $(cat CONTAINER_ID.txt) gdb -batch -x gdb_commands.txt -p $(cat PYTEST_IN_CONTAINER_PID.txt) > gdb_output.txt 2>&1

  exit 1

#  # Kill the timeout process
#  echo "Killing timeout process PID: $PYTEST_PID"
#  kill $PYTEST_PID
#  echo "Timeout process killed"
#
#  sleep 3
#
#  # Optional: Verify it's gone
#  if kill -0 $PYTEST_PID 2>/dev/null; then
#    echo "Process still running, force killing..."
#    kill -9 $PYTEST_PID
#  else
#    echo "Process successfully terminated"
#  fi
#
#  sleep 3
#
#  # Optional: Verify it's gone
#  if kill -0 $PYTEST_PID 2>/dev/null; then
#    echo "Process still running, force killing..."
#    kill -9 $PYTEST_PID
#  else
#    echo "Process successfully terminated"
#  fi

  # source /usr/share/gdb/auto-load/usr/bin/python3.9-gdb.py
fi


# docker exec -it $(cat CONTAINER_ID.txt) bash
# pgrep -f "python3 -m pytest -m not generate -n 8"
# ps aux --sort pmem


