#!/bin/bash
source ~/.bashrc
echo "running docker-entrypoint.sh"
conda activate container
echo $KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS
echo "printed TPU info"
export XRT_TPU_CONFIG="tpu_worker;0;${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS:7}"
exec "$@"#!/bin/bash
