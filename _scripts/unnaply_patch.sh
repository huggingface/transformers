# create a patch but exclude the _scripts directory
cd /root/repos/new-model-addition-cohere-july-2025

git restore --staged .
git checkout .


rm -rf _scripts/ src/transformers/models/cohere2_vision/ tests/models/cohere2_vision/ tests/test_images/