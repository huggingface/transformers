CUDA_VISIBLE_DEVICES=0 python3 tests2.py > test_hf.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 /llm_reco/lingzhixin/pub_models/models/versions_dev/v0_9_1/Keye-8B/tests2.py > test_bl.out 2>&1 &

