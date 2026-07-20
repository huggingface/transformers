#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Utility tests ==="
pytest tests/utils/test_image_processing_utils.py \
       tests/utils/test_feature_extraction_utils.py \
       tests/utils/test_image_utils.py \
       tests/utils/test_audio_utils.py \
       tests/utils/test_video_utils.py \
       -n 8 "$@"

echo "=== Base tests ==="
pytest tests/test_image_processing_common.py \
       tests/test_feature_extraction_common.py \
       tests/test_sequence_feature_extraction_common.py \
       tests/test_processing_common.py \
       tests/test_video_processing_common.py \
       tests/test_image_transforms.py \
       -n 8 "$@"

echo "=== Auto class tests ==="
pytest tests/models/auto/test_image_processing_auto.py \
       tests/models/auto/test_feature_extraction_auto.py \
       tests/models/auto/test_processor_auto.py \
       tests/models/auto/test_video_processing_auto.py \
       -n 8 "$@"

# echo "=== Per-model image processor tests ==="
# pytest tests/models/*/test_image_processing_*.py \
#        -n 8 "$@"

# echo "=== Per-model feature extractor tests ==="
# pytest tests/models/*/test_feature_extraction_*.py \
#        -n 8 "$@"

# echo "=== Per-model processor tests ==="
# pytest tests/models/*/test_processor*.py \
#        -n 8 "$@"

# echo "=== All preprocessing tests passed ==="
