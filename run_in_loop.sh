#!/usr/bin/env bash
for i in {0..40}; do
	RUN_PT_TF_CROSS_TESTS=1 pytest tests/test_modeling_tf_led.py::TFLEDModelTest::test_pt_tf_model_equivalence
done

