#!/usr/bin/env bash
for i in {1..20}; do
#	pytest -s tests/test_modeling_wav2vec2.py::Wav2Vec2RobustModelTest::test_retain_grad_hidden_states_attentions
	pytest -s tests/test_modeling_wav2vec2.py
done;
