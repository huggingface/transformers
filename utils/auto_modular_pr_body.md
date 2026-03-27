## Summary
- Auto-generated modular integration for `{model_name}`
- `modular_{model_name}.py` written via HF Inference API guided by `modular_model_detector.py`
- `modeling_{model_name}.py` regenerated from modular via `modular_model_converter.py`

## Test plan
- [ ] Review `modular_{model_name}.py` inheritance and overrides for correctness
- [ ] Run `python utils/modular_model_converter.py --files src/transformers/models/{model_name}/modular_{model_name}.py` and verify the output matches
- [ ] Add model to `__init__.py`, `auto` mappings, and configuration files
- [ ] Run model-specific tests

Generated via `utils/auto_modular_pr.py`
