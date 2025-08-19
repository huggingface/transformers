from unittest.mock import MagicMock

import pytest

import transformers.quantizers.quantizer_awq as _qa
from transformers.quantizers.quantizer_awq import AwqQuantizer
from transformers.utils.quantization_config import AWQLinearVersion


# Minimal shim for quantization_config
class _QCfg:
    def __init__(self, modules_to_not_convert):
        self.quant_method = "awq"
        self.version = AWQLinearVersion.GEMM  # any valid enum value is fine for these tests
        self.modules_to_not_convert = modules_to_not_convert
        self.do_fuse = False


# Minimal config objects (do NOT subclass PretrainedConfig to avoid attribute interception)
class DummyConfig:
    # Mimic enough attributes used by AwqQuantizer.validate_environment
    def __init__(self):
        self.quantization_config = _QCfg(["lm_head"])


class ConfigWithoutModules:
    def __init__(self):
        self.quantization_config = _QCfg(None)


def test_old_autoawq_raises(monkeypatch):
    # Allow validate_environment to proceed past both availability gates
    monkeypatch.setattr(_qa, "is_auto_awq_available", lambda: True)
    monkeypatch.setattr(_qa, "is_accelerate_available", lambda: True)
    # Mock GPU availability to prevent automatic IPEX conversion
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    mock_autoawq = MagicMock()
    # Set both common version attributes
    mock_autoawq.__version__ = "0.1.7"
    mock_autoawq.version = "0.1.7"
    monkeypatch.setitem(__import__("sys").modules, "autoawq", mock_autoawq)

    q = AwqQuantizer(DummyConfig().quantization_config)

    with pytest.raises(ImportError) as exc:
        q.validate_environment(device_map=None)

    assert "upgrade autoawq package to at least 0.1.8" in str(exc.value)


def test_new_autoawq_passes(monkeypatch):
    # Allow validate_environment to proceed past both availability gates
    monkeypatch.setattr(_qa, "is_auto_awq_available", lambda: True)
    monkeypatch.setattr(_qa, "is_accelerate_available", lambda: True)
    # Mock GPU availability to prevent automatic IPEX conversion
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    mock_autoawq = MagicMock()
    mock_autoawq.__version__ = "0.1.8"
    mock_autoawq.version = "0.1.8"
    monkeypatch.setitem(__import__("sys").modules, "autoawq", mock_autoawq)

    q = AwqQuantizer(DummyConfig().quantization_config)

    # Should not raise
    q.validate_environment(device_map=None)


def test_autoawq_not_installed(monkeypatch):
    """
    Expect the custom 'AutoAWQ >= 0.1.8 is required...' message from the version check.
    Force availability True to reach the version check, but fail import of 'autoawq'.
    """
    # Force past availability gates
    monkeypatch.setattr(_qa, "is_auto_awq_available", lambda: True)
    monkeypatch.setattr(_qa, "is_accelerate_available", lambda: True)
    # Mock GPU availability to prevent automatic IPEX conversion
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    sys = __import__("sys")
    if "autoawq" in sys.modules:
        monkeypatch.delitem(sys.modules, "autoawq")

    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "autoawq":
            raise ModuleNotFoundError("No module named 'autoawq'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    q = AwqQuantizer(DummyConfig().quantization_config)

    with pytest.raises(ImportError) as exc:
        q.validate_environment(device_map=None)

    assert "AutoAWQ >= 0.1.8 is required" in str(exc.value)


def test_no_modules_to_not_convert_no_check(monkeypatch):
    # Let availability gates pass
    monkeypatch.setattr(_qa, "is_auto_awq_available", lambda: True)
    monkeypatch.setattr(_qa, "is_accelerate_available", lambda: True)
    # Mock GPU availability to prevent automatic IPEX conversion
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    mock_autoawq = MagicMock()
    mock_autoawq.__version__ = "0.1.0"  # Should be ignored since no modules are skipped
    mock_autoawq.version = "0.1.0"
    monkeypatch.setitem(__import__("sys").modules, "autoawq", mock_autoawq)

    q = AwqQuantizer(ConfigWithoutModules().quantization_config)
    q.validate_environment(device_map=None)  # Should not raise


def test_empty_modules_list_no_check(monkeypatch):
    """
    If modules_to_not_convert is an empty list, do NOT enforce the autoawq>=0.1.8 gate.
    """
    # Let availability gates pass
    import transformers.quantizers.quantizer_awq as _qa

    monkeypatch.setattr(_qa, "is_auto_awq_available", lambda: True)
    monkeypatch.setattr(_qa, "is_accelerate_available", lambda: True)

    # Force GPU available to avoid auto-switch to IPEX
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    # Install a mock old autoawq; this should NOT be checked because list is empty
    from unittest.mock import MagicMock

    mock_autoawq = MagicMock()
    mock_autoawq.__version__ = "0.1.0"
    mock_autoawq.version = "0.1.0"

    import sys as _sys

    monkeypatch.setitem(_sys.modules, "autoawq", mock_autoawq)

    # Minimal quantization config with empty modules_to_not_convert
    from transformers.utils.quantization_config import AWQLinearVersion

    class _QCfg:
        def __init__(self):
            self.quant_method = "awq"
            self.version = AWQLinearVersion.GEMM
            self.modules_to_not_convert = []  # empty list -> should NOT trigger version check
            self.do_fuse = False

    from transformers.quantizers.quantizer_awq import AwqQuantizer

    q = AwqQuantizer(_QCfg())

    # Should NOT raise ImportError even though autoawq is "old"
    q.validate_environment(device_map=None)


def test_read_only_config_handles_gracefully(monkeypatch):
    """
    Test that when quantization_config doesn't allow setting version attribute,
    the code handles it gracefully and provides a helpful error message.
    """
    import transformers.quantizers.quantizer_awq as _qa

    monkeypatch.setattr(_qa, "is_auto_awq_available", lambda: True)
    monkeypatch.setattr(_qa, "is_accelerate_available", lambda: True)

    # Force no GPU/XPU available to trigger version switch logic
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.xpu.is_available", lambda: False)

    from unittest.mock import MagicMock
    import sys as _sys

    # Mock autoawq to be available
    mock_autoawq = MagicMock()
    mock_autoawq.__version__ = "0.2.6"
    mock_autoawq.version = "0.2.6"
    monkeypatch.setitem(_sys.modules, "autoawq", mock_autoawq)

    from transformers.utils.quantization_config import AWQLinearVersion

    # Create a read-only config that raises AttributeError when trying to set version
    class _ReadOnlyQCfg:
        def __init__(self):
            self.quant_method = "awq"
            self._version = AWQLinearVersion.GEMM
            self.modules_to_not_convert = None
            self.do_fuse = False

        @property
        def version(self):
            return self._version

        @version.setter
        def version(self, value):
            raise AttributeError("can't set attribute")

    from transformers.quantizers.quantizer_awq import AwqQuantizer

    q = AwqQuantizer(_ReadOnlyQCfg())

    # Should raise RuntimeError with helpful message, not AttributeError
    with pytest.raises(RuntimeError) as exc:
        q.validate_environment(device_map=None)

    assert "read-only" in str(exc.value)
