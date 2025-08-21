import importlib
from unittest.mock import MagicMock, patch
import pytest


class TestAwqVersionCheck:
    """Test suite for AutoAWQ version checking in AwqConfig"""

    def test_awq_version_check_not_available(self):
        """Test that ValueError is raised when AutoAWQ is not available"""
        with patch('transformers.utils.quantization_config.is_auto_awq_available', return_value=False):
            from transformers.utils.quantization_config import AwqConfig
            
            with pytest.raises(ValueError, match="does not support module quantization skipping"):
                AwqConfig(bits=4, modules_to_not_convert=['lm_head'])

    def test_awq_version_check_old_version(self):
        """Test that ValueError is raised when AutoAWQ version is too old"""
        with patch('transformers.utils.quantization_config.is_auto_awq_available', return_value=True):
            with patch('importlib.metadata.version', return_value='0.1.7'):
                from transformers.utils.quantization_config import AwqConfig
                
                with pytest.raises(ValueError, match="does not support module quantization skipping"):
                    AwqConfig(bits=4, modules_to_not_convert=['lm_head'])

    def test_awq_version_check_new_version(self):
        """Test that no error is raised when AutoAWQ version is sufficient"""
        with patch('transformers.utils.quantization_config.is_auto_awq_available', return_value=True):
            with patch('importlib.metadata.version', return_value='0.1.8'):
                from transformers.utils.quantization_config import AwqConfig
                
                config = AwqConfig(bits=4, modules_to_not_convert=['lm_head'])
                assert config.modules_to_not_convert == ['lm_head']

    def test_awq_version_check_metadata_fails_but_module_works(self):
        """Test fallback when importlib.metadata.version fails but module version works"""
        mock_awq = MagicMock()
        mock_awq.__version__ = '0.1.8'
        
        with patch('transformers.utils.quantization_config.is_auto_awq_available', return_value=True):
            with patch('importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError('autoawq')):
                with patch.dict('sys.modules', {'awq': mock_awq}):
                    from transformers.utils.quantization_config import AwqConfig
                    
                    config = AwqConfig(bits=4, modules_to_not_convert=['lm_head'])
                    assert config.modules_to_not_convert == ['lm_head']

    @pytest.mark.parametrize("modules_value", [None, []])
    def test_no_modules_to_not_convert(self, modules_value):
        """Test that no version check occurs when modules_to_not_convert is None or empty"""
        with patch('transformers.utils.quantization_config.is_auto_awq_available', return_value=False):
            from transformers.utils.quantization_config import AwqConfig
            
            config = AwqConfig(bits=4, modules_to_not_convert=modules_value)
            assert config.modules_to_not_convert == modules_value
