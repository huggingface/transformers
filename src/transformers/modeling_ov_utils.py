import os
from functools import partial
from pickle import UnpicklingError
from typing import Dict, Set, Tuple, Union

from .configuration_utils import PretrainedConfig
from .file_utils import (
    OV_WEIGHTS_NAME,
    PushToHubMixin,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
    cached_path,
    copy_func,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    replace_return_docstrings,
)
from .utils import logging


logger = logging.get_logger(__name__)
from openvino.inference_engine import IECore

ie = IECore()

class OpenVINOModel(object):
    def __init__(self, xml_path, bin_path):
        if not xml_path.endswith('.xml'):
            import shutil
            shutil.copyfile(xml_path, xml_path + ".xml")
            xml_path += ".xml"

        print(xml_path)
        print(bin_path)
        self.net = ie.read_network(xml_path, bin_path)


class OVPreTrainedModel(object):
    def __init__(self, xml_path):
        super().__init__()
        self.net = None


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_ov = kwargs.pop("from_ov", True)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "model", "framework": "openvino", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load model
        OV_BIN_NAME = OV_WEIGHTS_NAME.replace(".xml", ".bin")
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_ov and os.path.isfile(os.path.join(pretrained_model_name_or_path, OV_WEIGHTS_NAME)) and \
                   os.path.isfile(os.path.join(pretrained_model_name_or_path, OV_BIN_NAME)):
                    # Load from an OpenVINO IR
                    archive_files = [os.path.join(pretrained_model_name_or_path, OV_WEIGHTS_NAME) for name in [OV_WEIGHTS_NAME, OV_BIN_NAME]]
                # elif from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                #     # Load from a PyTorch checkpoint
                #     archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        f"Error no files named {[OV_WEIGHTS_NAME, OV_BIN_NAME]} found in directory "
                        f"{pretrained_model_name_or_path} or `from_ov` set to False"
                    )
            # elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            #     archive_file = pretrained_model_name_or_path
            else:
                archive_files = [hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=name,
                    revision=revision,
                ) for name in [OV_WEIGHTS_NAME, OV_BIN_NAME]]

            # redirect to the cache, if necessary
            try:
                resolved_archive_files = [cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                ) for archive_file in archive_files]
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n"
                    f"  (make sure '{pretrained_model_name_or_path}' is not a path to a local directory with something else, in that case)\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {OV_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_files == archive_files:
                logger.info(f"loading weights file {archive_files}")
            else:
                logger.info(f"loading weights file {archive_files} from cache at {resolved_archive_files}")
        else:
            resolved_archive_files = None

        return OpenVINOModel(*resolved_archive_files)
