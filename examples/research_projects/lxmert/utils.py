"""
coding=utf-8
Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal, Huggingface team :)
Adapted From Facebook Inc, Detectron2

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.import copy
"""

import copy
import fnmatch
import json
import os
import pickle as pkl
import shutil
import sys
import tarfile
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np
import requests
import wget
from filelock import FileLock
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from tqdm.auto import tqdm
from yaml import Loader, dump, load


try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False


try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )

default_cache_path = os.path.join(torch_cache_home, "transformers")

CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"
S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
PATH = "/".join(str(Path(__file__).resolve()).split("/")[:-1])
CONFIG = os.path.join(PATH, "config.yaml")
ATTRIBUTES = os.path.join(PATH, "attributes.txt")
OBJECTS = os.path.join(PATH, "objects.txt")
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.yaml"


def load_labels(objs=OBJECTS, attrs=ATTRIBUTES):
    vg_classes = []
    with open(objs) as f:
        for object in f.readlines():
            vg_classes.append(object.split(",")[0].lower().strip())

    vg_attrs = []
    with open(attrs) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(",")[0].lower().strip())
    return vg_classes, vg_attrs


def load_checkpoint(ckp):
    r = OrderedDict()
    with open(ckp, "rb") as f:
        ckp = pkl.load(f)["model"]
    for k in copy.deepcopy(list(ckp.keys())):
        v = ckp.pop(k)
        if isinstance(v, np.ndarray):
            v = torch.tensor(v)
        else:
            assert isinstance(v, torch.tensor), type(v)
        r[k] = v
    return r


class Config:
    _pointer = {}

    def __init__(self, dictionary: dict, name: str = "root", level=0):
        self._name = name
        self._level = level
        d = {}
        for k, v in dictionary.items():
            if v is None:
                raise ValueError()
            k = copy.deepcopy(k)
            v = copy.deepcopy(v)
            if isinstance(v, dict):
                v = Config(v, name=k, level=level + 1)
            d[k] = v
            setattr(self, k, v)

        self._pointer = d

    def __repr__(self):
        return str(list((self._pointer.keys())))

    def __setattr__(self, key, val):
        self.__dict__[key] = val
        self.__dict__[key.upper()] = val
        levels = key.split(".")
        last_level = len(levels) - 1
        pointer = self._pointer
        if len(levels) > 1:
            for i, l in enumerate(levels):
                if hasattr(self, l) and isinstance(getattr(self, l), Config):
                    setattr(getattr(self, l), ".".join(levels[i:]), val)
                if l == last_level:
                    pointer[l] = val
                else:
                    pointer = pointer[l]

    def to_dict(self):
        return self._pointer

    def dump_yaml(self, data, file_name):
        with open(f"{file_name}", "w") as stream:
            dump(data, stream)

    def dump_json(self, data, file_name):
        with open(f"{file_name}", "w") as stream:
            json.dump(data, stream)

    @staticmethod
    def load_yaml(config):
        with open(config) as stream:
            data = load(stream, Loader=Loader)
        return data

    def __str__(self):
        t = "    "
        if self._name != "root":
            r = f"{t * (self._level-1)}{self._name}:\n"
        else:
            r = ""
        level = self._level
        for i, (k, v) in enumerate(self._pointer.items()):
            if isinstance(v, Config):
                r += f"{t * (self._level)}{v}\n"
                self._level += 1
            else:
                r += f"{t * (self._level)}{k}: {v} ({type(v).__name__})\n"
            self._level = level
        return r[:-1]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls(config_dict)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            config_file = hf_bucket_url(pretrained_model_name_or_path, filename=CONFIG_NAME, use_cdn=False)

        try:
            # Load from URL or cache if already cached
            resolved_config_file = cached_path(
                config_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
            )
            # Load config dict
            if resolved_config_file is None:
                raise EnvironmentError

            config_file = Config.load_yaml(resolved_config_file)

        except EnvironmentError:
            msg = "Can't load config for"
            raise EnvironmentError(msg)

        if resolved_config_file == config_file:
            print("loading configuration file from path")
        else:
            print("loading configuration file cache")

        return Config.load_yaml(resolved_config_file), kwargs


# quick compare tensors
def compare(in_tensor):
    out_tensor = torch.load("dump.pt", map_location=in_tensor.device)
    n1 = in_tensor.numpy()
    n2 = out_tensor.numpy()[0]
    print(n1.shape, n1[0, 0, :5])
    print(n2.shape, n2[0, 0, :5])
    assert np.allclose(n1, n2, rtol=0.01, atol=0.1), (
        f"{sum([1 for x in np.isclose(n1, n2, rtol=0.01, atol=0.1).flatten() if x is False])/len(n1.flatten())*100:.4f} %"
        " element-wise mismatch"
    )
    raise Exception("tensors are all good")

    # Hugging face functions below


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def hf_bucket_url(model_id: str, filename: str, use_cdn=True) -> str:
    endpoint = CLOUDFRONT_DISTRIB_PREFIX if use_cdn else S3_BUCKET_PREFIX
    legacy_format = "/" not in model_id
    if legacy_format:
        return f"{endpoint}/{model_id}-{filename}"
    else:
        return f"{endpoint}/{model_id}/{filename}"


def http_get(
    url,
    temp_file,
    proxies=None,
    resume_size=0,
    user_agent=None,
):
    ua = "python/{}".format(sys.version.split()[0])
    if _torch_available:
        ua += "; torch/{}".format(torch.__version__)
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    headers = {"user-agent": ua}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    response = requests.get(url, stream=True, proxies=proxies, headers=headers)
    if response.status_code == 416:  # Range not satisfiable
        return
    content_length = response.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc="Downloading",
    )
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(
    url,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent=None,
    local_files_only=False,
):
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    etag = None
    if not local_files_only:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
            if response.status_code == 200:
                etag = response.headers.get("ETag")
        except (EnvironmentError, requests.exceptions.Timeout):
            # etag is already None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None = we don't have a connection, or url doesn't exist, or is otherwise inaccessible.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise ValueError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                return None

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "a+b") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            print(
                "%s not found in cache or force_download set to True, downloading to %s",
                url,
                temp_file.name,
            )

            http_get(
                url,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                user_agent=user_agent,
            )

        os.replace(temp_file.name, cache_path)

        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


def url_to_filename(url, etag=None):
    url_bytes = url.encode("utf-8")
    url_hash = insecure_hashlib.sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = insecure_hashlib.sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent=None,
    extract_compressed_file=False,
    force_extract=False,
    local_files_only=False,
):
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError("Archive format of {} could not be identified".format(output_path))

        return output_path_extracted

    return output_path


def get_data(query, delim=","):
    assert isinstance(query, str)
    if os.path.isfile(query):
        with open(query) as f:
            data = eval(f.read())
    else:
        req = requests.get(query)
        try:
            data = requests.json()
        except Exception:
            data = req.content.decode()
            assert data is not None, "could not connect"
            try:
                data = eval(data)
            except Exception:
                data = data.split("\n")
        req.close()
    return data


def get_image_from_url(url):
    response = requests.get(url)
    img = np.array(Image.open(BytesIO(response.content)))
    return img


# to load legacy frcnn checkpoint from detectron
def load_frcnn_pkl_from_url(url):
    fn = url.split("/")[-1]
    if fn not in os.listdir(os.getcwd()):
        wget.download(url)
    with open(fn, "rb") as stream:
        weights = pkl.load(stream)
    model = weights.pop("model")
    new = {}
    for k, v in model.items():
        new[k] = torch.from_numpy(v)
        if "running_var" in k:
            zero = torch.tensor([0])
            k2 = k.replace("running_var", "num_batches_tracked")
            new[k2] = zero
    return new


def get_demo_path():
    print(f"{os.path.abspath(os.path.join(PATH, os.pardir))}/demo.ipynb")


def img_tensorize(im, input_format="RGB"):
    assert isinstance(im, str)
    if os.path.isfile(im):
        img = cv2.imread(im)
    else:
        img = get_image_from_url(im)
        assert img is not None, f"could not connect to: {im}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if input_format == "RGB":
        img = img[:, :, ::-1]
    return img


def chunk(images, batch=1):
    return (images[i : i + batch] for i in range(0, len(images), batch))
