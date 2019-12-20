"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
import json
import logging
import os
import six
import shutil
import tempfile
import fnmatch
from functools import wraps
from hashlib import sha256
from io import open

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import requests
from tqdm.auto import tqdm
from contextlib import contextmanager
from . import __version__

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

try:
    os.environ.setdefault('USE_TORCH', 'YES')
    if os.environ['USE_TORCH'].upper() in ('1', 'ON', 'YES'):
        import torch
        _torch_available = True  # pylint: disable=invalid-name
        logger.info("PyTorch version {} available.".format(torch.__version__))
    else:
        logger.info("USE_TORCH override through env variable, disabling PyTorch")
        _torch_available = False
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name

try:
    os.environ.setdefault('USE_TF', 'YES')
    if os.environ['USE_TF'].upper() in ('1', 'ON', 'YES'):
        import tensorflow as tf
        assert hasattr(tf, '__version__') and int(tf.__version__[0]) >= 2
        _tf_available = True  # pylint: disable=invalid-name
        logger.info("TensorFlow version {} available.".format(tf.__version__))
    else:
        logger.info("USE_TF override through env variable, disabling Tensorflow")
        _tf_available = False
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name

try:
    from torch.hub import _get_torch_home
    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'transformers')

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(
        os.getenv('PYTORCH_TRANSFORMERS_CACHE', os.getenv('PYTORCH_PRETRAINED_BERT_CACHE', default_cache_path)))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_TRANSFORMERS_CACHE',
                                              os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                        default_cache_path))

PYTORCH_TRANSFORMERS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility
TRANSFORMERS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = 'tf_model.h5'
TF_WEIGHTS_NAME = 'model.ckpt'
CONFIG_NAME = "config.json"
MODEL_CARD_NAME = "modelcard.json"

DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://d2ws9o8vfrpkyk.cloudfront.net"


def is_torch_available():
    return _torch_available

def is_tf_available():

    return _tf_available

if not six.PY2:
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            fn.__doc__ = ''.join(docstr) + fn.__doc__
            return fn
        return docstring_decorator

    def add_end_docstrings(*docstr):
        def docstring_decorator(fn):
            fn.__doc__ = fn.__doc__ + ''.join(docstr)
            return fn
        return docstring_decorator
else:
    # Not possible to update class docstrings on python2
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            return fn
        return docstring_decorator

    def add_end_docstrings(*docstr):
        def docstring_decorator(fn):
            return fn
        return docstring_decorator


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ('http', 'https', 's3')

def hf_bucket_url(identifier, postfix=None, cdn=False):
    endpoint = CLOUDFRONT_DISTRIB_PREFIX if cdn else S3_BUCKET_PREFIX
    if postfix is None:
        return "/".join((endpoint, identifier))
    else:
        return "/".join((endpoint, identifier, postfix))


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    if url.endswith('.h5'):
        filename += '.h5'

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']

    return url, etag


def cached_path(url_or_filename, cache_dir=None, force_download=False, proxies=None, resume_download=False, user_agent=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletly recieved file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir=cache_dir,
            force_download=force_download, proxies=proxies,
            resume_download=resume_download, user_agent=user_agent)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif urlparse(url_or_filename).scheme == '':
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url, proxies=None):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3", config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file, proxies=None):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3", config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url, temp_file, proxies=None, resume_size=0, user_agent=None):
    ua = "transformers/{}; python/{}".format(__version__, sys.version.split()[0])
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(
            "{}/{}".format(k, v) for k, v in user_agent.items()
        )
    elif isinstance(user_agent, six.string_types):
        ua += "; "+ user_agent
    headers = {
        "user-agent": ua
    }
    if resume_size > 0:
        headers['Range'] = 'bytes=%d-' % (resume_size,)
    response = requests.get(url, stream=True, proxies=proxies, headers=headers)
    if response.status_code == 416:  # Range not satisfiable
        return
    content_length = response.headers.get('Content-Length')
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", unit_scale=True, total=total, initial=resume_size,
                    desc="Downloading", disable=bool(logger.level<=logging.INFO))
    for chunk in response.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url, cache_dir=None, force_download=False, proxies=None, etag_timeout=10, resume_download=False, user_agent=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if sys.version_info[0] == 2 and not isinstance(cache_dir, str):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url, proxies=proxies)
    else:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
            if response.status_code != 200:
                etag = None
            else:
                etag = response.headers.get("ETag")
        except (EnvironmentError, requests.exceptions.Timeout):
            etag = None

    if sys.version_info[0] == 2 and etag is not None:
        etag = etag.decode('utf-8')
    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # If we don't have a connection (etag is None) and can't identify the file
    # try to get the last downloaded one
    if not os.path.exists(cache_path) and etag is None:
        matching_files = fnmatch.filter(os.listdir(cache_dir), filename + '.*')
        matching_files = list(filter(lambda s: not s.endswith('.json'), matching_files))
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])

    if resume_download:
        incomplete_path = cache_path + '.incomplete'
        @contextmanager
        def _resumable_file_manager():
            with open(incomplete_path,'a+b') as f:
                yield f
            os.remove(incomplete_path)
        temp_file_manager = _resumable_file_manager
        if os.path.exists(incomplete_path):
            resume_size = os.stat(incomplete_path).st_size
        else:
            resume_size = 0
    else:
        temp_file_manager = tempfile.NamedTemporaryFile
        resume_size = 0

    if etag is not None and (not os.path.exists(cache_path) or force_download):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("%s not found in cache or force_download set to True, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                if resume_download:
                    logger.warn('Warning: resumable downloads are not implemented for "s3://" urls')
                s3_get(url, temp_file, proxies=proxies)
            else:
                http_get(url, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                output_string = json.dumps(meta)
                if sys.version_info[0] == 2 and isinstance(output_string, str):
                    output_string = unicode(output_string, 'utf-8')  # The beauty of python 2
                meta_file.write(output_string)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path
