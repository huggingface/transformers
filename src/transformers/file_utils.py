"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import fnmatch
import json
import linecache
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial, wraps
from hashlib import sha256
from typing import Optional
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from filelock import FileLock
from tqdm.auto import tqdm

from . import __version__


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
    if USE_TORCH in ("1", "ON", "YES", "AUTO") and USE_TF not in ("1", "ON", "YES"):
        import torch

        _torch_available = True  # pylint: disable=invalid-name
        logger.info("PyTorch version {} available.".format(torch.__version__))
    else:
        logger.info("Disabling PyTorch because USE_TF is set")
        _torch_available = False
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

    if USE_TF in ("1", "ON", "YES", "AUTO") and USE_TORCH not in ("1", "ON", "YES"):
        import tensorflow as tf

        assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2
        _tf_available = True  # pylint: disable=invalid-name
        logger.info("TensorFlow version {} available.".format(tf.__version__))
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set")
        _tf_available = False
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name

try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )
default_cache_path = os.path.join(torch_cache_home, "transformers")

try:
    from pathlib import Path

    PYTORCH_PRETRAINED_BERT_CACHE = Path(
        os.getenv("PYTORCH_TRANSFORMERS_CACHE", os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path))
    )
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv(
        "PYTORCH_TRANSFORMERS_CACHE", os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
    )

PYTORCH_TRANSFORMERS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility
TRANSFORMERS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
CONFIG_NAME = "config.json"
MODEL_CARD_NAME = "modelcard.json"


MULTIPLE_CHOICE_DUMMY_INPUTS = [[[0], [1]], [[0], [1]]]
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://d2ws9o8vfrpkyk.cloudfront.net"


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        class_name = ":class:`~transformers.{}`".format(fn.__qualname__.split(".")[0])
        intro = "   The {} forward method, overrides the :func:`__call__` special method.".format(class_name)
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        pre and post processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + "".join(docstr)
        return fn

    return docstring_decorator


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https", "s3")


def hf_bucket_url(identifier, postfix=None, cdn=False) -> str:
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
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


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
) -> Optional[str]:
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
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and overide the folder where it was extracted.

    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
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
    if is_torch_available():
        ua += "; torch/{}".format(torch.__version__)
    if is_tf_available():
        ua += "; tensorflow/{}".format(tf.__version__)
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
        disable=bool(logger.getEffectiveLevel() == logging.NOTSET),
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
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache.
    If it's not there, download it. Then return the path to the cached file.

    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    etag = None
    if not local_files_only:
        # Get eTag to add to filename, if it exists.
        if url.startswith("s3://"):
            etag = s3_etag(url, proxies=proxies)
        else:
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
            logger.info("%s not found in cache or force_download set to True, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                if resume_download:
                    logger.warn('Warning: resumable downloads are not implemented for "s3://" urls')
                s3_get(url, temp_file, proxies=proxies)
            else:
                http_get(url, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent)

        logger.info("storing %s in cache at %s", url, cache_path)
        os.rename(temp_file.name, cache_path)

        logger.info("creating metadata file for %s", cache_path)
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


_memory_tracing_enabled = False
Frame = namedtuple("Frame", ["filename", "module", "line_number", "event", "line_text"])
UsedMemoryState = namedtuple("UsedMemoryState", ["frame", "rss_memory"])
IncreasedMemoryState = namedtuple("IncreasedMemoryState", ["frame", "rss_memory_increase"])


def start_memory_tracing(modules_to_trace=None, modules_not_to_trace=None, events_to_trace="line"):
    """ Setup line-by-line tracing to record rss mem (RAM) at each line of a module or sub-module.
        See `../../examples/benchmarks.py for a usage example.
        Current memory consumption is returned using psutil and in particular is the RSS memory
            "Resident Set Size” (the non-swapped physical memory the process is using).
            See https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

        Args:
            - `modules_to_trace`: (None, string, list/tuple of string)
                if None, all events are recorded
                if string or list of strings: only events from the listed module/sub-module will be recorded (e.g. 'fairseq' or 'transformers.modeling_gpt2')
            - `modules_not_to_trace`: (None, string, list/tuple of string)
                if None, no module is avoided
                if string or list of strings: events from the listed module/sub-module will not be recorded (e.g. 'torch')
            - `events_to_trace`: string or list of string of events to be recorded (see official python doc for `sys.settrace` for the list of events)
                default to line

        Return: Tuple with two lists which will be updated during tracing:
            - `used_memory_list` is a list of `UsedMemoryState` for each event (default each line of the traced script).
                - `UsedMemoryState` are named tuples with the following fields:
                    - 'frame': a `Frame` namedtuple (see below) storing information on the current tracing frame (current file, location in current file)
                    - 'rss_memory': RSS memory state *before* executing the line

                `Frame` namedtuple used by `UsedMemoryState` to list the current frame state.
                    The have the following fields:
                    - 'filename' (string): Name of the file currently executed
                    - 'module' (string): Name of the module currently executed
                    - 'line_number' (int): Number of the line currently executed
                    - 'event' (string): Event that triggered the tracing (default will be "line")
                    - 'line_text' (string): Text of the line in the python script

    """
    try:
        import psutil
    except (ImportError):
        raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

    memory_trace = []
    process = psutil.Process(os.getpid())

    def traceit(frame, event, args):
        """ Tracing method executed before running each line in a module or sub-module
            Record memory allocated in a list with debugging information
        """
        global _memory_tracing_enabled

        if not _memory_tracing_enabled:
            return traceit

        # Filter events
        if events_to_trace is not None:
            if isinstance(events_to_trace, str) and event != events_to_trace:
                return traceit
            elif isinstance(events_to_trace, (list, tuple)) and event not in events_to_trace:
                return traceit

        # Filter modules
        name = frame.f_globals["__name__"]
        if not isinstance(name, str):
            return traceit
        else:
            # Filter whitelist of modules to trace
            if modules_to_trace is not None:
                if isinstance(modules_to_trace, str) and modules_to_trace not in name:
                    return traceit
                elif isinstance(modules_to_trace, (list, tuple)) and all(m not in name for m in modules_to_trace):
                    return traceit

            # Filter blacklist of modules not to trace
            if modules_not_to_trace is not None:
                if isinstance(modules_not_to_trace, str) and modules_not_to_trace in name:
                    return traceit
                elif isinstance(modules_not_to_trace, (list, tuple)) and any(m in name for m in modules_not_to_trace):
                    return traceit

        # Record current tracing state (file, location in file...)
        lineno = frame.f_lineno
        filename = frame.f_globals["__file__"]
        if filename.endswith(".pyc") or filename.endswith(".pyo"):
            filename = filename[:-1]
        line = linecache.getline(filename, lineno).rstrip()

        # Record current memory state (rss memory) and compute difference with previous memory state
        mem = process.memory_info()
        traced_state = Frame(filename, name, lineno, event, line)
        mem_state = UsedMemoryState(traced_state, mem.rss)
        memory_trace.append(mem_state)

        return traceit

    sys.settrace(traceit)

    global _memory_tracing_enabled
    _memory_tracing_enabled = True

    return memory_trace


TraceMemoryIncrease = namedtuple("TraceMemoryIncrease", ["frame", "bytes", "string"])
TotalMemoryIncrease = namedtuple("TotalMemoryIncrease", ["bytes", "string"])
MemorySummary = namedtuple("MemorySummary", ["sequential", "cumulative", "total"])


def stop_memory_tracing(memory_trace=None):
    """ Stop memory tracing cleanly and return a summary of the memory trace if a trace is given.

        Args:
            - memory_trace (optional, default: None): memmory trace to convert in summary

        Return:
            - None if `memory_trace` is None
            - `MemorySummary` namedtuple otherwise with the fields:
                - `sequential`: the list of tuple (Frame, memory increase) computed from the memory_trace list
                    by substracting the memory after executing each line from the memory before executing said line.
                - `cumulative`: an OrderedDict of (Frame, cumulative memory increase for the line) with the cumulative increase in memory for each line
                    (summing repeted memory increase for a line if it's executed several times).
                    The dictionnary is ordered from the line with the largest memory consumption to the line with the smallest (can be negative if memory is free)
                - `total`: total memory increaseduring the tracing.

        In the `MemorySummary`, frames are `Frame` namedtuple used to list the current frame state. A `Frame` has the following fields:
            - 'filename' (string): Name of the file currently executed
            - 'module' (string): Name of the module currently executed
            - 'line_number' (int): Number of the line currently executed
            - 'event' (string): Event that triggered the tracing (default will be "line")
            - 'line_text' (string): Text of the line in the python script
    """
    global _memory_tracing_enabled
    _memory_tracing_enabled = False

    if memory_trace is not None and len(memory_trace) > 1:
        memory_diff_trace = []
        cumulative_memory_dict = defaultdict(lambda: 0)
        for (frame, mem), (next_frame, next_mem) in zip(memory_trace[:-1], memory_trace[1:]):
            mem_inc = next_mem - mem
            mem_str = bytes_to_human_readable(mem_inc)
            memory_diff_trace.append(TraceMemoryIncrease(frame=frame, bytes=mem_inc, string=mem_str))
            cumulative_memory_dict[frame] += mem_inc

        cumulative_memory = sorted(list(cumulative_memory_dict.items()), key=lambda x: x[1], reverse=True)
        cumulative_memory = list(
            TraceMemoryIncrease(frame=frame, bytes=mem_inc, string=bytes_to_human_readable(mem_inc))
            for frame, mem_inc in cumulative_memory
        )

        total_memory = sum(step_trace.bytes for step_trace in memory_diff_trace)
        total_memory = TotalMemoryIncrease(bytes=total_memory, string=bytes_to_human_readable(total_memory))
        return MemorySummary(sequential=memory_diff_trace, cumulative=cumulative_memory, total=total_memory)

    return None


def bytes_to_human_readable(memory_amount):
    """ Utility to convert a number of bytes (int) in a human readable string (with units)
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if memory_amount > -1024.0 and memory_amount < 1024.0:
            return "{:.3f}{}".format(memory_amount, unit)
        memory_amount /= 1024.0
    return "{:.3f}TB".format(memory_amount)
