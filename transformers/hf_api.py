# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import os
from os.path import expanduser
import six

import requests
from requests.exceptions import HTTPError

ENDPOINT = "https://huggingface.co"

class S3Obj:
    def __init__(
        self,
        filename,     # type: str
        LastModified, # type: str
        ETag,         # type: str
        Size,         # type: int
        **kwargs
    ):
        self.filename = filename
        self.LastModified = LastModified
        self.ETag = ETag
        self.Size = Size


class PresignedUrl:
    def __init__(
        self,
        write,  # type: str
        access, # type: str
        type,   # type: str
        **kwargs
    ):
        self.write = write
        self.access = access
        self.type = type # mime-type to send to S3.


class HfApi:
    def __init__(self, endpoint=None):
        self.endpoint = endpoint if endpoint is not None else ENDPOINT

    def login(
        self,
        username, # type: str
        password, # type: str
    ):
        # type: (...) -> str
        """
        Call HF API to sign in a user and get a token if credentials are valid.

        Outputs:
            token if credentials are valid

        Throws:
            requests.exceptions.HTTPError if credentials are invalid
        """
        path = "{}/api/login".format(self.endpoint)
        r = requests.post(path, json={"username": username, "password": password})
        r.raise_for_status()
        d = r.json()
        return d["token"]

    def whoami(
        self,
        token, # type: str
    ):
        # type: (...) -> str
        """
        Call HF API to know "whoami"
        """
        path = "{}/api/whoami".format(self.endpoint)
        r = requests.get(path, headers={"authorization": "Bearer {}".format(token)})
        r.raise_for_status()
        d = r.json()
        return d["user"]

    def logout(self, token):
        # type: (...) -> void
        """
        Call HF API to log out.
        """
        path = "{}/api/logout".format(self.endpoint)
        r = requests.post(path, headers={"authorization": "Bearer {}".format(token)})
        r.raise_for_status()

    def presign(self, token, filename):
        # type: (...) -> PresignedUrl
        """
        Call HF API to get a presigned url to upload `filename` to S3.
        """
        path = "{}/api/presign".format(self.endpoint)
        r = requests.post(
            path,
            headers={"authorization": "Bearer {}".format(token)},
            json={"filename": filename},
        )
        r.raise_for_status()
        d = r.json()
        return PresignedUrl(**d)

    def presign_and_upload(self, token, filename, filepath):
        # type: (...) -> str
        """
        Get a presigned url, then upload file to S3.

        Outputs:
            url: Read-only url for the stored file on S3.
        """
        urls = self.presign(token, filename=filename)
        # streaming upload:
        # https://2.python-requests.org/en/master/user/advanced/#streaming-uploads
        # 
        # Even though we presign with the correct content-type,
        # the client still has to specify it when uploading the file.
        with open(filepath, "rb") as f:
            r = requests.put(urls.write, data=f, headers={
                "content-type": urls.type,
            })
            r.raise_for_status()
        return urls.access

    def list_objs(self, token):
        # type: (...) -> List[S3Obj]
        """
        Call HF API to list all stored files for user.
        """
        path = "{}/api/listObjs".format(self.endpoint)
        r = requests.get(path, headers={"authorization": "Bearer {}".format(token)})
        r.raise_for_status()
        d = r.json()
        return [S3Obj(**x) for x in d]





class HfFolder:
    path_token = expanduser("~/.huggingface/token")

    @classmethod
    def save_token(cls, token):
        """
        Save token, creating folder as needed.
        """
        if six.PY3:
            os.makedirs(os.path.dirname(cls.path_token), exist_ok=True)
        else:
            # Python 2
            try:
                os.makedirs(os.path.dirname(cls.path_token))
            except OSError as e:
                if e.errno != os.errno.EEXIST:
                    raise e
                pass
        with open(cls.path_token, 'w+') as f:
            f.write(token)

    @classmethod
    def get_token(cls):
        """
        Get token or None if not existent.
        """
        try:
            with open(cls.path_token, 'r') as f:
                return f.read()
        except:
            # this is too wide. When Py2 is dead use:
            # `except FileNotFoundError:` instead
            return None

    @classmethod
    def delete_token(cls):
        """
        Delete token.
        Do not fail if token does not exist.
        """
        try:
            os.remove(cls.path_token)
        except:
            return
