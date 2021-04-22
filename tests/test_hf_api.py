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


import os
import shutil
import subprocess
import time
import unittest

from requests.exceptions import HTTPError
from transformers.hf_api import HfApi, HfFolder, ModelInfo, RepoObj
from transformers.testing_utils import ENDPOINT_STAGING, PASS, USER, is_staging_test, require_git_lfs


ENDPOINT_STAGING_BASIC_AUTH = f"https://{USER}:{PASS}@moon-staging.huggingface.co"

REPO_NAME = f"my-model-{int(time.time())}"
REPO_NAME_LARGE_FILE = f"my-model-largefiles-{int(time.time())}"
WORKING_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/working_repo")
LARGE_FILE_14MB = "https://cdn-media.huggingface.co/lfs-largefiles/progit.epub"
LARGE_FILE_18MB = "https://cdn-media.huggingface.co/lfs-largefiles/progit.pdf"


class HfApiCommonTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)


class HfApiLoginTest(HfApiCommonTest):
    def test_login_invalid(self):
        with self.assertRaises(HTTPError):
            self._api.login(username=USER, password="fake")

    def test_login_valid(self):
        token = self._api.login(username=USER, password=PASS)
        self.assertIsInstance(token, str)


class HfApiEndpointsTest(HfApiCommonTest):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)

    def test_whoami(self):
        user, orgs = self._api.whoami(token=self._token)
        self.assertEqual(user, USER)
        self.assertIsInstance(orgs, list)

    def test_list_repos_objs(self):
        objs = self._api.list_repos_objs(token=self._token)
        self.assertIsInstance(objs, list)
        if len(objs) > 0:
            o = objs[-1]
            self.assertIsInstance(o, RepoObj)

    def test_create_and_delete_repo(self):
        self._api.create_repo(token=self._token, name=REPO_NAME)
        self._api.delete_repo(token=self._token, name=REPO_NAME)


class HfApiPublicTest(unittest.TestCase):
    def test_staging_model_list(self):
        _api = HfApi(endpoint=ENDPOINT_STAGING)
        _ = _api.model_list()

    def test_model_list(self):
        _api = HfApi()
        models = _api.model_list()
        self.assertGreater(len(models), 100)
        self.assertIsInstance(models[0], ModelInfo)


class HfFolderTest(unittest.TestCase):
    def test_token_workflow(self):
        """
        Test the whole token save/get/delete workflow,
        with the desired behavior with respect to non-existent tokens.
        """
        token = f"token-{int(time.time())}"
        HfFolder.save_token(token)
        self.assertEqual(HfFolder.get_token(), token)
        HfFolder.delete_token()
        HfFolder.delete_token()
        # ^^ not an error, we test that the
        # second call does not fail.
        self.assertEqual(HfFolder.get_token(), None)


@require_git_lfs
@is_staging_test
class HfLargefilesTest(HfApiCommonTest):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)

    def setUp(self):
        try:
            shutil.rmtree(WORKING_REPO_DIR)
        except FileNotFoundError:
            pass

    def tearDown(self):
        self._api.delete_repo(token=self._token, name=REPO_NAME_LARGE_FILE)

    def setup_local_clone(self, REMOTE_URL):
        REMOTE_URL_AUTH = REMOTE_URL.replace(ENDPOINT_STAGING, ENDPOINT_STAGING_BASIC_AUTH)
        subprocess.run(["git", "clone", REMOTE_URL_AUTH, WORKING_REPO_DIR], check=True, capture_output=True)
        subprocess.run(["git", "lfs", "track", "*.pdf"], check=True, cwd=WORKING_REPO_DIR)
        subprocess.run(["git", "lfs", "track", "*.epub"], check=True, cwd=WORKING_REPO_DIR)

    def test_end_to_end_thresh_6M(self):
        REMOTE_URL = self._api.create_repo(
            token=self._token, name=REPO_NAME_LARGE_FILE, lfsmultipartthresh=6 * 10 ** 6
        )
        self.setup_local_clone(REMOTE_URL)

        subprocess.run(["wget", LARGE_FILE_18MB], check=True, capture_output=True, cwd=WORKING_REPO_DIR)
        subprocess.run(["git", "add", "*"], check=True, cwd=WORKING_REPO_DIR)
        subprocess.run(["git", "commit", "-m", "commit message"], check=True, cwd=WORKING_REPO_DIR)

        # This will fail as we haven't set up our custom transfer agent yet.
        failed_process = subprocess.run(["git", "push"], capture_output=True, cwd=WORKING_REPO_DIR)
        self.assertEqual(failed_process.returncode, 1)
        self.assertIn("transformers-cli lfs-enable-largefiles", failed_process.stderr.decode())
        # ^ Instructions on how to fix this are included in the error message.

        subprocess.run(["transformers-cli", "lfs-enable-largefiles", WORKING_REPO_DIR], check=True)

        start_time = time.time()
        subprocess.run(["git", "push"], check=True, cwd=WORKING_REPO_DIR)
        print("took", time.time() - start_time)

        # To be 100% sure, let's download the resolved file
        pdf_url = f"{REMOTE_URL}/resolve/main/progit.pdf"
        DEST_FILENAME = "uploaded.pdf"
        subprocess.run(["wget", pdf_url, "-O", DEST_FILENAME], check=True, capture_output=True, cwd=WORKING_REPO_DIR)
        dest_filesize = os.stat(os.path.join(WORKING_REPO_DIR, DEST_FILENAME)).st_size
        self.assertEqual(dest_filesize, 18685041)

    def test_end_to_end_thresh_16M(self):
        # Here we'll push one multipart and one non-multipart file in the same commit, and see what happens
        REMOTE_URL = self._api.create_repo(
            token=self._token, name=REPO_NAME_LARGE_FILE, lfsmultipartthresh=16 * 10 ** 6
        )
        self.setup_local_clone(REMOTE_URL)

        subprocess.run(["wget", LARGE_FILE_18MB], check=True, capture_output=True, cwd=WORKING_REPO_DIR)
        subprocess.run(["wget", LARGE_FILE_14MB], check=True, capture_output=True, cwd=WORKING_REPO_DIR)
        subprocess.run(["git", "add", "*"], check=True, cwd=WORKING_REPO_DIR)
        subprocess.run(["git", "commit", "-m", "both files in same commit"], check=True, cwd=WORKING_REPO_DIR)

        subprocess.run(["transformers-cli", "lfs-enable-largefiles", WORKING_REPO_DIR], check=True)

        start_time = time.time()
        subprocess.run(["git", "push"], check=True, cwd=WORKING_REPO_DIR)
        print("took", time.time() - start_time)
