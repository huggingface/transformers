#!/usr/bin/env python
# coding=utf-8
# Copyright 2025
#
# Utility script to retrieve a CircleCI workflow ID for a given branch and commit SHA.
#
# Usage:
#   python scripts/find_circleci_workflow.py --branch main --sha <commit_sha>
#
# Environment:
#   CIRCLECI_TOKEN must be set with a token that has permission to query the CircleCI API.

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import requests


CIRCLE_API = "https://circleci.com/api/v2"
PROJECT_SLUG = "gh/huggingface/transformers"


def _get_circle_token(token: Optional[str]) -> str:
    token = token or os.environ.get("CIRCLECI_TOKEN") or os.environ.get("CCI_TOKEN") or os.environ.get("CIRCLE_TOKEN")
    if not token:
        raise SystemExit("CIRCLECI_TOKEN (or CCI_TOKEN / CIRCLE_TOKEN) must be provided.")
    return token


def _request(url: str, token: str, params: Optional[dict] = None) -> dict:
    response = requests.get(
        url,
        params=params,
        headers={"Circle-Token": token},
    )
    response.raise_for_status()
    return response.json()


def _find_pipeline_id(branch: str, revision: str, token: str) -> str:
    url = f"{CIRCLE_API}/project/{PROJECT_SLUG}/pipeline"
    params = {"branch": branch}
    pages_checked = 0
    while True:
        payload = _request(url, token, params=params)
        for pipeline in payload.get("items", []):
            vcs = pipeline.get("vcs") or {}
            if vcs.get("revision") == revision:
                return pipeline["id"]
        next_token = payload.get("next_page_token")
        if not next_token or pages_checked > 10:
            break
        params["page-token"] = next_token
        pages_checked += 1
    raise SystemExit(f"Unable to find CircleCI pipeline for branch {branch} and revision {revision}.")


def _workflow_has_collection_job(workflow_id: str, token: str) -> bool:
    jobs = _request(f"{CIRCLE_API}/workflow/{workflow_id}/job", token)
    return any(job.get("name") == "collection_job" for job in jobs.get("items", []))


def _find_workflow_with_collection_job(pipeline_id: str, token: str) -> str:
    payload = _request(f"{CIRCLE_API}/pipeline/{pipeline_id}/workflow", token)
    workflows = payload.get("items", [])
    for workflow in workflows:
        workflow_id = workflow["id"]
        if _workflow_has_collection_job(workflow_id, token):
            return workflow_id
    if workflows:
        return workflows[0]["id"]
    raise SystemExit(f"No workflows found for pipeline {pipeline_id}.")


def main():
    parser = argparse.ArgumentParser(description="Find CircleCI workflow id for a commit.")
    parser.add_argument("--branch", required=True, help="Branch name for the CircleCI pipeline.")
    parser.add_argument("--sha", required=True, help="Commit SHA to match.")
    parser.add_argument("--token", default=None, help="CircleCI API token.")
    args = parser.parse_args()

    token = _get_circle_token(args.token)
    pipeline_id = _find_pipeline_id(args.branch, args.sha, token)
    workflow_id = _find_workflow_with_collection_job(pipeline_id, token)
    print(workflow_id)


if __name__ == "__main__":
    main()
