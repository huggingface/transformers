#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from .tools import Tool


class DuckDuckGoSearchTool(Tool):
    name = "web_search"
    description = """Perform a web search based on your query (think a Google search) then returns the top search results as a list of dict elements.
    Each result has keys 'title', 'href' and 'body'."""
    inputs = {"query": {"type": "text", "description": "The search query to perform."}}
    output_type = "any"

    def forward(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "You must install package `duckduckgo_search`: for instance run `pip install duckduckgo-search`."
            )
        results = DDGS().text(query, max_results=7)
        return results
