<!--Copyright 2022 The HuggingFace Team and Microsoft. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-06-09 and added to Hugging Face Transformers on 2023-06-20 and contributed by [clefourrier](https://huggingface.co/clefourrier).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2. You can do so by running the following command: pip install -U transformers==4.40.2.

# Graphormer

[Graphormer](https://huggingface.co/papers/2106.05234) addresses the underperformance of Transformers in graph representation learning by introducing structural encoding methods. This enables Graphormer to achieve excellent results on graph-level prediction tasks, including the OGB Large-Scale Challenge. The model effectively encodes graph structure into the Transformer architecture, demonstrating that many GNN variants can be seen as special cases of Graphormer.

## Usage tips

- Graphormer works best on graphs with 100 or fewer nodes/edges. Large graphs cause memory issues. Reduce batch size, increase RAM, or decrease the `UNREACHABLE_NODE_DISTANCE` parameter in `algos_graphormer.pyx` to handle larger graphs. Expect difficulty scaling beyond 700 nodes/edges.
- Graphormer uses a special collator during training instead of a tokenizer.

## GraphormerConfig

[[autodoc]] GraphormerConfig

## GraphormerModel

[[autodoc]] GraphormerModel
    - forward

## GraphormerForGraphClassification

[[autodoc]] GraphormerForGraphClassification
    - forward

