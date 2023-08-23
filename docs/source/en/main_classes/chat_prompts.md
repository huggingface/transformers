<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Chat Prompts

An increasingly common use case for LLMs is *chat*. In a chat context, rather than continuing a single string
of text (as is the case with a standard language model), the model instead continues a conversation that consists
of one or more *messages*, each of which includes a *role* as well as message text.

All LLMs, including models fine-tuned for chat, operate on linear sequences of tokens, and do not intrinsically
have special handling for 'roles'. This means that role information is usually injected by adding control tokens
between messages, to indicate both the message boundary and the relevant roles.

Unfortunately, there isn't (yet!) a standard for which tokens to use, and so different models have been trained
with wildly different formatting and control tokens for chat. This is the problem that the **PromptConfig** class
aims to solve. It allows information about chat formatting and default prompts to be saved and loaded with the model,
ensuring that the model functions correctly out-of-the-box for inference, as well as allowing other users to fine-tune
the model further without accidentally changing the chat format and silently hurting performance.
