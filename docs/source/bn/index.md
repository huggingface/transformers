<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Transformers

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

ইনফারেন্স এবং ট্রেনিং—উভয় ক্ষেত্রেই টেক্সট, কম্পিউটার ভিশন, অডিও, ভিডিও এবং মাল্টিমোডাল স্টেট-অফ-দ্য-আর্ট মেশিন লার্নিং মডেলগুলোর মডেল-ডেফিনিশন ফ্রেমওয়ার্ক হিসেবে কাজ করে Transformers।

এটি মডেল ডেফিনিশনকে একটা কেন্দ্রীয় জায়গায় নিয়ে আসে, যাতে পুরো ইকোসিস্টেম জুড়ে সবাই একই স্ট্যান্ডার্ড মেনে চলতে পারে। বিভিন্ন ফ্রেমওয়ার্কের মধ্যে একটা সেতু বা পিভট হিসেবে কাজ করে `transformers`: কোনো মডেল ডেফিনিশন এখানে সাপোর্ট করলে, সেটি নিজে থেকেই বেশিরভাগ ট্রেনিং ফ্রেমওয়ার্ক (যেমন- Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning...), ইনফারেন্স ইঞ্জিন (যেমন- vLLM, SGLang, TGI...) এবং এর সাথে সম্পৃক্ত মডেলিং লাইব্রেরিগুলোর (যেমন- llama.cpp, mlx...) সাথে কাজ করবে, কারণ এরা সবাই `transformers` থেকে মডেল ডেফিনিশনটি ব্যবহার করে।

আমাদের লক্ষ্য হলো নতুন সব স্টেট-অফ-দ্য-আর্ট মডেলগুলোকে সাপোর্ট করা এবং সেগুলোর মডেল ডেফিনিশনকে সহজ, কাস্টমাইজযোগ্য ও এফিশিয়েন্ট রেখে সবার ব্যবহারের জন্য সহজলভ্য করে দেওয়া।

[Hugging Face Hub](https://huggingface.com/models)-এ ১ মিলিয়নেরও (1M+) বেশি Transformers [মডেল চেকময়েন্ট](https://huggingface.co/models?library=transformers&sort=trending) রয়েছে যেগুলো অনায়াসেই ব্যবহার করা সম্ভব।

আজই ঘুরে আসতে পারো [Hub](https://huggingface.com/) থেকে, নিজের পছন্দের মডেলটি বেছে নাও আর Transformers ব্যবহার করে চটজলদি কাজ শুরু করে দাও।

Transformers-এর লেটেস্ট টেক্সট, ভিশন, অডিও এবং মাল্টিমোডাল মডেল切换 আর্কিটেকচারগুলো সম্পর্কে জানতে চোখ বুলিয়ে নিতে পারো [Models Timeline](./models_timeline)-এ।

## ফিচারসমূহ

স্টেট-অফ-দ্য-আর্ট প্রিট্রেইন্ড মডেল নিয়ে ইনফারেন্স বা ট্রেনিংয়ের জন্য যা যা প্রয়োজন, তার সবকিছুই আছে Transformers-এ। এর প্রধান কিছু ফিচারের মধ্যে রয়েছে:

- [Pipeline](./pipeline_tutorial): টেক্সট জেনারেশন, 이미지 সেগমেন্টেশন, অটোমেটিক স্পিচ রিকগনিশন, ডকুমেন্ট কোশ্চেন অ্যানসারিংয়ের মতো দারুণ সব মেশিন লার্নিং টাস্কের জন্য একটি সহজ এবং অপ্টিমাইজড ইনফারেন্স ক্লাস।
- [Trainer](./trainer): একটি কমপ্রিহেনসিভ ট্রেইনার যা PyTorch মডেলগুলোর ট্রেনিং এবং ডিস্ট্রিবিউটেড ট্রেনিংয়ের জন্য মিক্সড প্রিসিশন, torch.compile এবং FlashAttention-এর মতো ফিচারগুলো সাপোর্ট করে।
- [generate](./llm_tutorial): লার্জ ল্যাঙ্গুয়েজ মডেল (LLM) এবং ভিশন ল্যাঙ্গুয়েজ মডেল (VLM) দিয়ে দ্রুত টেক্সট জেনারেট করার সুবিধা, যার মধ্যে স্ট্রিমিং এবং একাধিক ডিকোডিং স্ট্র্যাটেজির সাপোর্টও অন্তর্ভুক্ত।

## ডিজাইন

> [!TIP]
> Transformers-এর ডিজাইনের পেছনের মূল ভাবনাগুলো জানতে আমাদের [Philosophy](./philosophy) পেজটি পড়তে পারো।

ডেভেলপার, মেশিন লার্নিং ইঞ্জিনিয়ার এবং গবেষকদের কথা মাথায় রেখেই ডিজাইন করা হয়েছে Transformers। এর প্রধান ডিজাইনের মূলনীতিগুলো হলো:

১. দ্রুত ও সহজে ব্যবহারযোগ্য: প্রতিটি মডেল তৈরি করা হয়েছে মাত্র তিনটি প্রধান ক্লাস (configuration, model এবং preprocessor) দিয়ে, আর [`Pipeline`] বা [`Trainer`] ব্যবহার করে এগুলোকে খুব দ্রুত ইনফারেন্স বা ট্রেনিংয়ের কাজে লাগিয়ে দেওয়া যায়।
২. প্রিট্রেইন্ড মডেলসমূহ: একদম স্ক্র্যাচ থেকে নতুন কোনো মডেল ট্রেইন না করে একটি প্রিট্রেইন্ড মডেল ব্যবহার করলে কার্বন ফুটপ্রিন্ট যেমন কমে, তেমনই বাঁচে কম্পিউটেশন খরচ আর সময়। প্রতিটি প্রিট্রেইন্ড মডেলকে মূল মডেলের যতটা সম্ভব কাছাকাছি রেখে তৈরি করা হয়েছে এবং এগুলো চমৎকার পারফরম্যান্স দেয়।

<div class="flex justify-center">
  <a target="_blank" href="https://huggingface.co/support">
      <img alt="HuggingFace Expert Acceleration Program" src="https://hf.co/datasets/huggingface/documentation-images/resolve/81d7d9201fd4ceb537fc4cebc22c29c37a2ed216/transformers/transformers-index.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
  </a>
</div>

## শিখুন

Transformers-এ যদি একদম নতুন হয়ে থাকো কিংবা ট্রান্সফর্মার মডেল সম্পর্কে আরও বিস্তারিত জানতে চাও, তবে [LLM course](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt) দিয়ে শুরু করার পরামর্শ দেব। ট্রান্সফর্মার মডেল কীভাবে কাজ করে তার একদম বেসিক থেকে শুরু করে বিভিন্ন টাস্কে এর প্র্যাক্টিক্যাল ব্যবহার—সবকিছুই দারুণভাবে কভার করা হয়েছে এই কোর্সটিতে। হাই-কোয়ালিটি ডাটাফিড তৈরি করা থেকে শুরু করে লার্জ ল্যাঙ্গুয়েজ মডেল ফাইন-টিউন করা এবং সেগুলোতে রিজনিং ক্যাপাবিলিটি যুক্ত করার পুরো ওয়ার্কফ্লোটি শেখা যাবে এখানে। শেখার পাশাপাশি নিজের ভিত্তি মজবুত করার জন্য কোর্সটিতে থিওরেটিক্যাল আলোচনার সাথে সাথে হ্যান্ডস-অন এক্সারসাইজও রাখা হয়েছে।
