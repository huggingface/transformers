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

Transformers funcționează ca framework-ul de definire a modelelor pentru tehnologii de ultimă generație în machine learning aplicate pe text, computer vision, audio, video și modele multimodale, atât pentru inferență, cât și pentru antrenare.

Acesta centralizează definirea modelelor astfel încât această definiție să fie agreată la nivelul întregului ecosistem. `transformers` este pivotul dintre framework-uri: dacă definirea unui model este suportată, acesta va fi compatibil cu majoritatea framework-urilor de antrenare (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), a motoarelor de inferență (vLLM, SGLang, TGI, ...) și a bibliotecilor de modelare adiacente (llama.cpp, mlx, ...) care utilizează definirea modelului din `transformers`.

Ne angajăm să ajutăm suportarea noilor modele de ultimă generație și să le democratizăm utilizarea prin oferirea unei definiri a modelului simplă, personalizabilă și eficientă.

Avem peste 1M de [checkpoint-uri de model](https://huggingface.co/models?library=transformers&sort=trending) Transformers pe [Hub-ul Hugging Face](https://huggingface.co/models) pe care le poți utiliza.

Explorează [Hub-ul](https://huggingface.co/) chiar azi pentru a găsi un model și folosește Transformers pentru a începe imediat.

Explorează [Timeline-ul Modelelor](./models_timeline) pentru a descoperi cele mai recente arhitecturi de tip text, vision, audio și model multimodal din Transformers.

## Funcții

Transformers oferă tot ce ai nevoie pentru antrenarea sau inferența cu modele pre-antrenate de ultimă generație. Printre funcțiile principale se numără:

- [Pipeline](./pipeline_tutorial): Clasă de inferență simplă și optimizată pentru multe task-uri de machine learning, precum generarea de text, segmentarea de imagini, recunoașterea vocală automată, răspunsuri la întrebări din documente etc.
- [Trainer](./trainer): Un trainer comprehensiv ce suportă funcții precum precizie mixtă, torch.compile și FlashAttention pentru antrenarea simplă și distribuită a modelelor PyTorch.
- [generate](./llm_tutorial): Generare de text rapidă cu modele lingvistice mari (LLMs) și modele lingvistice vizuale (VLMs), incluzând suport pentru streaming și mai multe strategii de decodare.

## Design

> [!TIP]
> Citește [Filozofia](./philosophy) noastră pentru a învăța mai multe despre filozofia de design a Transformers.

Transformers este construit pentru developeri, ingineri și cercetători în machine learning. Principiile principale de design pentru Transformers sunt:

1. Rapid și ușor de folosit: Fiecare model este implementat din doar 3 clase principale (configuration, model, și preprocessor) și poate fi utilizat rapid și ușor pentru antrenare sau inferență cu [`Pipeline`] sau [`Trainer`].
2. Modele pre-antrenate: Redu-ți amprenta de carbon, costul de calcul și timpul utilizând un model pre-antrenat în loc să antrenezi unul nou. Fiecare model pre-antrenat este reprodus cât mai fidel posibil de modelul original și oferă performanță de ultimă generație.

<div class="flex justify-center">
  <a target="_blank" href="https://huggingface.co/support">
      <img alt="HuggingFace Expert Acceleration Program" src="https://hf.co/datasets/huggingface/documentation-images/resolve/81d7d9201fd4ceb537fc4cebc22c29c37a2ed216/transformers/transformers-index.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
  </a>
</div>

## Învață

Dacă acum începi să utilizezi Transformers sau vrei să înveți mai multe despre modelele noastre, recomandăm să începi cu [Cursul despre LLMs](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt). Acest curs comprehensiv acoperă tot, de la cunoștințele de bază despre cum funcționează modelele până la aplicații practice ale acestora în diferite task-uri. Vei învăța workflow-ul complet, de la crearea unor seturi de date de calitate înaltă până la manipularea avansată a modelelor lingvistice mari și implementarea capabilităților de raționament. Acest curs conține atât exerciții teoretice, cât și exerciții practice pentru construirea unor cunoștințe de bază despre modelele pe bază de transformers pe parcursul învățării.
