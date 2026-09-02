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

# Aplicații de machine learning

[Gradio](https://www.gradio.app/), o bibliotecă rapidă și ușoară pentru construirea și partajarea de aplicații de machine learning, este integrată cu [`Pipeline`] pentru a crea rapid o interfață simplă pentru inferență.

Înainte de a începe, asigură-te că Gradio este instalată.

```py
!pip install gradio
```

Creează un pipeline pentru task-ul tău, iar apoi transmite-l funcției [Interface.from_pipeline](https://www.gradio.app/docs/gradio/interface#interface-from_pipeline) din Gradio pentru a crea interfața. Gradio determină automat componentele de input și output potrivite pentru un [`Pipeline`].

Adaugă [launch](https://www.gradio.app/main/docs/gradio/blocks#blocks-launch) pentru a crea un web server și a porni aplicația.

```py
from transformers import pipeline
import gradio as gr

pipeline = pipeline("image-classification", model="google/vit-base-patch16-224")
gr.Interface.from_pipeline(pipeline).launch()
```

Aplicația web rulează implicit pe un server local. Pentru a partaja aplicația cu alți utilizatori, setează `share=True` în [launch](https://www.gradio.app/main/docs/gradio/blocks#blocks-launch) pentru a genera un link public temporar. Pentru o soluție mai permanentă, găzduiește aplicația pe Hugging Face [Spaces](https://hf.co/spaces).

```py
gr.Interface.from_pipeline(pipeline).launch(share=True)
```

Space-ul de mai jos este creat cu codul de mai sus și găzduit pe Spaces.

<iframe
 src="https://stevhliu-gradio-pipeline-demo.hf.space"
 frameborder="0"
 width="850"
 height="850"
></iframe>
