<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Rendimiento y Escalabilidad

Entrenar modelos grandes de transformadores y desplegarlos en producción presenta varios desafíos. Durante el entrenamiento, el modelo puede requerir más memoria de GPU de la disponible o mostrar una velocidad de entrenamiento lenta. En la fase de implementación, el modelo puede tener dificultades para manejar el rendimiento necesario en un entorno de producción.

Esta documentación tiene como objetivo ayudarte a superar estos desafíos y encontrar la configuración óptima para tu caso de uso. Las guías están divididas en secciones de entrenamiento e inferencia, ya que cada una presenta diferentes desafíos y soluciones. Dentro de cada sección, encontrarás guías separadas para diferentes configuraciones de hardware, como GPU única vs. multi-GPU para el entrenamiento o CPU vs. GPU para la inferencia.

Utiliza este documento como punto de partida para navegar hacia los métodos que se ajusten a tu escenario.

## Entrenamiento

Entrenar modelos grandes de transformadores de manera eficiente requiere un acelerador como una GPU o TPU. El caso más común es cuando tienes una GPU única. Los métodos que puedes aplicar para mejorar la eficiencia de entrenamiento en una GPU única también se aplican a otras configuraciones, como múltiples GPU. Sin embargo, también existen técnicas específicas para entrenamiento con múltiples GPU o CPU, las cuales cubrimos en secciones separadas.

* [Métodos y herramientas para un entrenamiento eficiente en una sola GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one): comienza aquí para aprender enfoques comunes que pueden ayudar a optimizar la utilización de memoria de la GPU, acelerar el entrenamiento o ambas cosas.
* [Sección de entrenamiento con varias GPU](https://huggingface.co/docs/transformers/perf_train_gpu_many): explora esta sección para conocer métodos de optimización adicionales que se aplican a configuraciones con varias GPU, como paralelismo de datos, tensores y canalizaciones.
* [Sección de entrenamiento en CPU](https://huggingface.co/docs/transformers/perf_train_cpu): aprende sobre entrenamiento de precisión mixta en CPU.
* [Entrenamiento eficiente en múltiples CPUs](https://huggingface.co/docs/transformers/perf_train_cpu_many): aprende sobre el entrenamiento distribuido en CPU.
* [Entrenamiento en TPU con TensorFlow](https://huggingface.co/docs/transformers/perf_train_tpu_tf): si eres nuevo en TPUs, consulta esta sección para obtener una introducción basada en opiniones sobre el entrenamiento en TPUs y el uso de XLA.
* [Hardware personalizado para el entrenamiento](https://huggingface.co/docs/transformers/perf_hardware): encuentra consejos y trucos al construir tu propia plataforma de aprendizaje profundo.
* [Búsqueda de hiperparámetros utilizando la API del Entrenador](https://huggingface.co/docs/transformers/hpo_train)

## Inferencia

Realizar inferencias eficientes con modelos grandes en un entorno de producción puede ser tan desafiante como entrenarlos. En las siguientes secciones, describimos los pasos para ejecutar inferencias en CPU y configuraciones con GPU única/múltiple.

* [Inferencia en una sola CPU](https://huggingface.co/docs/transformers/perf_infer_cpu)
* [Inferencia en una sola GPU](https://huggingface.co/docs/transformers/perf_infer_gpu_one)
* [Inferencia con múltiples GPU](https://huggingface.co/docs/transformers/perf_infer_gpu_one)
* [Integración de XLA para modelos de TensorFlow](https://huggingface.co/docs/transformers/tf_xla)

## Entrenamiento e Inferencia

Aquí encontrarás técnicas, consejos y trucos que aplican tanto si estás entrenando un modelo como si estás ejecutando inferencias con él.

* [Instanciar un modelo grande](https://huggingface.co/docs/transformers/big_models)
* [Solución de problemas de rendimiento](https://huggingface.co/docs/transformers/debugging)

## Contribuir

Este documento está lejos de estar completo y aún se deben agregar muchas cosas, así que si tienes adiciones o correcciones que hacer, no dudes en abrir un PR. Si no estás seguro, inicia un Issue y podemos discutir los detalles allí.

Cuando hagas contribuciones que indiquen que A es mejor que B, intenta incluir un benchmark reproducible y/o un enlace a la fuente de esa información (a menos que provenga directamente de ti).
