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

# ¿Cómo los 🤗 Transformers resuelven tareas?

En [Qué pueden hacer los 🤗 Transformers](task_summary), aprendiste sobre el procesamiento de lenguaje natural (NLP), tareas de voz y audio, visión por computadora y algunas aplicaciones importantes de ellas. Esta página se centrará en cómo los modelos resuelven estas tareas y explicará lo que está sucediendo debajo de la superficie. Hay muchas maneras de resolver una tarea dada, y diferentes modelos pueden implementar ciertas técnicas o incluso abordar la tarea desde un ángulo nuevo, pero para los modelos Transformer, la idea general es la misma. Debido a su arquitectura flexible, la mayoría de los modelos son una variante de una estructura de codificador, descodificador o codificador-descodificador. Además de los modelos Transformer, nuestra biblioteca también tiene varias redes neuronales convolucionales (CNNs) modernas, que todavía se utilizan hoy en día para tareas de visión por computadora. También explicaremos cómo funciona una CNN moderna.

Para explicar cómo se resuelven las tareas, caminaremos a través de lo que sucede dentro del modelo para generar predicciones útiles.

- [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) para clasificación de audio y reconocimiento automático de habla (ASR)
- [Transformador de Visión (ViT)](https://huggingface.co/docs/transformers/model_doc/vit) y [ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext) para clasificación de imágenes
- [DETR](https://huggingface.co/docs/transformers/model_doc/detr) para detección de objetos
- [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) para segmentación de imagen
- [GLPN](https://huggingface.co/docs/transformers/model_doc/glpn) para estimación de profundidad
- [BERT](https://huggingface.co/docs/transformers/model_doc/bert) para tareas de NLP como clasificación de texto, clasificación de tokens y preguntas y respuestas que utilizan un codificador
- [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) para tareas de NLP como generación de texto que utilizan un descodificador
- [BART](https://huggingface.co/docs/transformers/model_doc/bart) para tareas de NLP como resumen y traducción que utilizan un codificador-descodificador

<Tip>

Antes de continuar, es bueno tener un conocimiento básico de la arquitectura original del Transformer. Saber cómo funcionan los codificadores, decodificadores y la atención te ayudará a entender cómo funcionan los diferentes modelos de Transformer. Si estás empezando o necesitas repasar, ¡echa un vistazo a nuestro [curso](https://huggingface.co/course/chapter1/4?fw=pt) para obtener más información!

</Tip>

## Habla y audio

[Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) es un modelo auto-supervisado preentrenado en datos de habla no etiquetados y ajustado en datos etiquetados para clasificación de audio y reconocimiento automático de voz. 

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/wav2vec2_architecture.png"/>
</div>

Este modelo tiene cuatro componentes principales:

1. Un *codificador de características* toma la forma de onda de audio cruda, la normaliza a media cero y varianza unitaria, y la convierte en una secuencia de vectores de características, cada uno de 20 ms de duración.

2. Las formas de onda son continuas por naturaleza, por lo que no se pueden dividir en unidades separadas como una secuencia de texto se puede dividir en palabras. Por eso, los vectores de características se pasan a un *módulo de cuantificación*, que tiene como objetivo aprender unidades de habla discretas. La unidad de habla se elige de una colección de palabras de código, conocidas como *codebook* (puedes pensar en esto como el vocabulario). Del codebook, se elige el vector o unidad de habla que mejor representa la entrada de audio continua y se envía a través del modelo.

3. Alrededor de la mitad de los vectores de características se enmascaran aleatoriamente, y el vector de características enmascarado se alimenta a una *red de contexto*, que es un codificador Transformer que también agrega incrustaciones posicionales relativas.

4. El objetivo del preentrenamiento de la red de contexto es una *tarea contrastiva*. El modelo tiene que predecir la verdadera representación de habla cuantizada de la predicción enmascarada a partir de un conjunto de falsas, lo que anima al modelo a encontrar el vector de contexto y la unidad de habla cuantizada más similares (la etiqueta objetivo).

¡Ahora que wav2vec2 está preentrenado, puedes ajustarlo con tus datos para clasificación de audio o reconocimiento automático de voz!

### Clasificación de audio

Para usar el modelo preentrenado para la clasificación de audio, añade una capa de clasificación de secuencia encima del modelo base de Wav2Vec2. La capa de clasificación es una capa lineal que acepta los estados ocultos del codificador. Los estados ocultos representan las características aprendidas de cada fotograma de audio, que pueden tener longitudes variables. Para crear un vector de longitud fija, primero se agrupan los estados ocultos y luego se transforman en logits sobre las etiquetas de clase. La pérdida de entropía cruzada se calcula entre los logits y el objetivo para encontrar la clase más probable.

¿Listo para probar la clasificación de audio? ¡Consulta nuestra guía completa de [clasificación de audio](https://huggingface.co/docs/transformers/tasks/audio_classification) para aprender cómo ajustar Wav2Vec2 y usarlo para inferencia!

### Reconocimiento automático de voz

Para usar el modelo preentrenado para el reconocimiento automático de voz, añade una capa de modelado del lenguaje encima del modelo base de Wav2Vec2 para [CTC (clasificación temporal conexista)](glossary#connectionist-temporal-classification-ctc). La capa de modelado del lenguaje es una capa lineal que acepta los estados ocultos del codificador y los transforma en logits. Cada logit representa una clase de token (el número de tokens proviene del vocabulario de la tarea). La pérdida de CTC se calcula entre los logits y los objetivos para encontrar la secuencia de tokens más probable, que luego se decodifican en una transcripción.

¿Listo para probar el reconocimiento automático de voz? ¡Consulta nuestra guía completa de [reconocimiento automático de voz](tasks/asr) para aprender cómo ajustar Wav2Vec2 y usarlo para inferencia!

## Visión por computadora

Hay dos formas de abordar las tareas de visión por computadora:

1. Dividir una imagen en una secuencia de parches y procesarlos en paralelo con un Transformer.
2. Utilizar una CNN moderna, como [ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext), que se basa en capas convolucionales pero adopta diseños de redes modernas.

<Tip>

Un tercer enfoque combina Transformers con convoluciones (por ejemplo, [Convolutional Vision Transformer](https://huggingface.co/docs/transformers/model_doc/cvt) o [LeViT](https://huggingface.co/docs/transformers/model_doc/levit)). No discutiremos estos porque simplemente combinan los dos enfoques que examinamos aquí.

</Tip>

ViT y ConvNeXT se utilizan comúnmente para la clasificación de imágenes, pero para otras tareas de visión como la detección de objetos, la segmentación y la estimación de profundidad, veremos DETR, Mask2Former y GLPN, respectivamente; estos modelos son más adecuados para esas tareas.

### Clasificación de imágenes

ViT y ConvNeXT pueden usarse ambos para la clasificación de imágenes; la diferencia principal es que ViT utiliza un mecanismo de atención mientras que ConvNeXT utiliza convoluciones.

#### Transformer

[ViT](https://huggingface.co/docs/transformers/model_doc/vit) reemplaza completamente las convoluciones con una arquitectura de Transformer pura. Si estás familiarizado con el Transformer original, entonces ya estás en el camino para entender ViT.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg"/>
</div>

El cambio principal que introdujo ViT fue en cómo se alimentan las imágenes a un Transformer:

1. Una imagen se divide en parches cuadrados no superpuestos, cada uno de los cuales se convierte en un vector o *incrustación de parche*(patch embedding). Las incrustaciones de parche se generan a partir de una capa convolucional 2D que crea las dimensiones de entrada adecuadas (que para un Transformer base son 768 valores para cada incrustación de parche). Si tuvieras una imagen de 224x224 píxeles, podrías dividirla en 196 parches de imagen de 16x16. Al igual que el texto se tokeniza en palabras, una imagen se "tokeniza" en una secuencia de parches.

2. Se agrega una *incrustación aprendida* - un token especial `[CLS]` - al principio de las incrustaciones del parche, al igual que en BERT. El estado oculto final del token `[CLS]` se utiliza como la entrada para la cabecera de clasificación adjunta; otras salidas se ignoran. Este token ayuda al modelo a aprender cómo codificar una representación de la imagen.

3. Lo último que se agrega a las incrustaciones de parche e incrustaciones aprendidas son las *incrustaciones de posición* porque el modelo no sabe cómo están ordenados los parches de imagen. Las incrustaciones de posición también son aprendibles y tienen el mismo tamaño que las incrustaciones de parche. Finalmente, todas las incrustaciones se pasan al codificador Transformer.

4. La salida, específicamente solo la salida con el token `[CLS]`, se pasa a una cabecera de perceptrón multicapa (MLP). El objetivo del preentrenamiento de ViT es simplemente la clasificación. Al igual que otras cabeceras de clasificación, la cabecera de MLP convierte la salida en logits sobre las etiquetas de clase y calcula la pérdida de entropía cruzada para encontrar la clase más probable.

¿Listo para probar la clasificación de imágenes? ¡Consulta nuestra guía completa de [clasificación de imágenes](tasks/image_classification) para aprender cómo ajustar ViT y usarlo para inferencia!

#### CNN

<Tip>

Esta sección explica brevemente las convoluciones, pero sería útil tener un entendimiento previo de cómo cambian la forma y el tamaño de una imagen. Si no estás familiarizado con las convoluciones, ¡echa un vistazo al [capítulo de Redes Neuronales Convolucionales](https://github.com/fastai/fastbook/blob/master/13_convolutions.ipynb) del libro fastai!

</Tip>

[ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext) es una arquitectura de CNN que adopta diseños de redes nuevas y modernas para mejorar el rendimiento. Sin embargo, las convoluciones siguen siendo el núcleo del modelo. Desde una perspectiva de alto nivel, una [convolución](glossary#convolution) es una operación donde una matriz más pequeña (*kernel*) se multiplica por una pequeña ventana de píxeles de la imagen. Esta calcula algunas características de ella, como una textura particular o la curvatura de una línea. Luego, se desliza hacia la siguiente ventana de píxeles; la distancia que recorre la convolución se conoce como el *stride*. 

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convolution.gif"/>
</div>

<small>Una convolución básica sin relleno ni paso, tomada de <a href="https://arxiv.org/abs/1603.07285">Una guía para la aritmética de convoluciones para el aprendizaje profundo.</a></small>

Puedes alimentar esta salida a otra capa convolucional, y con cada capa sucesiva, la red aprende cosas más complejas y abstractas como perros calientes o cohetes. Entre capas convolucionales, es común añadir una capa de agrupación para reducir la dimensionalidad y hacer que el modelo sea más robusto a las variaciones de la posición de una característica.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnext_architecture.png"/>
</div>

ConvNeXT moderniza una CNN de cinco maneras:

1. Cambia el número de bloques en cada etapa y "fragmenta" una imagen con un paso y tamaño de kernel más grandes. La ventana deslizante no superpuesta hace que esta estrategia de fragmentación sea similar a cómo ViT divide una imagen en parches.

2. Una capa de *cuello de botella* reduce el número de canales y luego lo restaura porque es más rápido hacer una convolución de 1x1, y se puede aumentar la profundidad. Un cuello de botella invertido hace lo contrario al expandir el número de canales y luego reducirlos, lo cual es más eficiente en memoria.

3. Reemplaza la típica capa convolucional de 3x3 en la capa de cuello de botella con una convolución *depthwise*, que aplica una convolución a cada canal de entrada por separado y luego los apila de nuevo al final. Esto ensancha el ancho de la red para mejorar el rendimiento.

4. ViT tiene un campo receptivo global, lo que significa que puede ver más de una imagen a la vez gracias a su mecanismo de atención. ConvNeXT intenta replicar este efecto aumentando el tamaño del kernel a 7x7.

5. ConvNeXT también hace varios cambios en el diseño de capas que imitan a los modelos Transformer. Hay menos capas de activación y normalización, la función de activación se cambia a GELU en lugar de ReLU, y utiliza LayerNorm en lugar de BatchNorm.

La salida de los bloques convolucionales se pasa a una cabecera de clasificación que convierte las salidas en logits y calcula la pérdida de entropía cruzada para encontrar la etiqueta más probable.

### Object detection

[DETR](https://huggingface.co/docs/transformers/model_doc/detr), *DEtection TRansformer*, es un modelo de detección de objetos de un extremo a otro que combina una CNN con un codificador-decodificador Transformer.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/detr_architecture.png"/>
</div>

1. Una CNN preentrenada *backbone* toma una imagen, representada por sus valores de píxeles, y crea un mapa de características de baja resolución de la misma. A continuación, se aplica una convolución 1x1 al mapa de características para reducir la dimensionalidad y se crea un nuevo mapa de características con una representación de imagen de alto nivel. Dado que el Transformer es un modelo secuencial, el mapa de características se aplana en una secuencia de vectores de características que se combinan con incrustaciones posicionales.

2. Los vectores de características se pasan al codificador, que aprende las representaciones de imagen usando sus capas de atención. A continuación, los estados ocultos del codificador se combinan con *consultas de objeto* en el decodificador. Las consultas de objeto son incrustaciones aprendidas que se enfocan en las diferentes regiones de una imagen, y se actualizan a medida que avanzan a través de cada capa de atención. Los estados ocultos del decodificador se pasan a una red feedforward que predice las coordenadas del cuadro delimitador y la etiqueta de clase para cada consulta de objeto, o `no objeto` si no hay ninguno.

    DETR descodifica cada consulta de objeto en paralelo para producir *N* predicciones finales, donde *N* es el número de consultas. A diferencia de un modelo autoregresivo típico que predice un elemento a la vez, la detección de objetos es una tarea de predicción de conjuntos (`cuadro delimitador`, `etiqueta de clase`) que hace *N* predicciones en un solo paso.

3. DETR utiliza una **pérdida de coincidencia bipartita** durante el entrenamiento para comparar un número fijo de predicciones con un conjunto fijo de etiquetas de verdad básica. Si hay menos etiquetas de verdad básica en el conjunto de *N* etiquetas, entonces se rellenan con una clase `no objeto`. Esta función de pérdida fomenta que DETR encuentre una asignación uno a uno entre las predicciones y las etiquetas de verdad básica. Si los cuadros delimitadores o las etiquetas de clase no son correctos, se incurre en una pérdida. Del mismo modo, si DETR predice un objeto que no existe, se penaliza. Esto fomenta que DETR encuentre otros objetos en una imagen en lugar de centrarse en un objeto realmente prominente.

Se añade una cabecera de detección de objetos encima de DETR para encontrar la etiqueta de clase y las coordenadas del cuadro delimitador. Hay dos componentes en la cabecera de detección de objetos: una capa lineal para transformar los estados ocultos del decodificador en logits sobre las etiquetas de clase, y una MLP para predecir el cuadro delimitador.

¿Listo para probar la detección de objetos? ¡Consulta nuestra guía completa de [detección de objetos](https://huggingface.co/docs/transformers/tasks/object_detection) para aprender cómo ajustar DETR y usarlo para inferencia!

### Segmentación de imágenes

[Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) es una arquitectura universal para resolver todos los tipos de tareas de segmentación de imágenes. Los modelos de segmentación tradicionales suelen estar adaptados a una tarea particular de segmentación de imágenes, como la segmentación de instancias, semántica o panóptica. Mask2Former enmarca cada una de esas tareas como un problema de *clasificación de máscaras*. La clasificación de máscaras agrupa píxeles en *N* segmentos, y predice *N* máscaras y su etiqueta de clase correspondiente para una imagen dada. Explicaremos cómo funciona Mask2Former en esta sección, y luego podrás probar el ajuste fino de SegFormer al final.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/mask2former_architecture.png"/>
</div>

Hay tres componentes principales en Mask2Former:

1. Un [backbone Swin](https://huggingface.co/docs/transformers/model_doc/swin) acepta una imagen y crea un mapa de características de imagen de baja resolución a partir de 3 convoluciones consecutivas de 3x3.

2. El mapa de características se pasa a un *decodificador de píxeles* que aumenta gradualmente las características de baja resolución en incrustaciones de alta resolución por píxel. De hecho, el decodificador de píxeles genera características multiescala (contiene características de baja y alta resolución) con resoluciones de 1/32, 1/16 y 1/8 de la imagen original.

3. Cada uno de estos mapas de características de diferentes escalas se alimenta sucesivamente a una capa decodificadora Transformer a la vez para capturar objetos pequeños de las características de alta resolución. La clave de Mask2Former es el mecanismo de *atención enmascarada* en el decodificador. A diferencia de la atención cruzada que puede atender a toda la imagen, la atención enmascarada solo se centra en cierta área de la imagen. Esto es más rápido y conduce a un mejor rendimiento porque las características locales de una imagen son suficientes para que el modelo aprenda.

4. Al igual que [DETR](tasks_explained#object-detection), Mask2Former también utiliza consultas de objetos aprendidas y las combina con las características de la imagen del decodificador de píxeles para hacer una predicción de conjunto (`etiqueta de clase`, `predicción de máscara`). Los estados ocultos del decodificador se pasan a una capa lineal y se transforman en logits sobre las etiquetas de clase. Se calcula la pérdida de entropía cruzada entre los logits y la etiqueta de clase para encontrar la más probable.

    Las predicciones de máscara se generan combinando las incrustaciones de píxeles con los estados ocultos finales del decodificador. La pérdida de entropía cruzada sigmoidea y de la pérdida DICE se calcula entre los logits y la máscara de verdad básica para encontrar la máscara más probable.

¿Listo para probar la detección de objetos? ¡Consulta nuestra guía completa de [segmentación de imágenes](https://huggingface.co/docs/transformers/tasks/semantic_segmentation) para aprender cómo ajustar SegFormer y usarlo para inferencia!

### Estimación de profundidad

[GLPN](https://huggingface.co/docs/transformers/model_doc/glpn), *Global-Local Path Network*, es un Transformer para la estimación de profundidad que combina un codificador [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) con un decodificador ligero.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/glpn_architecture.jpg"/>
</div>

1. Al igual que ViT, una imagen se divide en una secuencia de parches, excepto que estos parches de imagen son más pequeños. Esto es mejor para tareas de predicción densa como la segmentación o la estimación de profundidad. Los parches de imagen se transforman en incrustaciones de parches (ver la sección de [clasificación de imágenes](#clasificación-de-imágenes) para más detalles sobre cómo se crean las incrustaciones de parches), que se alimentan al codificador.

2. El codificador acepta las incrustaciones de parches y las pasa a través de varios bloques codificadores. Cada bloque consiste en capas de atención y Mix-FFN. El propósito de este último es proporcionar información posicional. Al final de cada bloque codificador hay una capa de *fusión de parches* para crear representaciones jerárquicas. Las características de cada grupo de parches vecinos se concatenan, y se aplica una capa lineal a las características concatenadas para reducir el número de parches a una resolución de 1/4. Esto se convierte en la entrada al siguiente bloque codificador, donde se repite todo este proceso hasta que tengas características de imagen con resoluciones de 1/8, 1/16 y 1/32.

3. Un decodificador ligero toma el último mapa de características (escala 1/32) del codificador y lo aumenta a una escala de 1/16. A partir de aquí, la característica se pasa a un módulo de *Fusión Selectiva de Características (SFF)*, que selecciona y combina características locales y globales de un mapa de atención para cada característica y luego la aumenta a 1/8. Este proceso se repite hasta que las características decodificadas sean del mismo tamaño que la imagen original. La salida se pasa a través de dos capas de convolución y luego se aplica una activación sigmoide para predecir la profundidad de cada píxel.

## Natural language processing

The Transformer was initially designed for machine translation, and since then, it has practically become the default architecture for solving all NLP tasks. Some tasks lend themselves to the Transformer's encoder structure, while others are better suited for the decoder. Still, other tasks make use of both the Transformer's encoder-decoder structure.

### Text classification

[BERT](model_doc/bert) is an encoder-only model and is the first model to effectively implement deep bidirectionality to learn richer representations of the text by attending to words on both sides.

1. BERT uses [WordPiece](tokenizer_summary#wordpiece) tokenization to generate a token embedding of the text. To tell the difference between a single sentence and a pair of sentences, a special `[SEP]` token is added to differentiate them. A special `[CLS]` token is added to the beginning of every sequence of text. The final output with the `[CLS]` token is used as the input to the classification head for classification tasks. BERT also adds a segment embedding to denote whether a token belongs to the first or second sentence in a pair of sentences.

2. BERT is pretrained with two objectives: masked language modeling and next-sentence prediction. In masked language modeling, some percentage of the input tokens are randomly masked, and the model needs to predict these. This solves the issue of bidirectionality, where the model could cheat and see all the words and "predict" the next word. The final hidden states of the predicted mask tokens are passed to a feedforward network with a softmax over the vocabulary to predict the masked word.

    The second pretraining object is next-sentence prediction. The model must predict whether sentence B follows sentence A. Half of the time sentence B is the next sentence, and the other half of the time, sentence B is a random sentence. The prediction, whether it is the next sentence or not, is passed to a feedforward network with a softmax over the two classes (`IsNext` and `NotNext`).

3. The input embeddings are passed through multiple encoder layers to output some final hidden states.

To use the pretrained model for text classification, add a sequence classification head on top of the base BERT model. The sequence classification head is a linear layer that accepts the final hidden states and performs a linear transformation to convert them into logits. The cross-entropy loss is calculated between the logits and target to find the most likely label.

Ready to try your hand at text classification? Check out our complete [text classification guide](tasks/sequence_classification) to learn how to finetune DistilBERT and use it for inference!

### Token classification

To use BERT for token classification tasks like named entity recognition (NER), add a token classification head on top of the base BERT model. The token classification head is a linear layer that accepts the final hidden states and performs a linear transformation to convert them into logits. The cross-entropy loss is calculated between the logits and each token to find the most likely label.

Ready to try your hand at token classification? Check out our complete [token classification guide](tasks/token_classification) to learn how to finetune DistilBERT and use it for inference!

### Question answering

To use BERT for question answering, add a span classification head on top of the base BERT model. This linear layer accepts the final hidden states and performs a linear transformation to compute the `span` start and end logits corresponding to the answer. The cross-entropy loss is calculated between the logits and the label position to find the most likely span of text corresponding to the answer.

Ready to try your hand at question answering? Check out our complete [question answering guide](tasks/question_answering) to learn how to finetune DistilBERT and use it for inference!

<Tip>

💡 Notice how easy it is to use BERT for different tasks once it's been pretrained. You only need to add a specific head to the pretrained model to manipulate the hidden states into your desired output!

</Tip>

### Text generation

[GPT-2](model_doc/gpt2) is a decoder-only model pretrained on a large amount of text. It can generate convincing (though not always true!) text given a prompt and complete other NLP tasks like question answering despite not being explicitly trained to.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gpt2_architecture.png"/>
</div>

1. GPT-2 uses [byte pair encoding (BPE)](tokenizer_summary#bytepair-encoding-bpe) to tokenize words and generate a token embedding. Positional encodings are added to the token embeddings to indicate the position of each token in the sequence. The input embeddings are passed through multiple decoder blocks to output some final hidden state. Within each decoder block, GPT-2 uses a *masked self-attention* layer which means GPT-2 can't attend to future tokens. It is only allowed to attend to tokens on the left. This is different from BERT's [`mask`] token because, in masked self-attention, an attention mask is used to set the score to `0` for future tokens.

2. The output from the decoder is passed to a language modeling head, which performs a linear transformation to convert the hidden states into logits. The label is the next token in the sequence, which are created by shifting the logits to the right by one. The cross-entropy loss is calculated between the shifted logits and the labels to output the next most likely token.

GPT-2's pretraining objective is based entirely on [causal language modeling](glossary#causal-language-modeling), predicting the next word in a sequence. This makes GPT-2 especially good at tasks that involve generating text.

Ready to try your hand at text generation? Check out our complete [causal language modeling guide](tasks/language_modeling#causal-language-modeling) to learn how to finetune DistilGPT-2 and use it for inference!

<Tip>

For more information about text generation, check out the [text generation strategies](generation_strategies) guide!

</Tip>

### Summarization

Encoder-decoder models like [BART](model_doc/bart) and [T5](model_doc/t5) are designed for the sequence-to-sequence pattern of a summarization task. We'll explain how BART works in this section, and then you can try finetuning T5 at the end.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bart_architecture.png"/>
</div>

1. BART's encoder architecture is very similar to BERT and accepts a token and positional embedding of the text. BART is pretrained by corrupting the input and then reconstructing it with the decoder. Unlike other encoders with specific corruption strategies, BART can apply any type of corruption. The *text infilling* corruption strategy works the best though. In text infilling, a number of text spans are replaced with a **single** [`mask`] token. This is important because the model has to predict the masked tokens, and it teaches the model to predict the number of missing tokens. The input embeddings and masked spans are passed through the encoder to output some final hidden states, but unlike BERT, BART doesn't add a final feedforward network at the end to predict a word.

2. The encoder's output is passed to the decoder, which must predict the masked tokens and any uncorrupted tokens from the encoder's output. This gives additional context to help the decoder restore the original text. The output from the decoder is passed to a language modeling head, which performs a linear transformation to convert the hidden states into logits. The cross-entropy loss is calculated between the logits and the label, which is just the token shifted to the right.

Ready to try your hand at summarization? Check out our complete [summarization guide](tasks/summarization) to learn how to finetune T5 and use it for inference!

<Tip>

For more information about text generation, check out the [text generation strategies](generation_strategies) guide!

</Tip>

### Translation

Translation is another example of a sequence-to-sequence task, which means you can use an encoder-decoder model like [BART](model_doc/bart) or [T5](model_doc/t5) to do it. We'll explain how BART works in this section, and then you can try finetuning T5 at the end.

BART adapts to translation by adding a separate randomly initialized encoder to map a source language to an input that can be decoded into the target language. This new encoder's embeddings are passed to the pretrained encoder instead of the original word embeddings. The source encoder is trained by updating the source encoder, positional embeddings, and input embeddings with the cross-entropy loss from the model output. The model parameters are frozen in this first step, and all the model parameters are trained together in the second step.

BART has since been followed up by a multilingual version, mBART, intended for translation and pretrained on many different languages.

Ready to try your hand at translation? Check out our complete [translation guide](tasks/summarization) to learn how to finetune T5 and use it for inference!

<Tip>

For more information about text generation, check out the [text generation strategies](generation_strategies) guide!

</Tip>