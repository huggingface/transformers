<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Addestramento con script

Insieme ai [notebooks](./noteboks/README) 🤗 Transformers, ci sono anche esempi di script che dimostrano come addestrare un modello per un task con [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow), o [JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax).

Troverai anche script che abbiamo usato nei nostri [progetti di ricerca](https://github.com/huggingface/transformers/tree/main/examples/research_projects) e [precedenti esempi](https://github.com/huggingface/transformers/tree/main/examples/legacy) a cui contribuisce per lo più la comunità. Questi script non sono attivamente mantenuti e richiedono una specifica versione di 🤗 Transformers che sarà molto probabilmente incompatibile con l'ultima versione della libreria.

Non è dato per scontato che gli script di esempio funzionino senza apportare modifiche per ogni problema, bensì potrebbe essere necessario adattare lo script al tuo caso specifico. Per aiutarti in ciò, la maggioranza degli script espone le modalità di pre-processamento dei dati, consentendoti di modificare lo script come preferisci.

Per qualsiasi feature che vorresti implementare in uno script d'esempio, per favore discutine nel [forum](https://discuss.huggingface.co/) o in un'[issue](https://github.com/huggingface/transformers/issues) prima di inviare una Pull Request. Mentre accogliamo con piacere la correzione di bug, è più improbabile che faremo la stessa con una PR che aggiunge funzionalità sacrificando la leggibilità. 

Questa guida ti mostrerà come eseguire uno script di esempio relativo al task di summarization in [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) e [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization). Tutti gli esempi funzioneranno con entrambi i framework a meno che non sia specificato altrimenti. 

## Installazione

Per eseguire con successo l'ultima versione degli script di esempio, devi **installare 🤗 Transformers dalla fonte** in un nuovo ambiente virtuale:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```
Per le precedenti versioni degli script di esempio, clicca sul pulsante di seguito:

<details>
  <summary>Esempi per versioni precedenti di 🤗 Transformers</summary>
	<ul>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.5.1/examples">v4.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.4.2/examples">v4.4.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.3.3/examples">v4.3.3</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.2.2/examples">v4.2.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.1.1/examples">v4.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.0.1/examples">v4.0.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.5.1/examples">v3.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.4.0/examples">v3.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.3.1/examples">v3.3.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.2.0/examples">v3.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.1.0/examples">v3.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.0.2/examples">v3.0.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.11.0/examples">v2.11.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.10.0/examples">v2.10.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.9.1/examples">v2.9.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.8.0/examples">v2.8.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.7.0/examples">v2.7.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.6.0/examples">v2.6.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.5.1/examples">v2.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.4.0/examples">v2.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.3.0/examples">v2.3.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.2.0/examples">v2.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.1.0/examples">v2.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.0.0/examples">v2.0.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.2.0/examples">v1.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.1.0/examples">v1.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.0.0/examples">v1.0.0</a></li>
	</ul>
</details>

Successivamente, cambia la tua attuale copia di 🤗 Transformers specificandone la versione, ad esempio v3.5.1:

```bash
git checkout tags/v3.5.1
```

 Dopo aver configurato correttamente la versione della libreria, naviga nella cartella degli esempi di tua scelta e installa i requisiti:

```bash
pip install -r requirements.txt
```

## Esegui uno script

<frameworkcontent>
<pt>

Lo script di esempio scarica e pre-processa un dataset dalla libreria 🤗 [Datasets](https://huggingface.co/docs/datasets/). Successivamente, lo script esegue il fine-tuning su un dataset usando il [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) su un'architettura che supporta la summarization. Il seguente esempio mostra come eseguire il fine-tuning di [T5-small](https://huggingface.co/t5-small) sul dataset [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail). Il modello T5 richiede un parametro addizionale `source_prefix` a causa del modo in cui è stato addestrato. Questo prefisso permette a T5 di sapere che si tratta di un task di summarization.

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
</pt>
<tf>
Lo script di esempio scarica e pre-processa un dataset dalla libreria 🤗 [Datasets](https://huggingface.co/docs/datasets/). Successivamente, lo script esegue il fine-tuning su un dataset usando Keras su un'architettura che supporta la summarization. Il seguente esempio mostra come eseguire il fine-tuning di [T5-small](https://huggingface.co/t5-small) sul dataset [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail). Il modello T5 richiede un parametro addizionale `source_prefix` a causa del modo in cui è stato addestrato. Questo prefisso permette a T5 di sapere che si tratta di un task di summarization.

```bash
python examples/tensorflow/summarization/run_summarization.py  \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```
</tf>
</frameworkcontent>

## Addestramento distribuito e precisione mista

Il [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) supporta l'addestramento distribuito e la precisione mista, che significa che puoi anche usarla in uno script. Per abilitare entrambe le funzionalità:

- Aggiunto l'argomento `fp16` per abilitare la precisione mista.
- Imposta un numero di GPU da usare con l'argomento `nproc_per_node`.

```bash
python -m torch.distributed.launch \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

Gli script TensorFlow utilizzano una [`MirroredStrategy`](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy) per il training distribuito e non devi aggiungere alcun argomento addizionale allo script di training. Lo script TensorFlow userà multiple GPU in modo predefinito se quest'ultime sono disponibili:

## Esegui uno script su TPU

<frameworkcontent>
<pt>
Le Tensor Processing Units (TPU) sono state progettate per migliorare le prestazioni. PyTorch supporta le TPU con il compilatore per deep learning [XLA](https://www.tensorflow.org/xla) (guarda [questo link](https://github.com/pytorch/xla/blob/master/README.md) per maggiori dettagli). Per usare una TPU, avvia lo script `xla_spawn.py` e usa l'argomento `num_cores` per impostare il numero di core TPU che intendi usare.

```bash
python xla_spawn.py --num_cores 8 \
    summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
</pt>
<tf>
Le Tensor Processing Units (TPU) sono state progettate per migliorare le prestazioni. Gli script TensorFlow utilizzano una [`TPUStrategy`](https://www.tensorflow.org/guide/distributed_training#tpustrategy) per eseguire l'addestramento su TPU. Per usare una TPU, passa il nome della risorsa TPU all'argomento `tpu`.

```bash
python run_summarization.py  \
    --tpu name_of_tpu_resource \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```
</tf>
</frameworkcontent>

## Esegui uno script con 🤗 Accelerate

🤗 [Accelerate](https://huggingface.co/docs/accelerate) è una libreria compatibile solo con PyTorch che offre un metodo unificato per addestrare modelli su diverse tipologie di configurazioni (CPU, multiple GPU, TPU) mantenendo una completa visibilità rispetto al ciclo di training di PyTorch. Assicurati di aver effettuato l'installazione di 🤗 Accelerate, nel caso non lo avessi fatto:

> Nota: dato che Accelerate è in rapido sviluppo, è necessario installare la versione proveniente da git per eseguire gli script:
```bash
pip install git+https://github.com/huggingface/accelerate
```

Invece che usare lo script `run_summarization.py`, devi usare lo script `run_summarization_no_trainer.py`. Gli script supportati in 🤗 Accelerate avranno un file chiamato `task_no_trainer.py` nella rispettiva cartella. Per iniziare, esegui il seguente comando per creare e salvare un file di configurazione: 

```bash
accelerate config
```

Testa la tua configurazione per assicurarti della sua correttezza:

```bash
accelerate test
```

Ora sei pronto per avviare l'addestramento:

```bash
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization
```

## Uso di un dataset personalizzato

Lo script di summarization supporta dataset personalizzati purché siano file CSV o JSON Line. Quando usi il tuo dataset, devi specificare diversi argomenti aggiuntivi:

- `train_file` e `validation_file` specificano dove si trovano i file di addestramento e validazione.
- `text_column` è il file di input da riassumere.
- `summary_column` è il file di destinazione per l'output.

Uno script di summarization usando un dataset personalizzato sarebbe simile a questo:

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --text_column text_column_name \
    --summary_column summary_column_name \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

## Testare uno script

È spesso una buona idea avviare il tuo script su un numero inferiore di esempi tratti dal dataset, per assicurarti che tutto funzioni come previsto prima di eseguire lo script sull'intero dataset, che potrebbe necessitare di ore. Usa i seguenti argomenti per limitare il dataset ad un massimo numero di esempi:

- `max_train_samples`
- `max_eval_samples`
- `max_predict_samples`

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

Non tutti gli esempi di script supportano l'argomento `max_predict_samples`. Se non sei sicuro circa il supporto di questo argomento da parte del tuo script, aggiungi l'argomento `-h` per controllare:

```bash
examples/pytorch/summarization/run_summarization.py -h
```

## Riavviare addestramento da un checkpoint

Un'altra utile opzione è riavviare un addestramento da un checkpoint precedente. Questo garantirà che tu possa riprendere da dove hai interrotto senza ricominciare se l'addestramento viene interrotto. Ci sono due metodi per riavviare l'addestramento da un checkpoint: 

Il primo metodo usa l'argomento `output_dir previous_output_dir` per riavviare l'addestramento dall'ultima versione del checkpoint contenuto in `output_dir`. In questo caso, dovresti rimuovere `overwrite_output_dir`:

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --output_dir previous_output_dir \
    --predict_with_generate
```

Il secondo metodo usa l'argomento `resume_from_checkpoint path_to_specific_checkpoint` per riavviare un addestramento da una specifica cartella di checkpoint.

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --resume_from_checkpoint path_to_specific_checkpoint \
    --predict_with_generate
```

## Condividi il tuo modello

Tutti gli script possono caricare il tuo modello finale al [Model Hub](https://huggingface.co/models). Prima di iniziare, assicurati di aver effettuato l'accesso su Hugging Face:

```bash
huggingface-cli login
```

Poi, aggiungi l'argomento `push_to_hub` allo script. Questo argomento consentirà di creare un repository con il tuo username Hugging Face e la cartella specificata in `output_dir`.

Per dare uno specifico nome al repository, usa l'argomento `push_to_hub_model_id`. Il repository verrà automaticamente elencata sotto al tuo namespace.

Il seguente esempio mostra come caricare un modello specificando il nome del repository:

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --push_to_hub \
    --push_to_hub_model_id finetuned-t5-cnn_dailymail \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
