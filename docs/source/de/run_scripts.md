<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainieren mit einem Skript

Neben den 🤗 Transformers [notebooks](./notebooks) gibt es auch Beispielskripte, die zeigen, wie man ein Modell für eine Aufgabe mit [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow) oder [JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax) trainiert.

Sie werden auch Skripte finden, die wir in unseren [Forschungsprojekten](https://github.com/huggingface/transformers/tree/main/examples/research_projects) und [Legacy-Beispielen](https://github.com/huggingface/transformers/tree/main/examples/legacy) verwendet haben und die größtenteils von der Community stammen. Diese Skripte werden nicht aktiv gepflegt und erfordern eine bestimmte Version von 🤗 Transformers, die höchstwahrscheinlich nicht mit der neuesten Version der Bibliothek kompatibel ist.

Es wird nicht erwartet, dass die Beispielskripte bei jedem Problem sofort funktionieren. Möglicherweise müssen Sie das Skript an das Problem anpassen, das Sie zu lösen versuchen. Um Ihnen dabei zu helfen, legen die meisten Skripte vollständig offen, wie die Daten vorverarbeitet werden, so dass Sie sie nach Bedarf für Ihren Anwendungsfall bearbeiten können.

Für jede Funktion, die Sie in einem Beispielskript implementieren möchten, diskutieren Sie bitte im [Forum](https://discuss.huggingface.co/) oder in einem [issue](https://github.com/huggingface/transformers/issues), bevor Sie einen Pull Request einreichen. Wir freuen uns zwar über Fehlerkorrekturen, aber es ist unwahrscheinlich, dass wir einen Pull Request zusammenführen, der mehr Funktionalität auf Kosten der Lesbarkeit hinzufügt.

Diese Anleitung zeigt Ihnen, wie Sie ein Beispiel für ein Trainingsskript zur Zusammenfassung in [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) und [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization) ausführen können. Sofern nicht anders angegeben, sollten alle Beispiele mit beiden Frameworks funktionieren.

## Einrichtung

Um die neueste Version der Beispielskripte erfolgreich auszuführen, **müssen Sie 🤗 Transformers aus dem Quellcode** in einer neuen virtuellen Umgebung installieren:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

Für ältere Versionen der Beispielskripte klicken Sie auf die Umschalttaste unten:

<details>
  <summary>Beispiele für ältere Versionen von 🤗 Transformers</summary>
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

Dann stellen Sie Ihren aktuellen Klon von 🤗 Transformers auf eine bestimmte Version um, z.B. v3.5.1:

```bash
git checkout tags/v3.5.1
```

Nachdem Sie die richtige Bibliotheksversion eingerichtet haben, navigieren Sie zu dem Beispielordner Ihrer Wahl und installieren die beispielspezifischen Anforderungen:

```bash
pip install -r requirements.txt
```

## Ein Skript ausführen

<frameworkcontent>
<pt>
Das Beispielskript lädt einen Datensatz aus der 🤗 [Datasets](https://huggingface.co/docs/datasets/) Bibliothek herunter und verarbeitet ihn vor. Dann nimmt das Skript eine Feinabstimmung eines Datensatzes mit dem [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) auf einer Architektur vor, die eine Zusammenfassung unterstützt. Das folgende Beispiel zeigt, wie die Feinabstimmung von [T5-small](https://huggingface.co/google-t5/t5-small) auf dem Datensatz [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) durchgeführt wird. Das T5-Modell benötigt aufgrund der Art und Weise, wie es trainiert wurde, ein zusätzliches Argument `source_prefix`. Mit dieser Eingabeaufforderung weiß T5, dass es sich um eine Zusammenfassungsaufgabe handelt.

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
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
Das Beispielskript lädt einen Datensatz aus der 🤗 [Datasets](https://huggingface.co/docs/datasets/) Bibliothek herunter und verarbeitet ihn vor. Anschließend nimmt das Skript die Feinabstimmung eines Datensatzes mit Keras auf einer Architektur vor, die die Zusammenfassung unterstützt. Das folgende Beispiel zeigt, wie die Feinabstimmung von [T5-small](https://huggingface.co/google-t5/t5-small) auf dem [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) Datensatz durchgeführt wird. Das T5-Modell benötigt aufgrund der Art und Weise, wie es trainiert wurde, ein zusätzliches Argument `source_prefix`. Mit dieser Eingabeaufforderung weiß T5, dass es sich um eine Zusammenfassungsaufgabe handelt.

```bash
python examples/tensorflow/summarization/run_summarization.py  \
    --model_name_or_path google-t5/t5-small \
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

## Verteiltes Training und gemischte Präzision

Der [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) unterstützt verteiltes Training und gemischte Präzision, d.h. Sie können ihn auch in einem Skript verwenden. So aktivieren Sie diese beiden Funktionen:

- Fügen Sie das Argument `fp16` hinzu, um gemischte Genauigkeit zu aktivieren.
- Legen Sie die Anzahl der zu verwendenden GPUs mit dem Argument `nproc_per_node` fest.

```bash
torchrun \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path google-t5/t5-small \
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

TensorFlow-Skripte verwenden eine [`MirroredStrategy`](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy) für verteiltes Training, und Sie müssen dem Trainingsskript keine zusätzlichen Argumente hinzufügen. Das TensorFlow-Skript verwendet standardmäßig mehrere GPUs, wenn diese verfügbar sind.

## Ein Skript auf einer TPU ausführen

<frameworkcontent>
<pt>
Tensor Processing Units (TPUs) sind speziell für die Beschleunigung der Leistung konzipiert. PyTorch unterstützt TPUs mit dem [XLA](https://www.tensorflow.org/xla) Deep Learning Compiler (siehe [hier](https://github.com/pytorch/xla/blob/master/README.md) für weitere Details). Um eine TPU zu verwenden, starten Sie das Skript `xla_spawn.py` und verwenden das Argument `num_cores`, um die Anzahl der TPU-Kerne festzulegen, die Sie verwenden möchten.

```bash
python xla_spawn.py --num_cores 8 \
    summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
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
Tensor Processing Units (TPUs) sind speziell für die Beschleunigung der Leistung konzipiert. TensorFlow Skripte verwenden eine [`TPUStrategy`](https://www.tensorflow.org/guide/distributed_training#tpustrategy) für das Training auf TPUs. Um eine TPU zu verwenden, übergeben Sie den Namen der TPU-Ressource an das Argument `tpu`.

```bash
python run_summarization.py  \
    --tpu name_of_tpu_resource \
    --model_name_or_path google-t5/t5-small \
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

## Führen Sie ein Skript mit 🤗 Accelerate aus.

🤗 [Accelerate](https://huggingface.co/docs/accelerate) ist eine reine PyTorch-Bibliothek, die eine einheitliche Methode für das Training eines Modells auf verschiedenen Arten von Setups (nur CPU, mehrere GPUs, TPUs) bietet und dabei die vollständige Transparenz der PyTorch-Trainingsschleife beibehält. Stellen Sie sicher, dass Sie 🤗 Accelerate installiert haben, wenn Sie es nicht bereits haben:

> Hinweis: Da Accelerate schnell weiterentwickelt wird, muss die Git-Version von Accelerate installiert sein, um die Skripte auszuführen.
```bash
pip install git+https://github.com/huggingface/accelerate
```

Anstelle des Skripts `run_summarization.py` müssen Sie das Skript `run_summarization_no_trainer.py` verwenden. Die von Accelerate unterstützten Skripte haben eine Datei `task_no_trainer.py` im Ordner. Beginnen Sie mit dem folgenden Befehl, um eine Konfigurationsdatei zu erstellen und zu speichern:

```bash
accelerate config
```

Testen Sie Ihre Einrichtung, um sicherzustellen, dass sie korrekt konfiguriert ist:

```bash
accelerate test
```

Jetzt sind Sie bereit, das Training zu starten:

```bash
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization
```

## Verwenden Sie einen benutzerdefinierten Datensatz

Das Verdichtungsskript unterstützt benutzerdefinierte Datensätze, solange es sich um eine CSV- oder JSON-Line-Datei handelt. Wenn Sie Ihren eigenen Datensatz verwenden, müssen Sie mehrere zusätzliche Argumente angeben:

- `train_file` und `validation_file` geben den Pfad zu Ihren Trainings- und Validierungsdateien an.
- `text_column` ist der Eingabetext, der zusammengefasst werden soll.
- Summary_column" ist der auszugebende Zieltext.

Ein Zusammenfassungsskript, das einen benutzerdefinierten Datensatz verwendet, würde wie folgt aussehen:

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
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

## Testen Sie ein Skript

Es ist oft eine gute Idee, Ihr Skript an einer kleineren Anzahl von Beispielen für Datensätze auszuführen, um sicherzustellen, dass alles wie erwartet funktioniert, bevor Sie sich auf einen ganzen Datensatz festlegen, dessen Fertigstellung Stunden dauern kann. Verwenden Sie die folgenden Argumente, um den Datensatz auf eine maximale Anzahl von Stichproben zu beschränken:

- `max_train_samples`
- `max_eval_samples`
- `max_predict_samples`

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
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

Nicht alle Beispielskripte unterstützen das Argument `max_predict_samples`. Wenn Sie sich nicht sicher sind, ob Ihr Skript dieses Argument unterstützt, fügen Sie das Argument `-h` hinzu, um dies zu überprüfen:

```bash
examples/pytorch/summarization/run_summarization.py -h
```

## Training vom Kontrollpunkt fortsetzen

Eine weitere hilfreiche Option, die Sie aktivieren können, ist die Wiederaufnahme des Trainings von einem früheren Kontrollpunkt aus. Auf diese Weise können Sie im Falle einer Unterbrechung Ihres Trainings dort weitermachen, wo Sie aufgehört haben, ohne von vorne beginnen zu müssen. Es gibt zwei Methoden, um das Training von einem Kontrollpunkt aus wieder aufzunehmen.

Die erste Methode verwendet das Argument `output_dir previous_output_dir`, um das Training ab dem letzten in `output_dir` gespeicherten Kontrollpunkt wieder aufzunehmen. In diesem Fall sollten Sie `overwrite_output_dir` entfernen:

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
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

Die zweite Methode verwendet das Argument `Resume_from_checkpoint path_to_specific_checkpoint`, um das Training ab einem bestimmten Checkpoint-Ordner wieder aufzunehmen.

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
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

## Teilen Sie Ihr Modell

Alle Skripte können Ihr endgültiges Modell in den [Model Hub](https://huggingface.co/models) hochladen. Stellen Sie sicher, dass Sie bei Hugging Face angemeldet sind, bevor Sie beginnen:

```bash
huggingface-cli login
```

Dann fügen Sie dem Skript das Argument `push_to_hub` hinzu. Mit diesem Argument wird ein Repository mit Ihrem Hugging Face-Benutzernamen und dem in `output_dir` angegebenen Ordnernamen erstellt.

Wenn Sie Ihrem Repository einen bestimmten Namen geben möchten, fügen Sie ihn mit dem Argument `push_to_hub_model_id` hinzu. Das Repository wird automatisch unter Ihrem Namensraum aufgeführt.

Das folgende Beispiel zeigt, wie Sie ein Modell mit einem bestimmten Repository-Namen hochladen können:

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
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