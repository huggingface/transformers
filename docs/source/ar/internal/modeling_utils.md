# الطبقات المخصصة ووظائف المساعدة

تصف هذه الصفحة جميع الطبقات المخصصة التي تستخدمها المكتبة، بالإضافة إلى وظائف المساعدة التي توفرها لنمذجة. معظم هذه الوظائف مفيدة فقط إذا كنت تدرس شفرة نماذج المكتبة.

## وحدات PyTorch المخصصة

- pytorch_utils.Conv1D

- modeling_utils.PoolerStartLogits

    - forward

- modeling_utils.PoolerEndLogits

    - forward

- modeling_utils.PoolerAnswerClass

    - forward

- modeling_utils.SquadHeadOutput

- modeling_utils.SQuADHead

    - forward

- modeling_utils.SequenceSummary

    - forward

## وظائف مساعدة PyTorch

- pytorch_utils.apply_chunking_to_forward

- pytorch_utils.find_pruneable_heads_and_indices

- pytorch_utils.prune_layer

- pytorch_utils.prune_conv1d_layer

- pytorch_utils.prune_linear_layer

## طبقات TensorFlow المخصصة

- modeling_tf_utils.TFConv1D

- modeling_tf_utils.TFSequenceSummary

## دالات خسارة TensorFlow

- modeling_tf_utils.TFCausalLanguageModelingLoss

- modeling_tf_utils.TFMaskedLanguageModelingLoss

- modeling_tf_utils.TFMultipleChoiceLoss

- modeling_tf_utils.TFQuestionAnsweringLoss

- modeling_tf_utils.TFSequenceClassificationLoss

- modeling_tf_utils.TFTokenClassificationLoss

## وظائف مساعدة TensorFlow

- modeling_tf_utils.get_initializer

- modeling_tf_utils.keras_serializable

- modeling_tf_utils.shape_list