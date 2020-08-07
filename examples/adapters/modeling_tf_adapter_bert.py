import tensorflow as tf
import copy
import numpy as np
from transformers.modeling_tf_bert import gelu, gelu_new, swish, ACT2FN

class AdapterModule(tf.keras.Model):
  __instance_number = 0
  __instance = None
  @staticmethod 
  def getInstance(input_size, bottleneck_size, non_linearity):
    if AdapterModule.__instance_number % 2 == 0:
        AdapterModule(input_size, bottleneck_size, non_linearity)
    AdapterModule.__instance_number += 1
    return AdapterModule.__instance

  def __init__(self, input_size, bottleneck_size, non_linearity, *inputs, **kwargs):
    super(AdapterModule, self).__init__(name="AdapterModule")

    self.non_linearity = ACT2FN[non_linearity]

    self.down_project = tf.keras.layers.Dense(
                        bottleneck_size,
                        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                        bias_initializer="zeros",
                        name="feedforward_downproject")
    
    self.up_project = tf.keras.layers.Dense(
                        input_size,
                        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                        bias_initializer="zeros",
                        name="feedforward_upproject")
    
    AdapterModule.__instance = self

  def call(self, inputs, **kwargs):

    output = self.down_project(inputs)
    output = self.non_linearity(output)
    output = self.up_project(output)
    output = output + inputs
    return output


class TFBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, pretrained_self_dense, pretrained_self_ln, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = copy.deepcopy(pretrained_self_dense)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.adapter = AdapterModule.getInstance(input_size=config.hidden_size, bottleneck_size=config.bottleneck_size, non_linearity=config.non_linearity)
        self.LayerNorm = copy.deepcopy(pretrained_self_ln)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFBertOutput(tf.keras.layers.Layer):
    def __init__(self, pretrained_out_dense, pretrained_out_ln, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = copy.deepcopy(pretrained_out_dense)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.adapter = AdapterModule.getInstance(input_size=config.hidden_size, bottleneck_size=config.bottleneck_size, non_linearity=config.non_linearity)
        self.LayerNorm = copy.deepcopy(pretrained_out_ln)
        
    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class AdapterBertModel(tf.keras.Model):
  def __init__(self, bert_model, num_labels, *inputs, **kwargs):
    super(AdapterBertModel, self).__init__(name="AdapterBertModel")
    self.bert = bert_model
    self.dropout = tf.keras.layers.Dropout(0.1)
    self.classifier = tf.keras.layers.Dense(
                      num_labels,
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                      name="classifier")
    
  
  def call(self, inputs, **kwargs):

    outputs = self.bert(inputs, **kwargs)
    pooled_out = outputs[1]

    droped_out = self.dropout(pooled_out, training=kwargs.get("training", False))
    output = self.classifier(droped_out)
    return output
