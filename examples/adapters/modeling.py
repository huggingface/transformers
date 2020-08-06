import tensorflow as tf
import copy
import numpy as np

def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def swish(x):
    return x * tf.sigmoid(x)

def tanh(x):
    return x * tf.tanh(x)

ACT2FN = {
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
    "tanh": tf.keras.layers.Activation(tanh),
}

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



class TFBertLayer(tf.keras.layers.Layer):
    def __init__(self, pretrained_self_attn, pretrained_self_dense, pretrained_self_ln, pretrained_intermediate, pretrained_out_dense, pretrained_out_ln, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFBertAttention(pretrained_self_attn, pretrained_self_dense, pretrained_self_ln, config, name="attention")
        # shared new task layer normalization in each layer
        # self.TaskLayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="TaskLayerNorm")
        self.intermediate = copy.deepcopy(pretrained_intermediate)
        self.bert_output = TFBertOutput(pretrained_out_dense, pretrained_out_ln, config, name="output")


    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask, output_attentions = inputs

        attention_outputs = self.attention(
            [hidden_states, attention_mask, head_mask, output_attentions], training=training
        )
        attention_output = attention_outputs[0]
        # attention_output = self.TaskLayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output([intermediate_output, attention_output], training=training)
        # layer_output = self.TaskLayerNorm(layer_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs



class TFBertAttention(tf.keras.layers.Layer):
    def __init__(self, pretrained_self_attn, pretrained_self_dense, pretrained_self_ln, config, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = copy.deepcopy(pretrained_self_attn)
        self.dense_output = TFBertSelfOutput(pretrained_self_dense, pretrained_self_ln, config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        input_tensor, attention_mask, head_mask, output_attentions = inputs

        self_outputs = self.self_attention(
            [input_tensor, attention_mask, head_mask, output_attentions], training=training
        )
        attention_output = self.dense_output([self_outputs[0], input_tensor], training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


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
