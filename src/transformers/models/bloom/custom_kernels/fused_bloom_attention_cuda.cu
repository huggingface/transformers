#include <ATen/ATen.h>
#include <torch/torch.h>
#include <vector>

#include <iostream>
#include <optional>

std::vector<at::Tensor> bloom_attention_compute_attention(
    at::Tensor fused_qkv,
    std::optional<std::vector<at::Tensor>> layer_past,
    at::Tensor alibi,
    at::Tensor attention_mask,
    std::optional<at::Tensor> head_mask,
    float beta,
    float inv_norm_factor,
    int num_heads,
    bool use_cache
) {
    // batch_size, q_length, three_times_hidden_size = fused_qkv.shape
    auto batch_size = fused_qkv.size(0);
    auto q_length = fused_qkv.size(1);
    auto three_times_hidden_size = fused_qkv.size(2);

    // head_dim = three_times_hidden_size / (3 * num_heads)
    // batch_size_times_num_heads = batch_size * num_heads
    auto three_times_num_heads = 3 * num_heads;
    auto head_dim = three_times_hidden_size / three_times_num_heads;
    auto batch_size_times_num_heads = batch_size * num_heads;

    // # 3 x [batch_size, q_length, num_heads, head_dim]
    // (query_layer, key_layer, value_layer) = _split_heads(fused_qkv, num_heads=num_heads,
    //                                                      head_dim=head_dim)

    /*
    we flatten _split_heads
    */
    // fused_qkv = fused_qkv.view(batch_size, q_length, num_heads, 3 * head_dim)
    // query_layer, key_layer, value_layer = fused_qkv.split(head_dim, dim=-1)
    fused_qkv = fused_qkv.view({batch_size, q_length, num_heads, three_times_num_heads});
    auto tensor_list = fused_qkv.tensor_split(head_dim, -1);
    auto query_layer = tensor_list[0];
    auto key_layer = tensor_list[1];
    auto value_layer = tensor_list[2];

    // query_layer = query_layer.transpose(1, 2).reshape(batch_size_times_num_heads, q_length, head_dim)
    // key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size_times_num_heads, head_dim, q_length)
    // value_layer = value_layer.transpose(1, 2).reshape(batch_size_times_num_heads, q_length, head_dim)
    query_layer = query_layer.transpose(1, 2).reshape({batch_size_times_num_heads, q_length, three_times_num_heads});
    key_layer = key_layer.permute({0, 2, 3, 1}).reshape({batch_size_times_num_heads, three_times_num_heads, q_length});
    value_layer = value_layer.transpose(1, 2).reshape({batch_size_times_num_heads, q_length, three_times_num_heads});

    /*
    End of split_heads
    */

    //  if layer_past is not None:
    //      past_key, past_value = layer_past
    //      # concatenate along q_length dimension:
    //      #  - key: [batch_size * self.num_heads, head_dim, kv_length]
    //      #  - value: [batch_size * self.num_heads, kv_length, head_dim]
    //      key_layer = torch.cat((past_key, key_layer), dim=2)
    //      value_layer = torch.cat((past_value, value_layer), dim=1)
    if (layer_past) {
        auto past_key = layer_past[0];
        auto past_value = layer_past[1];
        key_layer = at::cat({past_key, key_layer}, 2);
        key_layer = at::cat({past_value, value_layer}, 1);
    }

    // TODO @thomasw21 probably unneeded
    //  _, _, kv_length = key_layer.shape

    //  if use_cache is True:
    //      present = (key_layer, value_layer)
    //  else:
    //      present = None
    std::optional<std::vector<at::Tensor>> present;
    if use_cache {
        present = {key_layer, value_layer};
    } else {
        present = at::nullopt;
    }

    //  # [batch_size * num_heads, q_length, kv_length]
    //  # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
    //  attention_scores = alibi.baddbmm(
    //      batch1=query_layer,
    //      batch2=key_layer,
    //      beta=beta,
    //      alpha=inv_norm_factor,
    //  )
    auto attention_scores = alibi.baddbmm(query_layer, key_layer, beta, inv_norm_factor);

    //  # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
    //  input_dtype = attention_scores.dtype
    //  # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
    //  if input_dtype == torch.float16:
    //      attention_scores = attention_scores.to(torch.float)
    //  # torch.finfo not supported by torch.jit, we temporarily remplace with `-1e34`
    //  attn_weights = attention_scores.masked_fill_(attention_mask, torch.finfo(attention_scores.dtype).min)
    //  attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)
    auto input_dtype = attention_scores.dtype;
    if input_dtype == at::ScalarType::Float {
        attention_scores = attention_scores.to(at::ScalarType::Float)
    };
    // TODO @thomasw21 Figure out how to get minimul value
    auto attn_weights = attention_scores.masked_fill_(attention_mask, -1e34);
    auto attention_probs = attn_weights.softmax(-1, at::ScalarType::Float).to(input_dtype);

    //  TODO @thomasw21: We ignore it for now
    //  # # [batch_size, num_heads, q_length, kv_length]
    //  # attention_probs = self.attention_dropout(attention_probs)

    //  TODO @thomasw21: We ignore it for now
    //  if head_mask is not None:
    //      attention_probs = attention_probs * head_mask

    //  # matmul: [batch_size * num_heads, q_length, head_dim]
    //  context_layer = torch.bmm(attention_probs, value_layer, out=query_layer)
    auto context_layer = attention_probs.bmm(value_layer)
    //  # change view [batch_size, num_heads, q_length, head_dim]
    //  context_layer = _merge_heads(context_layer, num_heads=num_heads, head_dim=head_dim)

    /*
    we flatten `_merge_heads`
    */

    /*
    end `_merge_heads`
    */
    // # What we want to achieve is:
    // # batch_size * num_heads, q_length, head_dim -> batch_size, q_length, num_heads * head_dim
    // batch_size_and_num_heads, q_length, _ = x.shape
    // batch_size = batch_size_and_num_heads // num_heads


    //  # First view to decompose the batch size
    //  # batch_size * num_heads, q_length, head_dim -> batch_size, num_heads, q_length, head_dim
    //  x = x.view(batch_size, num_heads, q_length, head_dim)
    context_layer = context_layer.view({batch_size, num_heads, q_length, head_dim});

    //  # batch_size, num_heads, q_length, head_dim -> batch_size, q_length, num_heads, head_dim
    //  x = x.permute(0, 2, 1, 3)
    context_layer = context_layer.permute({0, 2, 1, 3});

    //  # batch_size, q_length, num_heads, head_dim -> batch_size, q_length, num_heads * head_dim
    //  return x.reshape(batch_size, q_length, num_heads * head_dim)
    context_layer = context_layer.reshape({batch_size, q_length, three_times_hidden_size / 3});
    return {context_layer, present, attention_probs};
}