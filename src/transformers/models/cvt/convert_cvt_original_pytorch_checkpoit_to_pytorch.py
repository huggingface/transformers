import torch
from collections import OrderedDict
from transformers import CvtConfig, CvtForImageClassification

def embeddings(idx):
    """
    The function helps in renaming embedding layer weights.

    Args:
        idx: stage number in original model
    """
    embed = []
    embed.append((f'cvt.encoder.patch_embeddings.{idx}.conv_embeddings.projection.weight', f'stage{idx}.patch_embed.proj.weight'))
    embed.append((f'cvt.encoder.patch_embeddings.{idx}.conv_embeddings.projection.bias', f'stage{idx}.patch_embed.proj.bias'))
    embed.append((f'cvt.encoder.patch_embeddings.{idx}.conv_embeddings.norm.weight', f'stage{idx}.patch_embed.norm.weight'))
    embed.append((f'cvt.encoder.patch_embeddings.{idx}.conv_embeddings.norm.bias', f'stage{idx}.patch_embed.norm.bias'))
    return embed

def attention(idx, cnt):
    """
    The function helps in renaming attention block layers weights.

    Args:
        idx: stage number in original model
        cnt: count of blocks in each stage
    """
    attention_weights = []
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_query.convolution.weight', f'stage{idx}.blocks.{cnt}.attn.conv_proj_q.conv.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_query.batch_norm.weight', f'stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_query.batch_norm.bias', f'stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.bias'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_query.batch_norm.running_mean', f'stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.running_mean'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_query.batch_norm.running_var', f'stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.running_var'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_query.batch_norm.num_batches_tracked', f'stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.num_batches_tracked'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_key.convolution.weight', f'stage{idx}.blocks.{cnt}.attn.conv_proj_k.conv.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_key.batch_norm.weight', f'stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_key.batch_norm.bias', f'stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.bias'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_key.batch_norm.running_mean', f'stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.running_mean'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_key.batch_norm.running_var', f'stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.running_var'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_key.batch_norm.num_batches_tracked', f'stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.num_batches_tracked'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_value.convolution.weight', f'stage{idx}.blocks.{cnt}.attn.conv_proj_v.conv.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_value.batch_norm.weight', f'stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_value.batch_norm.bias', f'stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.bias'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_value.batch_norm.running_mean', f'stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.running_mean'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_value.batch_norm.running_var', f'stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.running_var'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.conv_projection_value.batch_norm.num_batches_tracked', f'stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.num_batches_tracked'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.projection_query.weight', f'stage{idx}.blocks.{cnt}.attn.proj_q.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.projection_query.bias', f'stage{idx}.blocks.{cnt}.attn.proj_q.bias'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.projection_key.weight', f'stage{idx}.blocks.{cnt}.attn.proj_k.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.projection_key.bias', f'stage{idx}.blocks.{cnt}.attn.proj_k.bias'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.projection_value.weight', f'stage{idx}.blocks.{cnt}.attn.proj_v.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.attention.projection_value.bias', f'stage{idx}.blocks.{cnt}.attn.proj_v.bias'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.output.dense.weight', f'stage{idx}.blocks.{cnt}.attn.proj.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.attention.output.dense.bias', f'stage{idx}.blocks.{cnt}.attn.proj.bias'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.intermediate.dense.weight', f'stage{idx}.blocks.{cnt}.mlp.fc1.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.intermediate.dense.bias', f'stage{idx}.blocks.{cnt}.mlp.fc1.bias'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.output.dense.weight', f'stage{idx}.blocks.{cnt}.mlp.fc2.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.output.dense.bias', f'stage{idx}.blocks.{cnt}.mlp.fc2.bias'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.layernorm_before.weight', f'stage{idx}.blocks.{cnt}.norm1.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.layernorm_before.bias', f'stage{idx}.blocks.{cnt}.norm1.bias'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.layernorm_after.weight', f'stage{idx}.blocks.{cnt}.norm2.weight'))
    attention_weights.append((f'cvt.encoder.block.{idx}.{cnt}.layernorm_after.bias', f'stage{idx}.blocks.{cnt}.norm2.bias'))
    return attention_weights

def cls_token():
    """
    Function helps in renaming cls_token weights
    """
    token = []
    token.append(('cvt.encoder.cls_token', 'stage2.cls_token'))
    return token

def final():
    """
    Function helps in renaming final classification layer
    """
    head = []
    head.append(('layernorm.weight', 'norm.weight'))
    head.append(('layernorm.bias', 'norm.bias'))
    head.append(('classifier.weight', 'head.weight'))
    head.append(('classifier.bias', 'head.bias'))
    return head

if __name__=="__main__":
    path = "new_hugging_face_model.bin"
    config = CvtConfig()
    model = CvtForImageClassification(config)
    original_file = "CvT-13-224x224-IN-1k.pth"
    original_weights = torch.load(original_file, map_location=torch.device("cpu"))

    hugging_face_weights = OrderedDict()
    list_of_state_dict = cls_token()
    for i in range(config.num_stages):
        list_of_state_dict = list_of_state_dict + embeddings(i)
    
    for i in range(config.num_stages):
        for j in range(config.depth[i]):
            list_of_state_dict = list_of_state_dict + attention(i, j)
    
    list_of_state_dict = list_of_state_dict + final()

    for i in range(len(list_of_state_dict)):
        hugging_face_weights[list_of_state_dict[i][0]] = original_weights[list_of_state_dict[i][1]]
    
    model.load_state_dict(hugging_face_weights)
    torch.save(model.state_dict(), path)

    
    