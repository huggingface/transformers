def rename_key(orig_key):
    if 'model' in orig_key:
        orig_key = orig_key.replace('model.', '')
    if 'norm1' in orig_key:
        orig_key = orig_key.replace('norm1', 'attention.output.LayerNorm')
    if 'norm2' in orig_key:
        orig_key = orig_key.replace('norm2', 'output.LayerNorm')    
    if 'norm' in orig_key:
        orig_key = orig_key.replace('norm', 'LayerNorm')
    if 'transformer' in orig_key:
        layer_num = orig_key.split('.')[0][-1]
        orig_key = orig_key.replace(f'transformer_{layer_num}', f'encoder.layer.{layer_num}')
    if 'mha' in orig_key:
        orig_key = orig_key.replace('mha', 'attention')
    if 'W_q' in orig_key:
        orig_key = orig_key.replace('W_q', 'self.query')
    if 'W_k' in orig_key:
        orig_key = orig_key.replace('W_k', 'self.key')
    if 'W_v' in orig_key:
        orig_key = orig_key.replace('W_v', 'self.value')
    if 'ff1' in orig_key:
        orig_key = orig_key.replace('ff1', 'intermediate.dense')
    if 'ff2' in orig_key:
        orig_key = orig_key.replace('ff2', 'output.dense')
    if 'ff' in orig_key:
        orig_key = orig_key.replace('ff', 'output.dense')
    
    return orig_key
    