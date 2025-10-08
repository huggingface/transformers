from transformers import MarianTokenizer, MarianMTModel
import torch
import numpy as np
import onnxruntime as ort

# Model & tokenizer
model_name = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
model.eval()

# Test PyTorch generation first
print(" Testing PyTorch generation...")
input_sentence = "Using handheld GPS devices and programs like Google Earth, members of the Trio Tribe..."
inputs = tokenizer(input_sentence, return_tensors="pt", padding="max_length",
                   truncation=True, max_length=64)

with torch.no_grad():
    hf_output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=20,
        num_beams=1,
        do_sample=False
    )

print(" Expected output:")
print(tokenizer.decode(hf_output_ids[0][:15], skip_special_tokens=False))
print("Expected token IDs:", hf_output_ids[0][:15].tolist())

# Compute encoder hidden states and cross-attention cache
print("\n Computing encoder outputs...")
with torch.no_grad():
    encoder_outputs = model.model.encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
    )
    encoder_hidden_states = encoder_outputs.last_hidden_state

    # Precompute cross-attention cache
    cross_attn_cache = []
    for layer in model.model.decoder.layers:
        cross_attn = layer.encoder_attn
        key_states = cross_attn.k_proj(encoder_hidden_states)
        value_states = cross_attn.v_proj(encoder_hidden_states)

        batch_size, seq_len = encoder_hidden_states.shape[:2]
        key_states = key_states.view(
            batch_size, seq_len, cross_attn.num_heads, cross_attn.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, cross_attn.num_heads, cross_attn.head_dim
        ).transpose(1, 2)

        cross_attn_cache.append((key_states, value_states))

print(
    f" Precomputed cross-attention cache for {len(cross_attn_cache)} layers")

# Decoder wrapper (with encoder context)

class DecoderWithPastWrapper(torch.nn.Module):
    def __init__(self, model, cross_attn_cache, encoder_attention_mask, encoder_hidden_states):
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head
        self.config = model.config
        # Capture final logits bias used by Marian for generation
        self.final_logits_bias = model.final_logits_bias
        # Store cross-attention cache as buffers
        for i, (k, v) in enumerate(cross_attn_cache):
            self.register_buffer(f'cross_k_{i}', k)
            self.register_buffer(f'cross_v_{i}', v)
        # Store encoder attention and hidden states as buffers (fallback)
        self.register_buffer('encoder_attention_mask', encoder_attention_mask)
        self.register_buffer('encoder_hidden_states', encoder_hidden_states)
        self.n_layers = len(cross_attn_cache)

    def forward(self, input_ids, *flat_past, encoder_attention_mask=None):
        """
        Args:
            input_ids: [batch, seq_len]
            encoder_attention_mask: optional [batch, enc_seq]
            flat_past: past tensors for self-attention per layer: (k, v, ...)
                       shape: [batch, n_heads, past_seq_len, head_dim]
        """
        batch_size = input_ids.shape[0]
        n_heads = self.config.decoder_attention_heads
        head_dim = self.config.d_model // n_heads

        # Decide which encoder_attention_mask to use
        if encoder_attention_mask is None:
            encoder_attention_mask = getattr(
                self, 'encoder_attention_mask', None)

        # Helper to check if an item in flat_past looks like a valid past (4-D)
        def _is_valid_past_tensor(x):
            return hasattr(x, 'ndim') and x.ndim == 4

        # Reconstruct past_key_values
        past_key_values = []

        provided_past = any(_is_valid_past_tensor(x) for x in flat_past)

        if provided_past:
            expected_len = self.n_layers * 2
            if len(flat_past) < expected_len:
                padded = list(flat_past) + [torch.zeros(
                    (batch_size, n_heads, 0, head_dim),
                    dtype=getattr(self, f'cross_k_0').dtype,
                    device=getattr(self, f'cross_k_0').device
                )] * (expected_len - len(flat_past))
            else:
                padded = list(flat_past[:expected_len])

            for i in range(self.n_layers):
                self_k = padded[i * 2]
                self_v = padded[i * 2 + 1]
                cross_k = getattr(self, f'cross_k_{i}')
                cross_v = getattr(self, f'cross_v_{i}')
                past_key_values.append((self_k, self_v, cross_k, cross_v))
        else:
            for i in range(self.n_layers):
                cross_k = getattr(self, f'cross_k_{i}')
                cross_v = getattr(self, f'cross_v_{i}')
                empty_k = torch.zeros(batch_size, n_heads, 0, head_dim,
                                      dtype=cross_k.dtype, device=cross_k.device)
                empty_v = torch.zeros(batch_size, n_heads, 0, head_dim,
                                      dtype=cross_v.dtype, device=cross_v.device)
                past_key_values.append((empty_k, empty_v, cross_k, cross_v))

        past_key_values = tuple(past_key_values)

        output = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=self.encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        logits = self.lm_head(output.last_hidden_state)
        # Marian adds a learned bias to logits
        logits = logits + self.final_logits_bias

        flat_output = [logits]
        for layer_past in output.past_key_values:
            flat_output.append(layer_past[0])  # self key
            flat_output.append(layer_past[1])  # self value

        return tuple(flat_output)


decoder_wrapper = DecoderWithPastWrapper(
    model, cross_attn_cache, inputs["attention_mask"], encoder_hidden_states)

# Test wrapper in PyTorch

print("\n Testing PyTorch wrapper...")
with torch.no_grad():
    decoder_input_ids_torch = torch.tensor(
        [[model.config.decoder_start_token_id]])

    # First call with empty past
    outputs = decoder_wrapper(
        decoder_input_ids_torch
    )

    logits = outputs[0]
    first_token = int(torch.argmax(logits[0, -1, :]))
    print(
        f"Wrapper first token: {first_token} '{tokenizer.decode([first_token])}'")

    # Compare with direct
    direct_out = model.model.decoder(
        input_ids=decoder_input_ids_torch,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=inputs["attention_mask"],
        use_cache=True,
        return_dict=True,
    )
    direct_logits = model.lm_head(direct_out.last_hidden_state)
    direct_token = int(torch.argmax(direct_logits[0, -1, :]))
    print(
        f"Direct first token: {direct_token} '{tokenizer.decode([direct_token])}'")

    if first_token == direct_token:
        print(" Wrapper matches direct decoder!")
    else:
        print(f" Mismatch: expected {direct_token}, got {first_token}")

    # Test second step with past
    print("\n Testing second step with past...")
    past_kvs = outputs[1:]  # Get past from first step
    second_input = torch.tensor([[first_token]])

    outputs2 = decoder_wrapper(
        second_input,
        *past_kvs,
        encoder_attention_mask=inputs["attention_mask"],
    )

    logits2 = outputs2[0]
    second_token = int(torch.argmax(logits2[0, -1, :]))
    print(
        f"Wrapper second token: {second_token} '{tokenizer.decode([second_token])}'")

    # Compare with direct using past
    direct_out2 = model.model.decoder(
        input_ids=second_input,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=inputs["attention_mask"],
        past_key_values=direct_out.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    direct_logits2 = model.lm_head(direct_out2.last_hidden_state)
    direct_token2 = int(torch.argmax(direct_logits2[0, -1, :]))
    print(
        f"Direct second token: {direct_token2} '{tokenizer.decode([direct_token2])}'")

    if second_token == direct_token2:
        print(" Second step matches!")
    else:
        print(f" Mismatch: expected {direct_token2}, got {second_token}")

# Export to ONNX
print("\n Exporting to ONNX...")
dummy_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

# Create dummy past with some sequence length for dynamic axes
n_heads = model.config.decoder_attention_heads
head_dim = model.config.d_model // n_heads
n_layers = model.config.decoder_layers
past_seq_len = 3

dummy_past = []
past_names = []
for i in range(n_layers):
    for kv in ['key', 'value']:
        name = f'past_{i}_self_{kv}'
        past_names.append(name)
        dummy_past.append(torch.randn(1, n_heads, past_seq_len, head_dim))

output_names = ['logits'] + [f'present_{i}_self_{kv}'
                             for i in range(n_layers)
                             for kv in ['key', 'value']]

torch.onnx.export(
    decoder_wrapper,
    args=(dummy_input_ids, *dummy_past),
    f="decoder_with_past_marian.onnx",
    input_names=['input_ids'] + past_names,
    output_names=output_names,
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch', 1: 'sequence'},
        **{f'past_{i}_self_{kv}': {0: 'batch', 2: 'past_seq'}
           for i in range(n_layers) for kv in ['key', 'value']},
        **{f'present_{i}_self_{kv}': {0: 'batch', 2: 'past_seq'}
           for i in range(n_layers) for kv in ['key', 'value']},
    },
    opset_version=17,
    do_constant_folding=False,
)
print(" Exported decoder_with_past_marian.onnx")

# ONNX inference
print("\n Loading ONNX model...")
session = ort.InferenceSession("decoder_with_past_marian.onnx")

print(f"\n ONNX Model I/O:")
print(f"Inputs: {len(session.get_inputs())}")
for inp in session.get_inputs()[:5]:
    print(f"  - {inp.name}: {inp.shape}")
print(f"  ... and {len(session.get_inputs()) - 5} more")

print(f"\n Outputs: {len(session.get_outputs())}")
for out in session.get_outputs()[:5]:
    print(f"  - {out.name}: {out.shape}")
print(f"  ... and {len(session.get_outputs()) - 5} more")

# Prepare inputs
inputs_np = tokenizer(input_sentence, return_tensors="np", padding="max_length",
                      truncation=True, max_length=64)
encoder_mask_np = inputs_np["attention_mask"].astype(np.int64)

print(f"\n ONNX decoding...")
decoder_start_id = model.config.decoder_start_token_id
decoder_input_ids = np.array([[decoder_start_id]], dtype=np.int64)
generated_ids = [decoder_start_id]

# Get actual input/output names
input_names = [i.name for i in session.get_inputs()]
output_names = [o.name for o in session.get_outputs()]

# Initialize empty past shapes to satisfy inputs (not reused for parity)
past_kvs = {}
for name in input_names:
    if name.startswith('past_'):
        past_kvs[name] = np.zeros((1, n_heads, 0, head_dim), dtype=np.float32)

print("Recomputing from full sequence each step (no cache reuse) for parity with HF")

for step in range(15):
    # Prepare inputs: full sequence + zero-length past for all layers
    ort_inputs = {'input_ids': decoder_input_ids}
    for name in input_names:
        if name.startswith('past_'):
            ort_inputs[name] = np.zeros(
                (1, n_heads, 0, head_dim), dtype=np.float32)

    # Run inference
    outputs = session.run(output_names, ort_inputs)

    # Get logits and next token
    logits = outputs[0]
    next_token = int(np.argmax(logits[0, -1, :]))
    generated_ids.append(next_token)
    decoder_input_ids = np.concatenate(
        [decoder_input_ids, [[next_token]]], axis=1)

    if step < 5:
        print(
            f"Step {step}: token={next_token} '{tokenizer.decode([next_token])}'")

    if next_token == tokenizer.eos_token_id:
        print(f"EOS at step {step}")
        break

print("\n Results:")
print(f"Expected: {hf_output_ids[0][:15].tolist()}")
print(f"ONNX:     {generated_ids[:15]}")

# Check match
expected_ids = hf_output_ids[0][:15].tolist()
actual_ids = generated_ids[:15]
match = expected_ids == actual_ids

print(f"\nMatch: {match}")
if match:
    print(" Perfect match!")
else:
    print(" Mismatch detected")
    print("\nDifferences:")
    for i, (exp, act) in enumerate(zip(expected_ids, actual_ids)):
        if exp != act:
            print(
                f"  Position {i}: expected {exp} '{tokenizer.decode([exp])}', got {act} '{tokenizer.decode([act])}'")

print("\n Decoded outputs:")
print(f"Expected: {tokenizer.decode(expected_ids, skip_special_tokens=True)}")
print(f"ONNX:     {tokenizer.decode(actual_ids, skip_special_tokens=True)}")
