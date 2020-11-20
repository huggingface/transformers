import torch

def split_heads(*tensors, num_heads):
    for tensor in tensors:
        old_shape = tensor.shape
        total_dims = old_shape[-1]
        
        assert total_dims % num_heads == 0 # Total dimensionality must be divisible by the number of heads
        head_dims = total_dims // num_heads
        
        # Split the last dimension in two
        new_shape = list(old_shape[:-1]) + [num_heads, head_dims]
        tensor = tensor.reshape(*new_shape)
        
        # Move the heads dimension right after the batch size, so it gets treated as a batch dimension during matmuls
        dimensions = [i for i in range(len(new_shape))]
        penultimate = dimensions[-2]
        dimensions[-2] = dimensions[-3]
        dimensions[-3] = penultimate
        
        yield tensor.permute(*dimensions)

def combine_heads(tensor):
    rank = len(tensor.shape)
    dimensions = [i for i in range(rank)]
    
    # Move the heads dimension to the second to last place
    num_heads = dimensions[-3]
    dimensions[-3] = dimensions[-2]
    dimensions[-2] = num_heads
    tensor = tensor.permute(*dimensions)
    
    new_shape = list(tensor.shape)
    new_shape[-2] *= new_shape[-1] # Multiply the number of heads by the dimensionality of each head
    new_shape = new_shape[:-1]
    
    return tensor.reshape(new_shape)

def gaussian_orthogonal_matrix(num_rows, num_cols, regularize=True, output_device='cuda'):
    def get_square_block(size):
        unstructured_block = torch.randn(size, size, device='cpu')
        q, r = torch.qr(unstructured_block, some = True)
        return q.t()
    
    num_full_blocks = num_rows // num_cols
    block_list = [get_square_block(num_cols) for _ in range(num_full_blocks)]

    remaining_rows = num_rows - num_full_blocks * num_cols
    if remaining_rows > 0:
        q = get_square_block(num_cols)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)
    
    # This option yields SMREG
    if regularize:
        final_matrix *= num_cols ** 0.5
    else:
        multiplier = torch.randn(num_rows, num_cols, device='cpu').norm(dim = 1)
        final_matrix = torch.diag(multiplier) @ final_matrix
    
    return final_matrix.to(output_device)

def exp_kernel(x, h, stabilizer):
    return torch.exp(h + x - stabilizer)

def cosh_kernel(x, h, stabilizer):
    f_1 = torch.exp(h + x - stabilizer)
    f_2 = torch.exp(h - x - stabilizer)
    return torch.cat((f_1, f_2), dim=-1)

def relu_kernel(x, h, stabilizer, epsilon=1e-9):
    return x * (x > 0) + epsilon # Adding epsilon keeps us from dividing by zero when we normalize

class FavorAttention(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.projection_matrix = torch.nn.Parameter(requires_grad=kwargs.get('trainable_features', False))
        self.params = kwargs
    
    def forward(self, q, k, v):
        kernel_fn = self.params.get('kernel_fn', exp_kernel)
        total_dims = q.shape[-1]
        
        # Split into heads
        num_heads = self.params['num_heads']
        q, k, v = split_heads(q, k, v, num_heads=num_heads)
        
        # Instead of dividing the product QK^T by sqrt(d), we divide Q and K separately by the 4th root of d
        q = q / (total_dims ** 0.25)
        k = k / (total_dims ** 0.25)
        
        # If the sequence length is less than 1.5 times as long as the number of random features we would need to accurately approximate
        # the softmax attention, just bail on the entire algorithm and calculate the real attention as we normally would. This prevents us
        # from pointlessly using up more time and/or memory than the vanilla attention mechanism when sequences are short.
        L, dims = q.shape[-2:]
        m = self.params.get('num_features', round(dims * np.log(dims))) # m is the number of random features
        if 1.5 * m > L:
            attn_map = kernel_fn(q @ k.transpose(-2, -1))
            attn_map /= attn_map.sum(dim=-1, keepdim=True) # Renormalize
            return combine_heads(attn_map @ v)
        
        # Lazily generate the random feature matrix
        if self.params.get('redraw_features', False) or self.projection_matrix.shape[-1] != dims:
            if self.params.get('use_orthog_features', True):
                self.projection_matrix.data = gaussian_orthogonal_matrix(m, dims, regularize=self.params.get('regularize_features', True))
            else:
                self.projection_matrix.data = torch.randn(m, dims, device='cuda')
        
        self.projection_matrix.expand(q.shape[0], num_heads, m, dims) # Broadcast the feature matrix across the batch dimension
        W_t = self.projection_matrix.data.transpose(-2, -1)
        epsilon = self.params.get('epsilon', 1e-4)
        
        # By multiplying Q' and K' by 1/sqrt(m), we ensure the final matrix product will contain a factor of 1/m. This means
        # that each row of Q'K'^T can be interpreted as an average over the exp(omega^T * q) * exp(omega^T * k) terms.
        def phi(x, is_query):
            # The h(x) function, defined in Lemma 1 in Choromanski et al. pg. 4
            h_of_x = -torch.sum(x ** 2, dim=-1, keepdim=True) / 2
            
            projected_x = x @ W_t
            stabilizer = torch.max(h_of_x) if not is_query else torch.max(h_of_x, axis=-1, keepdim=True).values
            kernel_output = kernel_fn(projected_x, h_of_x, stabilizer)
            
            return (kernel_output.shape[-1] ** -0.5) * (kernel_output + epsilon)
        
        q_prime, k_prime = phi(q, True), phi(k, False)
        
        if self.params.get('causal', False):
            from CausalDotProduct import CausalDotProduct
            return CausalDotProduct.apply(q_prime, k_prime, v)
        else:
            # Equivalent to multiplying K'^T by a ones vector
            d = q_prime @ k_prime.sum(dim=-2).unsqueeze(-1)
            d += 2 * epsilon * (torch.abs(d) <= epsilon) # Avoid dividing by very small numbers
            
            k_prime_t = k_prime.transpose(-2, -1)
            result = q_prime @ (k_prime_t @ v) / d
        
        return combine_heads(result)
