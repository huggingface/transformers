#include <pthread.h>
#include <torch/extension.h>

typedef struct decompress_args {
    int tid;
    int nthreads;

    int npids;
    int dim;
    int packed_dim;
    int npacked_vals_per_byte;

    int* pids;
    int64_t* lengths;
    int64_t* offsets;
    float* bucket_weights;
    uint8_t* reversed_bit_map;
    uint8_t* bucket_weight_combinations;
    uint8_t* binary_residuals;
    int* codes;
    float* centroids;
    int64_t* cumulative_lengths;

    float* output;
} decompress_args_t;

void* decompress(void* args) {
    decompress_args_t* decompress_args = (decompress_args_t*)args;

    int npids_per_thread = (int)std::ceil(((float)decompress_args->npids) /
                                          decompress_args->nthreads);
    int start = decompress_args->tid * npids_per_thread;
    int end = std::min((decompress_args->tid + 1) * npids_per_thread,
                       decompress_args->npids);

    // Iterate over all documents
    for (int i = start; i < end; i++) {
        int pid = decompress_args->pids[i];

        // Offset into packed list of token vectors for the given document
        int64_t offset = decompress_args->offsets[pid];

        // For each document, iterate over all token vectors
        for (int j = 0; j < decompress_args->lengths[pid]; j++) {
            const int code = decompress_args->codes[offset + j];

            // For each token vector, iterate over the packed (8-bit) residual
            // values
            for (int k = 0; k < decompress_args->packed_dim; k++) {
                uint8_t x =
                    decompress_args->binary_residuals
                        [(offset + j) * decompress_args->packed_dim + k];
                x = decompress_args->reversed_bit_map[x];

                // For each packed residual value, iterate over the bucket
                // weight indices. If we use n-bit compression, that means there
                // will be (8 / n) indices per packed value.
                for (int l = 0; l < decompress_args->npacked_vals_per_byte;
                     l++) {
                    const int output_dim_idx =
                        k * decompress_args->npacked_vals_per_byte + l;
                    const int bucket_weight_idx =
                        decompress_args->bucket_weight_combinations
                            [x * decompress_args->npacked_vals_per_byte + l];
                    decompress_args
                        ->output[(decompress_args->cumulative_lengths[i] + j) *
                                     decompress_args->dim +
                                 output_dim_idx] =
                        decompress_args->bucket_weights[bucket_weight_idx] +
                        decompress_args->centroids[code * decompress_args->dim +
                                                   output_dim_idx];
                }
            }
        }
    }

    return NULL;
}

torch::Tensor decompress_residuals(
    const torch::Tensor pids, const torch::Tensor lengths,
    const torch::Tensor offsets, const torch::Tensor bucket_weights,
    const torch::Tensor reversed_bit_map,
    const torch::Tensor bucket_weight_combinations,
    const torch::Tensor binary_residuals, const torch::Tensor codes,
    const torch::Tensor centroids, const int dim, const int nbits) {
    const int npacked_vals_per_byte = (8 / nbits);
    const int packed_dim = (int)(dim / npacked_vals_per_byte);

    int npids = pids.size(0);
    int* pids_a = pids.data_ptr<int>();
    int64_t* lengths_a = lengths.data_ptr<int64_t>();
    int64_t* offsets_a = offsets.data_ptr<int64_t>();
    float* bucket_weights_a = bucket_weights.data_ptr<float>();
    uint8_t* reversed_bit_map_a = reversed_bit_map.data_ptr<uint8_t>();
    uint8_t* bucket_weight_combinations_a =
        bucket_weight_combinations.data_ptr<uint8_t>();
    uint8_t* binary_residuals_a = binary_residuals.data_ptr<uint8_t>();
    int* codes_a = codes.data_ptr<int>();
    float* centroids_a = centroids.data_ptr<float>();

    int64_t cumulative_lengths[npids + 1];
    int noutputs = 0;
    cumulative_lengths[0] = 0;
    for (int i = 0; i < npids; i++) {
        noutputs += lengths_a[pids_a[i]];
        cumulative_lengths[i + 1] =
            cumulative_lengths[i] + lengths_a[pids_a[i]];
    }

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor output = torch::zeros({noutputs, dim}, options);
    float* output_a = output.data_ptr<float>();

    auto nthreads = at::get_num_threads();

    pthread_t threads[nthreads];
    decompress_args_t args[nthreads];

    for (int i = 0; i < nthreads; i++) {
        args[i].tid = i;
        args[i].nthreads = nthreads;

        args[i].npids = npids;
        args[i].dim = dim;
        args[i].packed_dim = packed_dim;
        args[i].npacked_vals_per_byte = npacked_vals_per_byte;

        args[i].pids = pids_a;
        args[i].lengths = lengths_a;
        args[i].offsets = offsets_a;
        args[i].bucket_weights = bucket_weights_a;
        args[i].reversed_bit_map = reversed_bit_map_a;
        args[i].bucket_weight_combinations = bucket_weight_combinations_a;
        args[i].binary_residuals = binary_residuals_a;
        args[i].codes = codes_a;
        args[i].centroids = centroids_a;
        args[i].cumulative_lengths = cumulative_lengths;

        args[i].output = output_a;

        int rc = pthread_create(&threads[i], NULL, decompress, (void*)&args[i]);
        if (rc) {
            fprintf(stderr, "Unable to create thread %d: %d\n", i, rc);
            std::exit(1);
        }
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decompress_residuals_cpp", &decompress_residuals,
          "Decompress residuals");
}
