#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <numeric>

typedef struct {
    int tid;
    int nthreads;

    int ndocs;
    int ndoc_vectors;
    int nquery_vectors;

    int64_t* lengths;
    float* scores;
    int64_t* offsets;

    float* max_scores;
} max_args_t;

void* max(void* args) {
    max_args_t* max_args = (max_args_t*)args;

    int ndocs_per_thread =
        std::ceil(((float)max_args->ndocs) / max_args->nthreads);
    int start = max_args->tid * ndocs_per_thread;
    int end = std::min((max_args->tid + 1) * ndocs_per_thread, max_args->ndocs);

    auto max_scores_offset =
        max_args->max_scores + (start * max_args->nquery_vectors);
    auto scores_offset =
        max_args->scores + (max_args->offsets[start] * max_args->nquery_vectors);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < max_args->lengths[i]; j++) {
            std::transform(max_scores_offset,
                           max_scores_offset + max_args->nquery_vectors,
                           scores_offset, max_scores_offset,
                           [](float a, float b) { return std::max(a, b); });
            scores_offset += max_args->nquery_vectors;
        }
        max_scores_offset += max_args->nquery_vectors;
    }

    return NULL;
}

torch::Tensor segmented_maxsim(const torch::Tensor scores,
                               const torch::Tensor lengths) {
    auto lengths_a = lengths.data_ptr<int64_t>();
    auto scores_a = scores.data_ptr<float>();
    auto ndocs = lengths.size(0);
    auto ndoc_vectors = scores.size(0);
    auto nquery_vectors = scores.size(1);
    auto nthreads = at::get_num_threads();

    torch::Tensor max_scores =
        torch::zeros({ndocs, nquery_vectors}, scores.options());

    int64_t offsets[ndocs + 1];
    offsets[0] = 0;
    std::partial_sum(lengths_a, lengths_a + ndocs, offsets + 1);

    pthread_t threads[nthreads];
    max_args_t args[nthreads];

    for (int i = 0; i < nthreads; i++) {
        args[i].tid = i;
        args[i].nthreads = nthreads;

        args[i].ndocs = ndocs;
        args[i].ndoc_vectors = ndoc_vectors;
        args[i].nquery_vectors = nquery_vectors;

        args[i].lengths = lengths_a;
        args[i].scores = scores_a;
        args[i].offsets = offsets;

        args[i].max_scores = max_scores.data_ptr<float>();

        int rc = pthread_create(&threads[i], NULL, max, (void*)&args[i]);
        if (rc) {
            fprintf(stderr, "Unable to create thread %d: %d\n", i, rc);
        }
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
    }

    return max_scores.sum(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segmented_maxsim_cpp", &segmented_maxsim, "Segmented MaxSim");
}
