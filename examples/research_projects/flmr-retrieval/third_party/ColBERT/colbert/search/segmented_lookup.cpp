#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <numeric>

typedef struct {
    int tid;
    pthread_mutex_t* mutex;
    std::queue<int>* queue;

    int64_t ndocs;
    int64_t noutputs;
    int64_t dim;

    void* input;
    int64_t* lengths;
    int64_t* offsets;
    int64_t* cumulative_lengths;

    void* output;
} lookup_args_t;

template <typename T>
void* lookup(void* args) {
    lookup_args_t* lookup_args = (lookup_args_t*)args;

    int64_t* lengths = lookup_args->lengths;
    int64_t* cumulative_lengths = lookup_args->cumulative_lengths;
    int64_t* offsets = lookup_args->offsets;
    int64_t dim = lookup_args->dim;

    T* input = static_cast<T*>(lookup_args->input);
    T* output = static_cast<T*>(lookup_args->output);

    while (1) {
        pthread_mutex_lock(lookup_args->mutex);
        if (lookup_args->queue->empty()) {
            pthread_mutex_unlock(lookup_args->mutex);
            return NULL;
        }
        int i = lookup_args->queue->front();
        lookup_args->queue->pop();
        pthread_mutex_unlock(lookup_args->mutex);

        std::memcpy(output + (cumulative_lengths[i] * dim),
                    input + (offsets[i] * dim), lengths[i] * dim * sizeof(T));
    }
}

template <typename T>
torch::Tensor segmented_lookup_impl(const torch::Tensor input,
                                    const torch::Tensor pids,
                                    const torch::Tensor lengths,
                                    const torch::Tensor offsets) {
    auto lengths_a = lengths.data_ptr<int64_t>();
    auto offsets_a = offsets.data_ptr<int64_t>();

    int64_t ndocs = pids.size(0);
    int64_t noutputs = std::accumulate(lengths_a, lengths_a + ndocs, 0);

    int nthreads = at::get_num_threads();

    int64_t dim;
    torch::Tensor output;

    if (input.dim() == 1) {
        dim = 1;
        output = torch::zeros({noutputs}, input.options());
    } else {
        assert(input.dim() == 2);
        dim = input.size(1);
        output = torch::zeros({noutputs, dim}, input.options());
    }

    int64_t cumulative_lengths[ndocs + 1];
    cumulative_lengths[0] = 0;
    std::partial_sum(lengths_a, lengths_a + ndocs, cumulative_lengths + 1);

    pthread_mutex_t mutex;
    int rc = pthread_mutex_init(&mutex, NULL);
    if (rc) {
        fprintf(stderr, "Unable to init mutex: %d\n", rc);
    }

    std::queue<int> queue;
    for (int i = 0; i < ndocs; i++) {
        queue.push(i);
    }

    pthread_t threads[nthreads];
    lookup_args_t args[nthreads];
    for (int i = 0; i < nthreads; i++) {
        args[i].tid = i;
        args[i].mutex = &mutex;
        args[i].queue = &queue;

        args[i].ndocs = ndocs;
        args[i].noutputs = noutputs;
        args[i].dim = dim;

        args[i].input = (void*)input.data_ptr<T>();
        args[i].lengths = lengths_a;
        args[i].offsets = offsets_a;
        args[i].cumulative_lengths = cumulative_lengths;

        args[i].output = (void*)output.data_ptr<T>();

        rc = pthread_create(&threads[i], NULL, lookup<T>, (void*)&args[i]);
        if (rc) {
            fprintf(stderr, "Unable to create thread %d: %d\n", i, rc);
        }
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
    }

    rc = pthread_mutex_destroy(&mutex);
    if (rc) {
        fprintf(stderr, "Unable to destroy mutex: %d\n", rc);
    }

    return output;
}

torch::Tensor segmented_lookup(const torch::Tensor input,
                               const torch::Tensor pids,
                               const torch::Tensor lengths,
                               const torch::Tensor offsets) {
    if (input.dtype() == torch::kUInt8) {
        return segmented_lookup_impl<uint8_t>(input, pids, lengths, offsets);
    } else if (input.dtype() == torch::kInt32) {
        return segmented_lookup_impl<int>(input, pids, lengths, offsets);
    } else if (input.dtype() == torch::kInt64) {
        return segmented_lookup_impl<int64_t>(input, pids, lengths, offsets);
    } else if (input.dtype() == torch::kFloat32) {
        return segmented_lookup_impl<float>(input, pids, lengths, offsets);
    } else if (input.dtype() == torch::kFloat16) {
        return segmented_lookup_impl<at::Half>(input, pids, lengths, offsets);
    } else {
        assert(false);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segmented_lookup_cpp", &segmented_lookup, "Segmented lookup");
}
