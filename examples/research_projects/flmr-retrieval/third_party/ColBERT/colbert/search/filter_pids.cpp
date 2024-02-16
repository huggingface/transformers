#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <utility>

typedef struct maxsim_args {
    int tid;
    int nthreads;

    int ncentroids;
    int nquery_vectors;
    int npids;

    int* pids;
    float* centroid_scores;
    int* codes;
    int64_t* doclens;
    int64_t* offsets;
    bool* idx;

    std::priority_queue<std::pair<float, int>> approx_scores;
} maxsim_args_t;

void* maxsim(void* args) {
    maxsim_args_t* maxsim_args = (maxsim_args_t*)args;

    float per_doc_approx_scores[maxsim_args->nquery_vectors];
    for (int k = 0; k < maxsim_args->nquery_vectors; k++) {
        per_doc_approx_scores[k] = -9999;
    }

    int ndocs_per_thread =
        (int)std::ceil(((float)maxsim_args->npids) / maxsim_args->nthreads);
    int start = maxsim_args->tid * ndocs_per_thread;
    int end =
        std::min((maxsim_args->tid + 1) * ndocs_per_thread, maxsim_args->npids);

    std::unordered_set<int> seen_codes;

    for (int i = start; i < end; i++) {
        auto pid = maxsim_args->pids[i];
        for (int j = 0; j < maxsim_args->doclens[pid]; j++) {
            auto code = maxsim_args->codes[maxsim_args->offsets[pid] + j];
            assert(code < maxsim_args->ncentroids);
            if (maxsim_args->idx[code] &&
                seen_codes.find(code) == seen_codes.end()) {
                for (int k = 0; k < maxsim_args->nquery_vectors; k++) {
                    per_doc_approx_scores[k] =
                        std::max(per_doc_approx_scores[k],
                                 maxsim_args->centroid_scores
                                     [code * maxsim_args->nquery_vectors + k]);
                }
                seen_codes.insert(code);
            }
        }
        float score = 0;
        for (int k = 0; k < maxsim_args->nquery_vectors; k++) {
            score += per_doc_approx_scores[k];
            per_doc_approx_scores[k] = -9999;
        }
        maxsim_args->approx_scores.push(std::make_pair(score, pid));
        seen_codes.clear();
    }

    return NULL;
}

void filter_pids_helper(int ncentroids, int nquery_vectors, int npids,
                        int* pids, float* centroid_scores, int* codes,
                        int64_t* doclens, int64_t* offsets, bool* idx,
                        int nfiltered_docs, int* filtered_pids) {
    auto nthreads = at::get_num_threads();

    pthread_t threads[nthreads];
    maxsim_args_t args[nthreads];

    for (int i = 0; i < nthreads; i++) {
        args[i].tid = i;
        args[i].nthreads = nthreads;

        args[i].ncentroids = ncentroids;
        args[i].nquery_vectors = nquery_vectors;
        args[i].npids = npids;

        args[i].pids = pids;
        args[i].centroid_scores = centroid_scores;
        args[i].codes = codes;
        args[i].doclens = doclens;
        args[i].offsets = offsets;
        args[i].idx = idx;

        args[i].approx_scores = std::priority_queue<std::pair<float, int>>();

        int rc = pthread_create(&threads[i], NULL, maxsim, (void*)&args[i]);
        if (rc) {
            fprintf(stderr, "Unable to create thread %d: %d\n", i, rc);
            std::exit(1);
        }
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
    }

    std::priority_queue<std::pair<float, int>> global_approx_scores;
    for (int i = 0; i < nthreads; i++) {
        for (int j = 0; j < nfiltered_docs; j++) {
            if (args[i].approx_scores.empty()) {
                break;
            }
            global_approx_scores.push(args[i].approx_scores.top());
            args[i].approx_scores.pop();
        }
    }

    for (int i = 0; i < nfiltered_docs; i++) {
        std::pair<float, int> score_and_pid = global_approx_scores.top();
        filtered_pids[i] = score_and_pid.second;
        global_approx_scores.pop();
    }
}

torch::Tensor filter_pids(const torch::Tensor pids,
                          const torch::Tensor centroid_scores,
                          const torch::Tensor codes,
                          const torch::Tensor doclens,
                          const torch::Tensor offsets, const torch::Tensor idx,
                          int nfiltered_docs) {
    auto ncentroids = centroid_scores.size(0);
    auto nquery_vectors = centroid_scores.size(1);
    auto npids = pids.size(0);

    auto pids_a = pids.data_ptr<int>();
    auto centroid_scores_a = centroid_scores.data_ptr<float>();
    auto codes_a = codes.data_ptr<int>();
    auto doclens_a = doclens.data_ptr<int64_t>();
    auto offsets_a = offsets.data_ptr<int64_t>();
    auto idx_a = idx.data_ptr<bool>();

    int filtered_pids[nfiltered_docs];
    filter_pids_helper(ncentroids, nquery_vectors, npids, pids_a,
                       centroid_scores_a, codes_a, doclens_a, offsets_a, idx_a,
                       nfiltered_docs, filtered_pids);

    int nfinal_filtered_docs = (int)(nfiltered_docs / 4);
    int final_filtered_pids[nfinal_filtered_docs];
    bool ones[ncentroids];
    for (int i = 0; i < ncentroids; i++) {
        ones[i] = true;
    }
    filter_pids_helper(ncentroids, nquery_vectors, nfiltered_docs,
                       filtered_pids, centroid_scores_a, codes_a, doclens_a,
                       offsets_a, ones, nfinal_filtered_docs,
                       final_filtered_pids);

    auto options =
        torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);
    return torch::from_blob(final_filtered_pids, {nfinal_filtered_docs},
                            options)
        .clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_pids_cpp", &filter_pids, "Filter pids");
}

