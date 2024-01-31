import os


if __name__ == "__main__":
    tests = os.getcwd()
    model_tests = os.listdir(os.path.join(tests, "models"))
    d1 = sorted(filter(os.path.isdir, os.listdir(tests)))
    d2 = sorted(filter(os.path.isdir, [f"models/{x}" for x in model_tests]))
    d1.remove("models")
    d = d2 + d1
    d = d[:256]

    num_jobs = len(d)
    num_splits = int(os.getenv("NUM_SLICES", 2))
    num_jobs_per_splits = num_jobs // num_splits

    model_splits = []
    end = 0
    for idx in range(num_splits):
        start = end
        end = start + num_jobs_per_splits + (1 if idx < num_jobs % num_splits else 0)
        model_splits.append(d[start:end])
    print(model_splits)
