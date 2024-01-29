import os


if __name__ == "__main__":
    tests = os.getcwd()
    model_tests = os.listdir(os.path.join(tests, "models"))
    d1 = sorted(list(filter(os.path.isdir, os.listdir(tests))))
    d2 = sorted(list(filter(os.path.isdir, [f"models/{x}" for x in model_tests])))
    d1.remove("models")
    d = d2[1:3] + d1[:1]

    num_jobs = len(d)
    num_splits = int(os.getenv("NUM_SLICES", 2))
    num_jobs_per_splits = num_jobs // num_splits + int(num_jobs % num_splits > 0)

    model_splits = []
    for idx in range(num_splits):
        model_splits.append(d[(num_jobs_per_splits * idx):(num_jobs_per_splits * (idx + 1))])
    print(model_splits)
