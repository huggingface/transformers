import json

from benchmark_utils_generic import BenchMark, SpeedBenchMark


class FromPretrainedBenchMark(BenchMark):

    def target(self, model_class, repo_id):

        def target():
            _ = model_class.from_pretrained(repo_id)

        return target


class FromPretrainedSpeedBenchMark(SpeedBenchMark, FromPretrainedBenchMark):
    pass


if __name__ == "__main__":

    from transformers import AutoModel
    repo_id = "bert-base-uncased"

    benchmakr = FromPretrainedSpeedBenchMark()

    run_kwargs = {
        "measure_kwargs": {"number": 2, "repeat": 3},
        "target_kwargs": {"model_class": AutoModel, "repo_id": repo_id},
        "inputs_kwargs": [{}],
        "report_kwargs": {},
        # "report_kwargs": {"output_path": None, "keys_to_keep": None}
    }
    result = benchmakr.run(**run_kwargs)
    print(json.dumps(result, indent=4))

    run_kwargs["report_kwargs"]["only_result"] = True
    result = benchmakr.run(**run_kwargs)
    print(json.dumps(result, indent=4))
