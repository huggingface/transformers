import json

from benchmark_utils_generic import BenchMark, SpeedBenchMark


class FromPretrainedBenchMark(BenchMark):
    def _target(self, model_class, repo_id):
        self._buffer["target_kwargs"]["model_class"] = model_class
        self._buffer["target_kwargs"]["repo_id"] = repo_id

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
        "report_kwargs": {"output_path": "benchmark_report.json"},
    }
    result = benchmakr.run(**run_kwargs)
    print(json.dumps(result, indent=4))
