import json
import tempfile

from transformers.distributed import DistributedConfig


class TestDistributedConfig:
    def test_2d_parallelism(self):
        dc = DistributedConfig(tp_size=2, tp_plan="auto", fsdp_size=2, fsdp_plan="auto")
        assert dc.tp_size == 2
        assert dc.fsdp_size == 2
        assert dc.tp_plan == "auto"
        assert dc.fsdp_plan == "auto"

    def test_tp_only_defaults_fsdp_to_1(self):
        dc = DistributedConfig(tp_size=4)
        assert dc.tp_size == 4
        assert dc.fsdp_size == 1
        assert dc.tp_plan == "auto"  # size given → plan defaults to "auto"

    def test_fsdp_only_defaults_tp_to_1(self):
        dc = DistributedConfig(fsdp_size=4)
        assert dc.tp_size == 1
        assert dc.fsdp_size == 4
        assert dc.fsdp_plan == "auto"  # size given → plan defaults to "auto"
        assert dc.tp_plan == "auto"  # tp_size got set to 1 → plan also defaults

    def test_empty_config(self):
        dc = DistributedConfig()
        assert dc.tp_size is None
        assert dc.fsdp_size is None
        assert dc.tp_plan is None
        assert dc.fsdp_plan is None

    def test_from_dict(self):
        dc = DistributedConfig.from_dict({"tp_size": 2, "fsdp_size": 4, "tp_plan": "auto"})
        assert dc.tp_size == 2
        assert dc.fsdp_size == 4
        assert dc.tp_plan == "auto"

    def test_from_dict_ignores_unknown_keys(self):
        dc = DistributedConfig.from_dict({"tp_size": 2, "unknown_key": 42})
        assert dc.tp_size == 2
        assert not hasattr(dc, "unknown_key")

    def test_from_dict_kwargs_override(self):
        dc = DistributedConfig.from_dict({"tp_size": 2}, tp_size=8)
        assert dc.tp_size == 8

    def test_to_dict(self):
        dc = DistributedConfig(tp_size=2, fsdp_size=4)
        d = dc.to_dict()
        assert d == {
            "tp_size": 2,
            "tp_plan": "auto",
            "enable_sequence_parallel": False,
            "fsdp_size": 4,
            "fsdp_plan": "auto",
        }

    def test_to_dict_is_a_copy(self):
        dc = DistributedConfig(tp_plan={"layer": "colwise"})
        d = dc.to_dict()
        d["tp_plan"]["layer"] = "rowwise"
        assert dc.tp_plan["layer"] == "colwise"

    def test_to_json_string(self):
        dc = DistributedConfig(tp_size=2, fsdp_size=2)
        s = dc.to_json_string()
        parsed = json.loads(s)
        assert parsed["tp_size"] == 2
        assert parsed["fsdp_size"] == 2

    def test_to_json_file(self):
        dc = DistributedConfig(tp_size=4, tp_plan="auto")
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f:
            dc.to_json_file(f.name)
            f.seek(0)
            parsed = json.load(f)
        assert parsed["tp_size"] == 4
        assert parsed["tp_plan"] == "auto"

    def test_roundtrip_dict(self):
        original = DistributedConfig(tp_size=2, tp_plan="auto", fsdp_size=4, fsdp_plan="auto")
        restored = DistributedConfig.from_dict(original.to_dict())
        assert original == restored

    def test_repr(self):
        dc = DistributedConfig(tp_size=2)
        r = repr(dc)
        assert "DistributedConfig" in r
        assert '"tp_size": 2' in r
