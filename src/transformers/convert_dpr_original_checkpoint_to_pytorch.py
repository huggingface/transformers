import argparse
import collections
from pathlib import Path

import torch
from torch.serialization import default_restore_location

from transformers import BertConfig, DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRReader


CheckpointState = collections.namedtuple(
    "CheckpointState", ["model_dict", "optimizer_dict", "scheduler_dict", "offset", "epoch", "encoder_params"]
)


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    print("Reading saved model from %s", model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    return CheckpointState(**state_dict)


class DPRState:
    def __init__(self, src_file: Path):
        self.src_file = src_file

    def load_dpr_model(self):
        raise NotImplementedError

    @staticmethod
    def from_type(comp_type: str, *args, **kwargs) -> "DPRState":
        if comp_type.startswith("c"):
            return DPRContextEncoderState(*args, **kwargs)
        if comp_type.startswith("q"):
            return DPRQuestionEncoderState(*args, **kwargs)
        if comp_type.startswith("r"):
            return DPRReaderState(*args, **kwargs)
        else:
            raise ValueError("Component type must be either 'ctx_encoder', 'question_encoder' or 'reader'.")


class DPRContextEncoderState(DPRState):
    def load_dpr_model(self):
        model = DPRContextEncoder(DPRConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
        print("Loading DPR biencoder from {}".format(self.src_file))
        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = model.ctx_encoder, "ctx_model."
        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        state_dict = {"bert_model.embeddings.position_ids": model.ctx_encoder.bert_model.embeddings.position_ids}
        for key, value in saved_state.model_dict.items():
            if key.startswith(prefix):
                key = key[len(prefix) :]
                if not key.startswith("encode_proj."):
                    key = "bert_model." + key
                state_dict[key] = value
        encoder.load_state_dict(state_dict)
        return model


class DPRQuestionEncoderState(DPRState):
    def load_dpr_model(self):
        model = DPRQuestionEncoder(DPRConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
        print("Loading DPR biencoder from {}".format(self.src_file))
        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = model.question_encoder, "question_model."
        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        state_dict = {"bert_model.embeddings.position_ids": model.question_encoder.bert_model.embeddings.position_ids}
        for key, value in saved_state.model_dict.items():
            if key.startswith(prefix):
                key = key[len(prefix) :]
                if not key.startswith("encode_proj."):
                    key = "bert_model." + key
                state_dict[key] = value
        encoder.load_state_dict(state_dict)
        return model


class DPRReaderState(DPRState):
    def load_dpr_model(self):
        model = DPRReader(DPRConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
        print("Loading DPR reader from {}".format(self.src_file))
        saved_state = load_states_from_checkpoint(self.src_file)
        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        state_dict = {
            "encoder.bert_model.embeddings.position_ids": model.span_predictor.encoder.bert_model.embeddings.position_ids
        }
        for key, value in saved_state.model_dict.items():
            if key.startswith("encoder.") and not key.startswith("encoder.encode_proj"):
                key = "encoder.bert_model." + key[len("encoder.") :]
            state_dict[key] = value
        model.span_predictor.load_state_dict(state_dict)
        return model


def convert(comp_type: str, src_file: Path, dest_dir: Path):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    dpr_state = DPRState.from_type(comp_type, src_file=src_file)
    model = dpr_state.load_dpr_model()
    model.save_pretrained(dest_dir)
    model.from_pretrained(dest_dir)  # sanity check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--type", type=str, help="Type of the component to convert: 'ctx_encoder', 'question_encoder' or 'reader'."
    )
    parser.add_argument(
        "--src",
        type=str,
        help="Path to the dpr checkpoint file. They can be downloaded from the official DPR repo https://github.com/facebookresearch/DPR. Note that in the official repo, both encoders are stored in the 'retriever' checkpoints.",
    )
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model directory.")
    args = parser.parse_args()

    src_file = Path(args.src)
    dest_dir = f"converted-{src_file.name}" if args.dest is None else args.dest
    dest_dir = Path(dest_dir)
    assert src_file.exists()
    assert (
        args.type is not None
    ), "Please specify the component type of the DPR model to convert: 'ctx_encoder', 'question_encoder' or 'reader'."
    convert(args.type, src_file, dest_dir)
