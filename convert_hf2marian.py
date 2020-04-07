import numpy as np
import yaml
import argparse

from transformers import BertModel
from durbango import pickle_save

parser = argparse.ArgumentParser(description='Convert Huggingface Bert model to Marian weight file.')
parser.add_argument('--bert', default='bert-base-cased', help='Path to Huggingface Bert PyTorch model')
parser.add_argument('--marian', default='marian_output.npz',
                    help='Output path for Marian weight file')
args = parser.parse_args()

huggingface = BertModel.from_pretrained(args.bert)
huggingface.eval()



print(huggingface.config)

config = dict()
config["type"] = "bert-classifier"
config["input-types"] = ["sequence"]
config["tied-embeddings-all"] = True
config["tied-embeddings-src"] = False

config["transformer-ffn-depth"] = 2
config["transformer-train-position-embeddings"] = True
config["transformer-preprocess"] = ""
config["transformer-postprocess"] = "dan"
config["transformer-postprocess-emb"] = "nd"
config["bert-train-type-embeddings"] = False
config["version"] = "huggingface2marian.py conversion"

config["enc-depth"] = 0
config["transformer-dim-ffn"] = huggingface.config.intermediate_size
config["transformer-heads"] = huggingface.config.num_attention_heads
config["transformer-ffn-activation"] = huggingface.config.hidden_act

config["bert-sep-symbol"] = "</s>"
config["bert-class-symbol"] = "</s>"

marianModel = dict()


def transposeOrder(mat):
    matT = np.transpose(mat)  # just a view with changed row order
    return matT.flatten(order="C").reshape(matT.shape)  # force row order change and reshape


def reverse_transposeOrder(mat):
    raise NotImplementedError()


ELAYER_CONVERT = {'self_Wq': 'attention.self.query.weight',
 'self_Wk': 'attention.self.key.weight',
 'self_Wv': 'attention.self.value.weight',
 'self_bq': 'attention.self.query.bias',
 'self_bk': 'attention.self.key.bias',
 'self_bv': 'attention.self.value.bias',
 'self_Wo': 'attention.output.dense.weight',
 'self_bo': 'attention.output.dense.bias',
 'self_Wo_ln_scale': 'attention.output.LayerNorm.weight',
 'self_Wo_ln_bias': 'attention.output.LayerNorm.bias',
 'ffn_W1': 'intermediate.dense.weight',
 'ffn_b1': 'intermediate.dense.bias',
 'ffn_W2': 'output.dense.weight',
 'ffn_b2': 'output.dense.bias',
 'ffn_ffn_ln_scale': 'output.LayerNorm.weight',
 'ffn_ffn_ln_bias': 'output.LayerNorm.bias'}


DLAYER_CONVERT = ELAYER_CONVERT.copy()



ALL_CONVERT_CALLS = []
def convert(pd, srcs, trg, transpose=True, bias=False):
    ALL_CONVERT_CALLS.append((srcs, trg, transpose, bias))
    if len(srcs) == 1:
        for src in srcs:
            num = pd[src].detach().numpy()
            if bias:
                marianModel[trg] = np.atleast_2d(num)
            else:
                if transpose:
                    marianModel[trg] = transposeOrder(num)  # transpose with row order change
                else:
                    marianModel[trg] = num
    else:  # path that joins matrices together for fused self-attention
        nums = [pd[src].detach().numpy() for src in srcs]
        if bias:
            nums = [np.transpose(np.atleast_2d(num)) for num in nums]
        marianModel[trg] = np.stack(nums, axis=0)

# def inject(layer, nth, level):

def extract(layer, nth, level):
    name = type(layer).__name__
    print("  " * level, nth, name)
    if name == "BertLayer":
        pd = dict(layer.named_parameters())
        for n in pd:
            print("  " * (level + 1), n, pd[n].shape)

        convert(pd, ["attention.self.query.weight"], f"encoder_l{nth + 1}_self_Wq", transpose=True)
        convert(pd, ["attention.self.key.weight"], f"encoder_l{nth + 1}_self_Wk")
        convert(pd, ["attention.self.value.weight"], f"encoder_l{nth + 1}_self_Wv")

        convert(pd, ["attention.self.query.bias"], f"encoder_l{nth + 1}_self_bq", bias=True)
        convert(pd, ["attention.self.key.bias"], f"encoder_l{nth + 1}_self_bk", bias=True)
        convert(pd, ["attention.self.value.bias"], f"encoder_l{nth + 1}_self_bv", bias=True)

        convert(pd, ["attention.output.dense.weight"], f"encoder_l{nth + 1}_self_Wo")
        convert(pd, ["attention.output.dense.bias"], f"encoder_l{nth + 1}_self_bo", bias=True)

        convert(pd, ["attention.output.LayerNorm.weight"], f"encoder_l{nth + 1}_self_Wo_ln_scale", bias=True)
        convert(pd, ["attention.output.LayerNorm.bias"], f"encoder_l{nth + 1}_self_Wo_ln_bias", bias=True)

        convert(pd, ["intermediate.dense.weight"], f"encoder_l{nth + 1}_ffn_W1")
        convert(pd, ["intermediate.dense.bias"], f"encoder_l{nth + 1}_ffn_b1", bias=True)
        convert(pd, ["output.dense.weight"], f"encoder_l{nth + 1}_ffn_W2")
        convert(pd, ["output.dense.bias"], f"encoder_l{nth + 1}_ffn_b2", bias=True)

        convert(pd, ["output.LayerNorm.weight"], f"encoder_l{nth + 1}_ffn_ffn_ln_scale", bias=True)
        convert(pd, ["output.LayerNorm.bias"], f"encoder_l{nth + 1}_ffn_ffn_ln_bias", bias=True)

        config["enc-depth"] += 1

    elif name == "BertEmbeddings":
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
        pd = dict(layer.named_parameters())
        convert(pd, ["word_embeddings.weight"], f"Wemb", transpose=False)
        convert(pd, ["position_embeddings.weight"], f"Wpos", transpose=False)

        config["bert-type-vocab-size"] = 0
        if hasattr(layer, "token_type_embeddings"):
            convert(pd, ["token_type_embeddings.weight"], f"Wtype", transpose=False)
            config["bert-type-vocab-size"] = pd["token_type_embeddings.weight"].shape[0]
            config["bert-train-type-embeddings"] = True

        convert(pd, ["LayerNorm.weight"], f"encoder_emb_ln_scale_pre", bias=True)
        convert(pd, ["LayerNorm.bias"], f"encoder_emb_ln_bias_pre", bias=True)

        config["dim-emb"] = pd["word_embeddings.weight"].shape[1]
        config["dim-vocabs"] = [pd["word_embeddings.weight"].shape[0]]
        config["max-length"] = pd["position_embeddings.weight"].shape[0]

    elif name == "BertPooler":
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)

        pd = dict(layer.named_parameters())
        convert(pd, ["dense.weight"], "classifier_ff_logit_l1_W")
        convert(pd, ["dense.bias"], "classifier_ff_logit_l1_b", bias=True)

    else:
        recurse(layer, level + 1)


def recurse(parent, level=0):
    for i, child in enumerate(parent.children()):
        extract(child, i, level)


recurse(huggingface)

for m in marianModel:
    print(m, marianModel[m].shape)

configYamlStr = yaml.dump(config, default_flow_style=False)
#pickle_save(configYamlStr, config)
desc = list(configYamlStr)
npDesc = np.chararray((len(desc),))
npDesc[:] = desc
import ipdb; ipdb.set_trace()
npDesc.dtype = np.int8

marianModel["special:model.yml"] = npDesc

print("\nMarian config:")
print(configYamlStr)
print("Saving Marian model to %s" % (args.marian,))
np.savez(args.marian, **marianModel)


pickle_save(ALL_CONVERT_CALLS, 'all_convert_calls.pkl')
