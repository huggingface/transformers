#!/usr/bin/env python
import argparse, json, os, re, torch
from collections import OrderedDict
from safetensors.torch import load_file, save_file
from tokenizers import processors
from transformers import Glm4Config, PreTrainedTokenizerFast

STATE_DICT_MAPPING = OrderedDict([
    (r"transformer.output_layer.weight",                           r"lm_head.weight"),
    (r"transformer.output_layer.bias",                             r"lm_head.bias"),
    (r"transformer.embedding.word_embeddings.weight",              r"model.embed_tokens.weight"),
    (r"transformer.rotary_pos_emb.inv_freq",                       None),
    (r"transformer.encoder.final_layernorm.weight",                r"model.norm.weight"),
    (r"transformer.encoder.layers.(\d+).input_layernorm.weight",   r"model.layers.\1.input_layernorm.weight"),
    (r"transformer.encoder.layers.(\d+).post_mlp_layernorm.weight",r"model.layers.\1.post_mlp_layernorm.weight"),
    (r"transformer.encoder.layers.(\d+).post_self_attn_layernorm.weight",r"model.layers.\1.post_self_attn_layernorm.weight"),
    (r"transformer.encoder.layers.(\d+).post_attention_layernorm.weight",r"model.layers.\1.post_attention_layernorm.weight"),
    (r"transformer.encoder.layers.(\d+).self_attention.dense.weight",r"model.layers.\1.self_attn.o_proj.weight"),
    (r"transformer.encoder.layers.(\d+).self_attention.dense.bias",r"model.layers.\1.self_attn.o_proj.bias"),
    (r"transformer.encoder.layers.(\d+).self_attention.query_key_value.(weight|bias)",r"model.layers.\1.self_attn.qkv_proj.\2"),
    (r"transformer.encoder.layers.(\d+).mlp.dense_h_to_4h.weight", r"model.layers.\1.mlp.gate_up_proj.weight"),
    (r"transformer.encoder.layers.(\d+).mlp.dense_h_to_4h.bias",   r"model.layers.\1.mlp.gate_up_proj.bias"),
    (r"transformer.encoder.layers.(\d+).mlp.dense_4h_to_h.weight", r"model.layers.\1.mlp.down_proj.weight"),
    (r"transformer.encoder.layers.(\d+).mlp.dense_4h_to_h.bias",   r"model.layers.\1.mlp.down_proj.bias"),
    (r"transformer.encoder.layers.(\d+).mlp.router.weight",        r"model.layers.\1.mlp.gate.weight"),
    (r"transformer.encoder.layers.(\d+).mlp.router.bias",          r"model.layers.\1.mlp.gate.bias"),
    (r"transformer.encoder.layers.(\d+).mlp.experts.(\d+).gate_proj.(weight|bias)",r"model.layers.\1.mlp.experts.\2.gate_proj.\3"),
    (r"transformer.encoder.layers.(\d+).mlp.experts.(\d+).up_proj.(weight|bias)",  r"model.layers.\1.mlp.experts.\2.up_proj.\3"),
    (r"transformer.encoder.layers.(\d+).mlp.experts.(\d+).down_proj.(weight|bias)",r"model.layers.\1.mlp.experts.\2.down_proj.\3"),
])

def load_raw_state(d):
    print("Scanning weight files …")
    fs = sorted([f for f in os.listdir(d) if f.endswith((".safetensors",".bin"))],
                key=lambda x:int(x.rsplit("-",3)[1]) if "-" in x else 0)
    sd = {}
    for f in fs:
        p = os.path.join(d,f)
        print(f"Loading {p}")
        if f.endswith(".safetensors"):
            sd.update(load_file(p))
        else:
            sd.update(torch.load(p,map_location="cpu"))
    print(f"Total tensors loaded: {len(sd)}")
    return sd

def map_key(k):
    for pat,rep in STATE_DICT_MAPPING.items():
        if rep is None and re.fullmatch(pat,k):
            return None
        nk,n=re.subn(pat,rep or "",k)
        if n:
            return nk
    return None

def split_qkv(t,cfg):
    hd=cfg.hidden_size//cfg.num_attention_heads
    q=cfg.num_attention_heads*hd
    k=cfg.num_key_value_heads*hd
    return t[:q],t[q:q+k],t[q+k:]

def convert_state(sd,cfg):
    new_state={}
    dropped=0
    for k,v in sd.items():
        nk=map_key(k)
        if nk is None:
            dropped+=1
            continue
        if "qkv_proj." in nk:
            q,k_,v_=split_qkv(v,cfg)
            new_state[nk.replace("qkv_proj.","q_proj.")]=q
            new_state[nk.replace("qkv_proj.","k_proj.")]=k_
            new_state[nk.replace("qkv_proj.","v_proj.")]=v_
        else:
            new_state[nk]=v
    print(f"Converted tensors: {len(new_state)}, dropped: {dropped}")
    return new_state

def write_sharded(state,out_dir,size_bytes=5*1024**3,prefix="model"):
    print("Writing sharded safetensors …")
    os.makedirs(out_dir,exist_ok=True)
    wm,shards,cur,cur_size={},[],{},0
    idx=1
    for n,t in state.items():
        nb=t.numel()*t.element_size()
        if nb>size_bytes:
            name=f"{prefix}-{idx:05d}-of-xxxxx.safetensors"
            save_file({n:t},os.path.join(out_dir,name))
            shards.append(name)
            wm[n]=name
            idx+=1
            continue
        if cur_size+nb>size_bytes and cur:
            name=f"{prefix}-{idx:05d}-of-xxxxx.safetensors"
            save_file(cur,os.path.join(out_dir,name))
            shards.append(name)
            for k in cur:
                wm[k]=name
            idx+=1
            cur,cur_size={},0
        cur[n]=t
        cur_size+=nb
    if cur:
        name=f"{prefix}-{idx:05d}-of-xxxxx.safetensors"
        save_file(cur,os.path.join(out_dir,name))
        shards.append(name)
        for k in cur:
            wm[k]=name
    total=len(shards)
    print(f"Total shards written: {total}")
    rename={}
    for s in shards:
        ns=s.replace("xxxxx",f"{total:05d}")
        os.rename(os.path.join(out_dir,s),os.path.join(out_dir,ns))
        rename[s]=ns
    for k in wm:
        wm[k]=rename[wm[k]]
    total_size=sum(t.numel()*t.element_size() for t in state.values())
    index={"metadata":{"total_size":total_size},"weight_map":wm}
    with open(os.path.join(out_dir,f"{prefix}.safetensors.index.json"),"w") as f:
        json.dump(index,f,indent=2)
    print("Index file saved")

def convert_cfg(o):
    mp={"vocab_size":"padded_vocab_size","intermediate_size":"ffn_hidden_size","num_hidden_layers":"num_layers",
        "max_position_embeddings":"seq_length","rms_norm_eps":"layernorm_epsilon","head_dim":"kv_channels",
        "attention_bias":"add_qkv_bias"}
    keep=["num_attention_heads","hidden_size","attention_dropout","use_cache","eos_token_id","pad_token_id","tie_word_embeddings"]
    kw={k:o[v] for k,v in mp.items() if v in o}
    kw.update({k:o[k] for k in keep if k in o})
    kw["num_key_value_heads"]=kw["num_attention_heads"] if not o.get("multi_query_attention",False) else o["multi_query_group_num"]
    kw["rope_theta"]=10000.0*o.get("rope_ratio",1.0)
    return Glm4Config(**kw)

def build_tok(d,post):
    print("Building tokenizer …")
    t=PreTrainedTokenizerFast.from_pretrained(d,model_input_names=["input_ids","attention_mask"])
    if post:
        t._tokenizer.post_processor=processors.Sequence([
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(single="[gMASK]:0 <sop>:0 $A:0",
                                          pair="[gMASK]:0 <sop>:0 $A:0 $B:1",
                                          special_tokens=[("[gMASK]",151331),("<sop>",151333)])
        ])
    else:
        t._tokenizer.post_processor=processors.Sequence([processors.ByteLevel(trim_offsets=False)])
    return t

def convert(src,dst,post):
    print("Starting conversion …")
    os.makedirs(dst,exist_ok=True)
    with open(os.path.join(src,"config.json")) as f:
        cfg=convert_cfg(json.load(f))
    cfg.save_pretrained(dst)
    print("Config saved")
    raw=load_raw_state(src)
    new=convert_state(raw,cfg)
    write_sharded(new,dst)
    build_tok(src,post).save_pretrained(dst)
    print("Tokenizer saved")
    print("Conversion completed successfully")

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("input_dir")
    pa.add_argument("output_dir")
    pa.add_argument("--use_post_processor",action="store_true")
    a=pa.parse_args()
    convert(a.input_dir,a.output_dir,a.use_post_processor)