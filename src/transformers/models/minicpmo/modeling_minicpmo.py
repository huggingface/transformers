import os
import torch
import json
from PIL import Image
import base64
import io
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from transformers import AutoTokenizer, AutoModel

from defines import *

from omnilmm.train.train_utils import omni_preprocess
from processing_minicpmo import MiniCPMoProcessing
from configuration_minicpmo import MiniCPM_oConfig

class OmniLMM12B:
    def __init__(self, model_path) -> None:
        model, img_processor, image_token_len, tokenizer = MiniCPM_oConfig.init_omni_lmm(model_path)
        self.model = model
        self.image_token_len = image_token_len
        self.image_transform = img_processor
        self.tokenizer = tokenizer
        self.model.eval()

    def decode(self, image, input_ids):
        with torch.inference_mode():
            output = self.model.generate_vllm(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=image.unsqueeze(0).half().cuda(),
                temperature=0.6,
                max_new_tokens=1024,
                # num_beams=num_beams,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1,
                top_k=30,
                top_p=0.9,
            )

            response = self.tokenizer.decode(
                output.sequences[0], skip_special_tokens=True)
            response = response.strip()
            return response

    def chat(self, input):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
        except Exception as e:
            return "Image decode error"

        msgs = json.loads(input['question'])
        input_ids = MiniCPMoProcessing.wrap_question_for_omni_lmm(
            msgs, self.image_token_len, self.tokenizer)['input_ids']
        input_ids = torch.as_tensor(input_ids)
        #print('input_ids', input_ids)
        image = self.image_transform(image)

        out = self.decode(image, input_ids)

        return out
        
class MiniCPMV:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval().cuda()

    def chat(self, input):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
        except Exception as e:
            return "Image decode error"

        msgs = json.loads(input['question'])
        
        answer, context, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
    	)
        return answer

class MiniCPMV2_5:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval().cuda()

    def chat(self, input):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
        except Exception as e:
            return "Image decode error"

        msgs = json.loads(input['question'])
        
        answer = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
    	)
        return answer

class MiniCPMV2_6:
    def __init__(self, model_path, multi_gpus=False) -> None:

        print('torch_version:', torch.__version__)
        if multi_gpus: # inference on multi-gpus
            from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map
            with init_empty_weights():
                model = AutoModel.from_pretrained(model_path, trust_remote_code=True, 
                    attn_implementation='sdpa', torch_dtype=torch.bfloat16)

            device_map = infer_auto_device_map(model, max_memory={0: "10GB", 1: "10GB"},
                no_split_module_classes=['SiglipVisionTransformer', 'Qwen2DecoderLayer'])
            device_id = device_map["llm.model.embed_tokens"]
            device_map["llm.lm_head"] = device_id # first and last layer of llm should be in the same device
            device_map["vpm"] = device_id
            device_map["resampler"] = device_id
            device_id2 = device_map["llm.model.layers.26"]
            device_map["llm.model.layers.8"] = device_id2
            device_map["llm.model.layers.9"] = device_id2
            device_map["llm.model.layers.10"] = device_id2
            device_map["llm.model.layers.11"] = device_id2
            device_map["llm.model.layers.12"] = device_id2
            device_map["llm.model.layers.13"] = device_id2
            device_map["llm.model.layers.14"] = device_id2
            device_map["llm.model.layers.15"] = device_id2
            device_map["llm.model.layers.16"] = device_id2
            print(device_map)

            self.model = load_checkpoint_and_dispatch(model, model_path, dtype=torch.bfloat16, device_map=device_map)
            self.model.eval()
        else:
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                attn_implementation='sdpa', torch_dtype=torch.bfloat16)
            self.model.eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def chat(self, input):
        image = None
        if "image" in input and len(input["image"]) > 10: # legacy API
            try:
                image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
            except Exception as e:
                return "Image decode error"

        msgs = json.loads(input["question"])

        for msg in msgs:
            contents = msg.pop('content') # support str or List[Dict]
            if isinstance(contents, str):
                contents = [contents]
            
            new_cnts = []
            for c in contents:
                if isinstance(c, dict):
                    if c['type'] == 'text':
                        c = c['pairs']
                    elif c['type'] == 'image':
                        c = Image.open(io.BytesIO(base64.b64decode(c["pairs"]))).convert('RGB')
                    else:
                        raise ValueError("content type only support text and image.")
                new_cnts.append(c)
            msg['content'] = new_cnts 
        print(f'msgs: {str(msgs)}')

        answer = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
        )
        return answer


class MiniCPMVChat:
    def __init__(self, model_path, multi_gpus=False) -> None:
        if '12B' in model_path:
            self.model = OmniLMM12B(model_path)
        elif 'MiniCPM-Llama3-V' in model_path:
            self.model = MiniCPMV2_5(model_path)
        elif 'MiniCPM-V-2_6' in model_path:
            self.model = MiniCPMV2_6(model_path, multi_gpus)
        else:
            self.model = MiniCPMV(model_path)

    def chat(self, input):
        return self.model.chat(input)

def getPretrainedModel(path='openbmb/OmniLMM-12B'):
    return MiniCPMVChat(path)


__all__ = ["getPretrainedModel","MiniCPMVChat","MiniCPMV2_6","MiniCPMV2_5","MiniCPMV",]