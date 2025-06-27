import gradio as gr
import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration, TextIteratorStreamer
from pathlib import Path
import threading
import re
import argparse
import copy
import spaces
import tempfile
import subprocess
import os
import fitz

MODEL_PATH = "/model/glm-4v-9b-0529"


class GLM4VModel:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, disable_grouping=False)
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
        )

    def _strip_html(self, t):
        return re.sub(r"<[^>]+>", "", t).strip()

    def _wrap_text(self, t):
        return [{"type": "text", "text": t}]

    def _pdf_to_imgs(self, pdf_path):
        doc = fitz.open(pdf_path)
        imgs = []
        for i in range(doc.page_count):
            pix = doc.load_page(i).get_pixmap(dpi=180)
            img_p = os.path.join(tempfile.gettempdir(), f"{Path(pdf_path).stem}_{i}.png")
            pix.save(img_p)
            imgs.append(img_p)
        doc.close()
        return imgs

    def _ppt_to_imgs(self, ppt_path):
        tmp = tempfile.mkdtemp()
        subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, ppt_path], check=True)
        pdf_path = os.path.join(tmp, Path(ppt_path).stem + ".pdf")
        return self._pdf_to_imgs(pdf_path)

    def _files_to_content(self, media):
        out = []
        for f in media or []:
            ext = Path(f.name).suffix.lower()
            if ext in [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg", ".m4v"]:
                out.append({"type": "video", "url": f.name})
            elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
                out.append({"type": "image", "url": f.name})
            elif ext in [".ppt", ".pptx"]:
                for p in self._ppt_to_imgs(f.name):
                    out.append({"type": "image", "url": p})
            elif ext == ".pdf":
                for p in self._pdf_to_imgs(f.name):
                    out.append({"type": "image", "url": p})
        return out

    def _format_output(self, txt):
        think_pat, ans_pat = r"<think>(.*?)</think>", r"<answer>(.*?)</answer>"
        think = re.findall(think_pat, txt, re.DOTALL)
        ans = re.findall(ans_pat, txt, re.DOTALL)
        html = ""
        if think:
            html += "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking Process</summary><div style='color:#cccccc;line-height:1.4;'>" + think[0].strip() + "</div></details><br>"
        body = ans[0] if ans else re.sub(think_pat, "", txt, flags=re.DOTALL)
        html += f"<div style='color:#ffffff;'>{body.strip()}</div>"
        return html

    def _stream_fragment(self, buf):
        think_html = ""
        if "<think>" in buf:
            if "</think>" in buf:
                seg = re.search(r"<think>(.*?)</think>", buf, re.DOTALL)
                if seg:
                    think_html = "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking Process</summary><div style='color:#cccccc;line-height:1.4;'>" + seg.group(1).strip() + "</div></details><br>"
            else:
                part = buf.split("<think>", 1)[1]
                think_html = "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking Process</summary><div style='color:#cccccc;line-height:1.4;'>" + part
        answer_html = ""
        if "<answer>" in buf:
            if "</answer>" in buf:
                seg = re.search(r"<answer>(.*?)</answer>", buf, re.DOTALL)
                if seg:
                    answer_html = "<div style='color:#ffffff;'>" + seg.group(1).strip() + "</div>"
            else:
                part = buf.split("<answer>", 1)[1]
                answer_html = "<div style='color:#ffffff;'>" + part
        if not think_html and not answer_html:
            return self._strip_html(buf)
        return think_html + answer_html

    def _build_messages(self, hist, sys_prompt):
        msgs = []
        if sys_prompt.strip():
            msgs.append({"role": "system", "content": [{"type": "text", "text": sys_prompt.strip()}]})
        for h in hist:
            if h["role"] == "user":
                payload = h.get("file_info") or self._wrap_text(self._strip_html(h["content"]))
                msgs.append({"role": "user", "content": payload})
            else:
                raw = re.sub(r"<think>.*?</think>", "", h["content"], flags=re.DOTALL)
                raw = re.sub(r"<details.*?</details>", "", raw, flags=re.DOTALL)
                msgs.append({"role": "assistant", "content": self._wrap_text(self._strip_html(raw))})
        print(msgs)
        return msgs

    @spaces.GPU(duration=240)
    def stream_generate(self, hist, sys_prompt):
        msgs = self._build_messages(hist, sys_prompt)
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to(self.device)
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=False)
        gen_args = dict(inputs, max_new_tokens=8192, repetition_penalty=1.1, do_sample=True, top_k=2, temperature=None, top_p=1e-5, streamer=streamer)
        threading.Thread(target=self.model.generate, kwargs=gen_args).start()
        buf = ""
        for tok in streamer:
            buf += tok
            yield self._stream_fragment(buf)
        yield self._format_output(buf)


glm4v = GLM4VModel()
glm4v.load()


def check_files(files):
    vids = imgs = ppts = pdfs = 0
    for f in files or []:
        ext = Path(f.name).suffix.lower()
        if ext in [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg", ".m4v"]:
            vids += 1
        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
            imgs += 1
        elif ext in [".ppt", ".pptx"]:
            ppts += 1
        elif ext == ".pdf":
            pdfs += 1
    if vids > 1 or ppts > 1 or pdfs > 1:
        return False, "只允许上传 1 个视频或 1 个 PPT 或 1 个 PDF"
    if imgs > 10:
        return False, "最多上传 10 张图片"
    if (ppts or pdfs) and (vids or imgs) or (vids and imgs):
        return False, "文档、视频、图片不可混合上传"
    return True, ""


def chat(files, msg, hist, sys_prompt):
    ok, err = check_files(files)
    if not ok:
        hist.append({"role": "assistant", "content": err})
        yield copy.deepcopy(hist), None, ""
        return
    payload = glm4v._files_to_content(files) if files else None
    if msg.strip():
        if payload is None:
            payload = glm4v._wrap_text(msg.strip())
        else:
            payload.append({"type": "text", "text": msg.strip()})
    display = f"[{len(files)} file(s) uploaded]\n{msg}" if files else msg
    user_rec = {"role": "user", "content": display}
    if payload:
        user_rec["file_info"] = payload
    hist.append(user_rec)
    place = {"role": "assistant", "content": ""}
    hist.append(place)
    yield copy.deepcopy(hist), None, ""
    for chunk in glm4v.stream_generate(hist[:-1], sys_prompt):
        place["content"] = chunk
        yield copy.deepcopy(hist), None, ""
    yield copy.deepcopy(hist), None, ""


def reset():
    return [], None, ""


css = """.chatbot-container .message-wrap .message{font-size:14px!important}
details summary{cursor:pointer;font-weight:bold}
details[open] summary{margin-bottom:10px}"""

demo = gr.Blocks(title="GLM-4.1V-9B-Thinking Chat", theme=gr.themes.Soft(), css=css)
with demo:
    gr.Markdown("<div style='text-align:center;font-size:32px;font-weight:bold;margin-bottom:20px;'>GLM-4.1V-9B-Thinking Gradio Space🤗</div><div style='text-align:center;'><a href='https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking'>🤗 Model Hub</a> | <a href='https://github.com/THUDM/CogVLM'>🌐 Github</a> | <a href='https://arxiv.org/abs/'>📜 arxiv</a></div>")
    with gr.Row():
        with gr.Column(scale=7):
            chatbox = gr.Chatbot(label="Conversation", type="messages", height=600, elem_classes="chatbot-container")
            textbox = gr.Textbox(label="💭 Message", lines=3)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=3):
            up = gr.File(label="📁 Upload", file_count="multiple", file_types=["file"], type="filepath")
            gr.Markdown("<span style='color:red'>支持图片 / 视频 / PPT / PDF</span>")
            sys = gr.Textbox(label="⚙️ System Prompt", lines=6)
    send.click(chat, inputs=[up, textbox, chatbox, sys], outputs=[chatbox, up, textbox])
    textbox.submit(chat, inputs=[up, textbox, chatbox, sys], outputs=[chatbox, up, textbox])
    clear.click(reset, outputs=[chatbox, up, textbox])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(server_port=args.port, server_name=args.host, share=args.share)