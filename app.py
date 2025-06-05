from argparse import ArgumentParser
from pathlib import Path
import gradio as gr
import os
import tempfile
import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from transformers.video_utils import load_video
import cv2

MODEL_PATH = "/model/glm-4v-9b-0603"
MAX_VIDEO_DURATION = 3600
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="10.244.99.109",
                        help="Demo server name.")
    args = parser.parse_args()
    return args


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        return duration
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0


def is_video_file(filename):
    """检查文件是否为视频文件"""
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg', '.m4v']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


class GLM4VModel:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载模型"""
        if self.model is None:
            print("Loading GLM-4V model...")
            self.processor = AutoProcessor.from_pretrained(
                MODEL_PATH,
                use_fast=True,
                trust_remote_code=True
            )
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            print("Model loaded successfully!")

    def analyze_video(self, video_path, question):
        """分析视频并返回结果"""
        try:
            # 加载模型（如果还没加载）
            self.load_model()

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"file://{video_path}"
                            },
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                }
            ]

            # 加载视频
            video_tensor, video_metadata = load_video(video_path)

            # 处理输入
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=text,
                videos=[video_tensor],
                video_metadata=[video_metadata],
                return_tensors="pt"
            ).to(self.device)

            # 生成回答
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=1.0
                )

            # 解码输出
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            return output_text

        except Exception as e:
            return f"处理视频时出现错误: {str(e)}"


# 全局模型实例
glm4v_model = GLM4VModel()


def _launch_demo(args):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )

    def process_video_and_question(video_file, question, chatbot, history):
        """处理视频和问题"""
        if video_file is None:
            error_msg = "请先上传视频文件"
            chatbot.append({"role": "user", "content": question})
            chatbot.append({"role": "assistant", "content": error_msg})
            return chatbot, history

        if not question.strip():
            error_msg = "请输入您的问题"
            chatbot.append({"role": "assistant", "content": error_msg})
            return chatbot, history

        # 检查文件类型
        if not is_video_file(video_file.name):
            error_msg = "请上传有效的视频文件"
            chatbot.append({"role": "user", "content": question})
            chatbot.append({"role": "assistant", "content": error_msg})
            return chatbot, history

        # 检查视频时长
        duration = get_video_duration(video_file.name)
        if duration > MAX_VIDEO_DURATION:
            error_msg = f"视频时长({duration / 60:.1f}分钟)超过限制(60分钟)，请上传较短的视频"
            chatbot.append({"role": "user", "content": question})
            chatbot.append({"role": "assistant", "content": error_msg})
            return chatbot, history

        # 添加用户消息到聊天框
        user_message = f"📹 视频: {os.path.basename(video_file.name)} ({duration / 60:.1f}分钟)\n💬 问题: {question}"
        chatbot.append({"role": "user", "content": user_message})
        chatbot.append({"role": "assistant", "content": "正在分析视频，请稍候..."})

        # 分析视频
        try:
            response = glm4v_model.analyze_video(video_file.name, question)
            chatbot[-1] = {"role": "assistant", "content": response}

            # 更新历史记录
            history.append({
                "video": video_file.name,
                "question": question,
                "response": response
            })

        except Exception as e:
            error_msg = f"处理过程中出现错误: {str(e)}"
            chatbot[-1] = {"role": "assistant", "content": error_msg}

        return chatbot, history

    def clear_all():
        """清除所有内容"""
        return [], [], "", None

    def regenerate_last(chatbot, history):
        """重新生成最后一个回答"""
        if not history:
            return chatbot

        last_item = history[-1]
        video_path = last_item["video"]
        question = last_item["question"]

        # 更新聊天框显示正在重新生成
        if chatbot and len(chatbot) >= 2:
            user_message = chatbot[-2]["content"]
            chatbot[-1] = {"role": "assistant", "content": "正在重新分析视频，请稍候..."}

        try:
            response = glm4v_model.analyze_video(video_path, question)
            chatbot[-1] = {"role": "assistant", "content": response}
            history[-1]["response"] = response
        except Exception as e:
            error_msg = f"重新生成时出现错误: {str(e)}"
            chatbot[-1] = {"role": "assistant", "content": error_msg}

        return chatbot

    # 创建Gradio界面
    with gr.Blocks(title="GLM-4V 视频分析") as demo:
        gr.Markdown("""
        <center>
        <h1>🎬 GLM-4V 视频分析助手</h1>
        <p>上传视频文件，提出问题，让AI为您详细分析视频内容</p>
        <p><small>⚠️ 视频时长限制：最长60分钟</small></p>
        </center>
        """)

        # 状态变量
        history = gr.State([])

        # 主要组件
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.File(
                    label="📁 上传视频文件",
                    file_types=["video"],
                    type="filepath"
                )
                question_input = gr.Textbox(
                    label="💭 输入您的问题",
                    placeholder="例如：详细描述一下这个视频的内容",
                    lines=3
                )

                with gr.Row():
                    submit_btn = gr.Button("🚀 开始分析", variant="primary")
                    clear_btn = gr.Button("🧹 清除所有")
                    regen_btn = gr.Button("🔄 重新生成")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="分析结果",
                    height=500,
                    type="messages",
                    elem_classes="chatbot-container"
                )

        submit_btn.click(
            fn=process_video_and_question,
            inputs=[video_input, question_input, chatbot, history],
            outputs=[chatbot, history],
            show_progress=True
        ).then(
            fn=lambda: "",
            outputs=question_input
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[chatbot, history, question_input, video_input]
        )

        regen_btn.click(
            fn=regenerate_last,
            inputs=[chatbot, history],
            outputs=chatbot,
        )

        question_input.submit(
            fn=process_video_and_question,
            inputs=[video_input, question_input, chatbot, history],
            outputs=[chatbot, history],
        ).then(
            fn=lambda: "",
            outputs=question_input
        )

    demo.queue().launch(server_port=args.server_port, server_name=args.server_name,share=True)


def main():
    args = _get_args()
    _launch_demo(args)


if __name__ == '__main__':
    main()