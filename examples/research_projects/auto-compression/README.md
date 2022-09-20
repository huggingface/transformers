# Huggingface BERT Quantization Compression Example

This example uses [ACT](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression) from [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) for BERT quantization.
The quantized model can be deployed on TensorRT.

### Experiment

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/main/examples/research_projects/auto-compression/run_glue.py).

#### 1. Environment Dependencies Installation

- paddlepaddle>=2.3.2
- paddleslim>=2.3.4
- pycocotools

```shell
# Take Ubuntu and CUDA 11.2 as an example for GPU, and other environments can be installed directly according to Paddle's official website.
#  https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html
python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# CPU
#pip install paddlepaddle==2.3.2
pip install paddleslim==2.3.4
```

#### 2. Prepare the dataset
This script can be used for a dataset hosted on GLUE dataset support by datasets or your own data in a csv or a JSON file. The data will be load automatically, so you can specify the task you want (with the ``--task_name`` argument).

#### 3. Fine-tuning the FP32 model for compress
Finetune a fp32 precision model with [transformers/examples/pytorch/text-classification/](../../pytorch/text-classification/):

```bash
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./tmp/$TASK_NAME/
```

#### 4. Export FP32 model to ONNX


#### 5. Auto Compression

- Single card training:
```
export TASK_NAME=mrpc

export CUDA_VISIBLE_DEVICES=0
python run_glue.py --config_path=./train_config.yaml --task_name=$TASK_NAME --max_length=128 --model_name_or_path=./tmp/$TASK_NAME --per_device_train_batch_size 40 --per_device_eval_batch_size 40 --output_dir ./tmp_ac/$TASK_NAME 
```

#### 6. Deployment

##### Deploy with Paddle TensorRT

- Python test:

First install the [PaddlePaddle with TensorRT](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python).

Then use [paddle_trt_infer.py](./paddle_trt_infer.py) to deploy:
```shell
python paddle_trt_infer.py --task_name=$TASK_NAME --model_name_or_path=./tmp_ac/$TASK_NAME --device='gpu' --perf --int8
```

#### 6.FAQ

