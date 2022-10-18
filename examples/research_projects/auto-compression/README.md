# Huggingface BERT Quantization Compression Example

This example uses [ACT](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression) from [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) for BERT quantization.
The quantized model can be deployed on Paddle Inference and ONNXRuntime.

### Experiment

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/main/examples/research_projects/auto-compression/run_glue.py).
We provide a compression example on the bert-base-cased, if you want to compress other model, only need to change the dataload and model before compress.

#### 1. benchmark

| model | strategy | CoLA | MRPC | QNLI | QQP | RTE | SST2  | STSB  | AVG |
|:------:|:------:|:------:|:------:|:-----------:|:------:|:------:|:------:|:------:|:------:|
| bert-base-cased | origin model | 60.06 | 84.31 | 90.68 | 90.84 | 63.53 | 91.63  | 88.46 |  81.35  |
| bert-base-cased | structure prune & quant aware | 58.69 | 85.05 | 90.74 | 90.42 | 65.34 | 92.08 | 88.22 |  81.51 |

|  bert-base-cased | Accuracy（avg） | Latency(ms) | SpeedUp |
|:-------:|:----------:|:------------:| :------:|
| origin model |  81.35 | - | - |
| structure prune & quant aware |  81.51 | - | - |

- Nvidia GPU ：
  - hardware：One Nvidia Tesla T4 GPU
  - software：CUDA 11.2, cuDNN 8.0, TensorRT 8.4
  - hyperparameter：batch_size: 40, seqence length: 128


#### 2. Environment Dependencies Installation

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

#### 3. Prepare the dataset
This script can be used for a dataset hosted on GLUE dataset support by datasets or your own data in a csv or a JSON file. The data will be load automatically, so you can specify the task you want (with the ``--task_name`` argument).

#### 4. Fine-tuning the FP32 model for compress
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

#### 5. Export FP32 Models for Compression Acceleration

##### 5.1 Export FP32 Model
```
pip install x2paddle
pip install torch == 1.6.0
python convert_pt2pd.py  --task_name=$TASK_NAME --model_name_or_path=../../pytorch/text-classification/tmp/$TASK_NAME/ --task_name=$TASK_NAME --save_dir=./x2paddle/$TASK_NAME
```

##### 5.2 Convert FP32 Model to dynamic batch size
if you want to get PaddlePaddle inference model with dynamic batch size, please modify ```x2paddle_code.py``` generate by x2paddle in directory reference by ```convert_dynamic_shape.py``` or can use ```convert_dynamic_shape.py``` to convert bert model.

```
cp convert_dynamic_shape.py ./x2paddle/$TASK_NAME/
cd ./x2paddle/$TASK_NAME/
python convert_dynamic_shape.py --task_name=$TASK_NAME
```

NOTE: If you want to train/eval model with batch_size > 1, please convert PaddlePaddle inference model with dynamic batch size.

#### 6. Auto Compression

- Train on a Single GPU:
```
export TASK_NAME=mrpc

export CUDA_VISIBLE_DEVICES=0
python run_glue.py --config_path=./train_config.yaml --task_name=$TASK_NAME --max_length=128 --model_name_or_path=./x2paddle/$TASK_NAME --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --output_dir ./tmp_ac/$TASK_NAME 
```

#### 7. Deployment

##### 7.1 Deploy with Paddle Inference

- Python test:

First install the [PaddlePaddle with TensorRT](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python).

Then use [paddle_trt_infer.py](./paddle_trt_infer.py) to deploy:
```shell
python paddle_trt_infer.py --task_name=$TASK_NAME --model_name_or_path=./tmp_ac/$TASK_NAME --batch_size=32 --device='gpu' --perf --int8
```

#### 8.FAQ

