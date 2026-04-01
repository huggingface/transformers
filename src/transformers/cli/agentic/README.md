# Agentic CLI for Transformers

Single-command access to all major Transformers use-cases. Designed for AI
agents and humans who need to run inference, training, quantization, export,
and model inspection **without writing Python scripts**.

Every command below is available as `transformers <command>`. Run
`transformers <command> --help` for full option documentation.

## How it works

The module integrates with the main CLI through a single function call in
`transformers.py` — removing it disables everything with no side effects.

```
src/transformers/cli/agentic/
├── app.py            # register_agentic_commands(app) — the single integration point
├── _common.py        # Shared helpers (input resolution, output formatting, media loaders)
├── inference.py      # Pipeline-based inference (~25 commands)
├── generate.py       # Text generation with streaming, decoding control, tool calling
├── train.py          # Fine-tuning / pretraining via Trainer
├── quantize.py       # Model quantization (BnB, GPTQ, AWQ)
├── export.py         # Model export (ONNX, GGUF, ExecuTorch)
└── utilities.py      # Embeddings, tokenization, model inspection, benchmarking
```

## Common options

Every inference command supports:

| Option | Description |
|--------|-------------|
| `--model` / `-m` | Model ID (Hub) or local path |
| `--device` | `cpu`, `cuda`, `cuda:0`, `mps` |
| `--dtype` | `auto`, `float16`, `bfloat16`, `float32` |
| `--trust-remote-code` | Trust custom model code from the Hub |
| `--token` | HF Hub token for gated/private models |
| `--revision` | Model revision (branch, tag, SHA) |
| `--json` | Machine-readable JSON output |

Text commands also accept `--file` to read input from a file, or stdin
via pipe (`echo "hello" | transformers classify`).

## Commands

### Text Inference

1. Classify text into categories (supervised)
   ```bash
   transformers classify --model distilbert/distilbert-base-uncased-finetuned-sst-2-english --text "Great movie!"
   ```

2. Classify text into arbitrary categories without training (zero-shot)
   ```bash
   transformers classify --text "The stock market crashed today." --labels "politics,finance,sports"
   ```

3. Extract named entities from text (NER)
   ```bash
   transformers ner --model dslim/bert-base-NER --text "Apple CEO Tim Cook met with President Biden in Washington."
   ```

4. Tag tokens with labels (POS tagging, chunking)
   ```bash
   transformers token-classify --model vblagoje/bert-english-uncased-finetuned-pos --text "The cat sat on the mat."
   ```

5. Answer a question given a context paragraph (extractive QA)
   ```bash
   transformers qa --question "Who invented the telephone?" --context "Alexander Graham Bell invented the telephone in 1876."
   ```

6. Answer a question about tabular data
   ```bash
   transformers table-qa --question "What is the total revenue?" --table financials.csv
   ```

7. Summarize text
   ```bash
   transformers summarize --model facebook/bart-large-cnn --file article.txt
   ```

8. Translate text between languages
   ```bash
   transformers translate --model Helsinki-NLP/opus-mt-en-de --text "The weather is nice today."
   ```

9. Fill in masked tokens in a sentence
   ```bash
   transformers fill-mask --model bert-base-uncased --text "The capital of France is [MASK]."
   ```

### Text Generation

10. Generate text from a prompt
    ```bash
    transformers generate --model meta-llama/Llama-3.2-1B-Instruct --prompt "Once upon a time"
    ```

11. Stream text generation token-by-token
    ```bash
    transformers generate --model meta-llama/Llama-3.2-1B-Instruct --prompt "Hello" --stream
    ```

12. Generate with sampling (temperature, top-p, top-k)
    ```bash
    transformers generate --prompt "The future of AI" --temperature 0.7 --top-p 0.9
    ```

13. Generate with beam search
    ```bash
    transformers generate --prompt "Translate this:" --num-beams 4
    ```

14. Run speculative decoding with a draft model
    ```bash
    transformers generate --model meta-llama/Llama-3.1-8B-Instruct --assistant-model meta-llama/Llama-3.2-1B-Instruct --prompt "Explain gravity."
    ```

15. Generate with tool/function calling
    ```bash
    transformers generate --model meta-llama/Llama-3.2-1B-Instruct --prompt "What is the weather?" --tools tools.json
    ```

16. Generate with constrained JSON output
    ```bash
    transformers generate --prompt "List 3 items as JSON:" --grammar json
    ```

17. Watermark generated text
    ```bash
    transformers generate --model meta-llama/Llama-3.2-1B-Instruct --prompt "Write an essay." --watermark
    ```

18. Detect whether text was watermarked
    ```bash
    transformers detect-watermark --model meta-llama/Llama-3.2-1B-Instruct --text "The generated essay text..."
    ```

19. Generate with a quantized model (4-bit)
    ```bash
    transformers generate --model meta-llama/Llama-3.1-8B-Instruct --prompt "Hello" --quantization bnb-4bit
    ```

20. Generate with quantized KV cache for long context
    ```bash
    transformers generate --model meta-llama/Llama-3.1-8B-Instruct --prompt "Summarize this long text..." --cache-quantization 4bit
    ```

### Vision

21. Classify an image into categories
    ```bash
    transformers image-classify --model google/vit-base-patch16-224 --image photo.jpg
    ```

22. Classify an image into arbitrary categories without training (zero-shot)
    ```bash
    transformers image-classify --model openai/clip-vit-base-patch32 --image photo.jpg --labels "cat,dog,bird,fish"
    ```

23. Detect objects in an image with bounding boxes
    ```bash
    transformers detect --model facebook/detr-resnet-50 --image street.jpg
    ```

24. Detect objects from a text description (grounded detection)
    ```bash
    transformers detect --model IDEA-Research/grounding-dino-base --image kitchen.jpg --text "red mug on the counter"
    ```

25. Segment an image by class (semantic segmentation)
    ```bash
    transformers segment --model nvidia/segformer-b0-finetuned-ade-512-512 --image scene.jpg
    ```

26. Segment individual object instances in an image
    ```bash
    transformers segment --model facebook/mask2former-swin-base-coco-instance --image crowd.jpg
    ```

27. Generate segmentation masks interactively (SAM-style)
    ```bash
    transformers segment --model facebook/sam-vit-base --image photo.jpg --points "[[120,45]]" --point-labels "[1]"
    ```

28. Estimate depth from a single image
    ```bash
    transformers depth --model Intel/dpt-large --image room.jpg --output depth_map.png
    ```

29. Detect and match keypoints across an image pair
    ```bash
    transformers keypoints --model magic-leap-community/superglue --images img1.jpg --images img2.jpg
    ```

30. Extract feature vectors from an image
    ```bash
    transformers embed --model google/vit-base-patch16-224 --image photo.jpg --output features.npy
    ```

### Audio

31. Transcribe speech to text
    ```bash
    transformers transcribe --model openai/whisper-small --audio recording.wav
    ```

32. Transcribe speech with word-level timestamps
    ```bash
    transformers transcribe --model openai/whisper-small --audio recording.wav --timestamps word --json
    ```

33. Classify an audio clip into categories
    ```bash
    transformers audio-classify --model MIT/ast-finetuned-audioset-10-10-0.4593 --audio clip.wav
    ```

34. Classify audio into arbitrary categories without training (zero-shot)
    ```bash
    transformers audio-classify --model laion/clap-htsat-unfused --audio clip.wav --labels "speech,music,noise,silence"
    ```

35. Generate speech from text (text-to-speech)
    ```bash
    transformers speak --model suno/bark-small --text "Hello, how are you today?" --output speech.wav
    ```

36. Generate audio from a text description (music, sound effects)
    ```bash
    transformers audio-generate --model facebook/musicgen-small --text "A calm piano melody" --output music.wav
    ```

### Video

37. Classify a video clip into categories
    ```bash
    transformers video-classify --model MCG-NJU/videomae-base-finetuned-kinetics --video clip.mp4
    ```

### Multimodal

38. Answer a question about an image (visual QA)
    ```bash
    transformers vqa --model Salesforce/blip2-opt-2.7b --image chart.png --question "What is the trend shown?"
    ```

39. Answer a question about a document image (document QA)
    ```bash
    transformers document-qa --model impira/layoutlm-document-qa --image invoice.png --question "What is the total amount?"
    ```

40. Generate a caption for an image
    ```bash
    transformers caption --model Salesforce/blip2-opt-2.7b --image sunset.jpg
    ```

41. Extract text from a document image (OCR)
    ```bash
    transformers ocr --model naver-clova-ix/donut-base-finetuned-cord-v2 --image receipt.png
    ```

42. Single-turn conversation with mixed inputs (image, audio, text)
    ```bash
    transformers multimodal-chat --model meta-llama/Llama-4-Scout-17B-16E-Instruct --prompt "Describe what you see and hear." --image photo.jpg --audio clip.wav
    ```

### Training

43. Fine-tune a text classification model
    ```bash
    transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./sst2-finetuned --epochs 3 --lr 2e-5
    ```

44. Fine-tune a token classification model (NER)
    ```bash
    transformers train token-classification --model bert-base-uncased --dataset conll2003 --output ./ner-finetuned --epochs 5
    ```

45. Fine-tune a question answering model
    ```bash
    transformers train question-answering --model bert-base-uncased --dataset squad --output ./qa-finetuned --epochs 2
    ```

46. Fine-tune a summarization model
    ```bash
    transformers train summarization --model t5-small --dataset cnn_dailymail --output ./summarizer --epochs 3
    ```

47. Fine-tune a translation model
    ```bash
    transformers train translation --model t5-small --dataset wmt16/de-en --output ./translator
    ```

48. Continued pretraining on a domain-specific corpus
    ```bash
    transformers train language-modeling --model bert-base-uncased --dataset ./corpus.txt --output ./domain-bert --mlm
    ```

49. Fine-tune an LLM with LoRA
    ```bash
    transformers train text-generation --model meta-llama/Llama-3.2-1B --dataset ./instructions.jsonl --output ./llama-lora --lora --lora-r 16
    ```

50. Fine-tune a 4-bit quantized LLM with QLoRA
    ```bash
    transformers train text-generation --model meta-llama/Llama-3.1-8B --dataset ./instructions.jsonl --output ./llama-qlora --lora --quantization bnb-4bit
    ```

51. Pretrain a language model from scratch
    ```bash
    transformers train language-modeling --model-config gpt2 --dataset ./corpus.txt --output ./my-lm --from-scratch
    ```

52. Fine-tune an image classification model
    ```bash
    transformers train image-classification --model google/vit-base-patch16-224 --dataset food101 --output ./food-classifier --epochs 5
    ```

53. Fine-tune an object detection model
    ```bash
    transformers train object-detection --model facebook/detr-resnet-50 --dataset cppe-5 --output ./detector --epochs 10
    ```

54. Fine-tune a segmentation model
    ```bash
    transformers train semantic-segmentation --model nvidia/segformer-b0-finetuned-ade-512-512 --dataset scene_parse_150 --output ./segmenter
    ```

55. Fine-tune an ASR model on domain-specific audio
    ```bash
    transformers train speech-recognition --model openai/whisper-small --dataset ./medical-audio/ --output ./medical-whisper --epochs 5
    ```

56. Fine-tune an audio classification model
    ```bash
    transformers train audio-classification --model MIT/ast-finetuned-audioset-10-10-0.4593 --dataset superb/ks --output ./audio-classifier
    ```

57. Run hyperparameter search with Optuna
    ```bash
    transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./hpo-run --hpo optuna --hpo-trials 20
    ```

58. Resume training from a checkpoint
    ```bash
    transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./sst2-finetuned --resume-from-checkpoint ./sst2-finetuned/checkpoint-500
    ```

59. Train with early stopping
    ```bash
    transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./sst2-finetuned --early-stopping --early-stopping-patience 3
    ```

60. Evaluate periodically during training
    ```bash
    transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./sst2-finetuned --eval-strategy steps --eval-steps 100
    ```

### Distributed & Large-Scale Training

61. Train across multiple GPUs on a single machine
    ```bash
    transformers train text-generation --model meta-llama/Llama-3.2-1B --dataset ./data.jsonl --output ./multi-gpu --multi-gpu
    ```

62. Train across multiple nodes
    ```bash
    transformers train text-generation --model meta-llama/Llama-3.2-1B --dataset ./data.jsonl --output ./multi-node --nnodes 4
    ```

63. Train with DeepSpeed ZeRO
    ```bash
    transformers train text-generation --model meta-llama/Llama-3.1-8B --dataset ./data.jsonl --output ./deepspeed-run --deepspeed zero3
    ```

64. Train with FSDP
    ```bash
    transformers train text-generation --model meta-llama/Llama-3.1-8B --dataset ./data.jsonl --output ./fsdp-run --fsdp full-shard
    ```

65. Train on TPUs
    ```bash
    transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./tpu-run --device tpu
    ```

66. Train on Apple Silicon (MPS)
    ```bash
    transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./mps-run --device mps
    ```

67. Train with mixed precision
    ```bash
    transformers train text-generation --model meta-llama/Llama-3.2-1B --dataset ./data.jsonl --output ./bf16-run --dtype bf16
    ```

68. Train with gradient checkpointing
    ```bash
    transformers train text-generation --model meta-llama/Llama-3.1-8B --dataset ./data.jsonl --output ./gc-run --gradient-checkpointing
    ```

69. Train with gradient accumulation
    ```bash
    transformers train text-generation --model meta-llama/Llama-3.2-1B --dataset ./data.jsonl --output ./ga-run --gradient-accumulation-steps 8
    ```

### Quantization

70. Quantize a model to 4-bit
    ```bash
    transformers quantize --model meta-llama/Llama-3.1-8B --method bnb-4bit --output ./llama-4bit
    ```

71. Quantize a model to 8-bit
    ```bash
    transformers quantize --model meta-llama/Llama-3.1-8B --method bnb-8bit --output ./llama-8bit
    ```

72. Run GPTQ quantization with calibration data
    ```bash
    transformers quantize --model meta-llama/Llama-3.1-8B --method gptq --calibration-dataset wikitext --output ./llama-gptq
    ```

73. Run AWQ quantization
    ```bash
    transformers quantize --model meta-llama/Llama-3.1-8B --method awq --output ./llama-awq
    ```

74. Compare quality across quantization methods
    ```bash
    transformers benchmark-quantization --model meta-llama/Llama-3.1-8B --methods none,bnb-4bit,bnb-8bit --json
    ```

### Export

75. Export a model to ONNX
    ```bash
    transformers export onnx --model bert-base-uncased --output ./bert-onnx/
    ```

76. Convert a model to GGUF for llama.cpp
    ```bash
    transformers export gguf --model meta-llama/Llama-3.2-1B --output llama-1b.gguf
    ```

77. Export a model to ExecuTorch for mobile/edge
    ```bash
    transformers export executorch --model distilbert-base-uncased --output ./model.pte
    ```

### Utilities

78. Compute text embeddings
    ```bash
    transformers embed --model sentence-transformers/all-MiniLM-L6-v2 --text "The quick brown fox." --output embeddings.npy
    ```

79. Tokenize text and display tokens
    ```bash
    transformers tokenize --model meta-llama/Llama-3.2-1B-Instruct --text "Hello, world!" --ids
    ```

80. Inspect a model's configuration (no weight download)
    ```bash
    transformers inspect meta-llama/Llama-3.2-1B-Instruct --json
    ```

81. Examine attention weights and hidden states
    ```bash
    transformers inspect-forward --model bert-base-uncased --text "The cat sat on the mat." --output ./activations/
    ```
