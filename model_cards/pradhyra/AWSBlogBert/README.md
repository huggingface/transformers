This model is pre-trained on blog articles from AWS Blogs.

## Pre-training corpora
The input text contains around 3000 blog articles on [AWS Blogs website](https://aws.amazon.com/blogs/) technical subject matter including AWS products, tools and tutorials. 

## Pre-training details
I picked a Roberta architecture for masked language modeling (6-layer, 768-hidden, 12-heads, 82M parameters) and its corresponding ByteLevelBPE tokenization strategy. I then followed HuggingFace's Transformers [blog post](https://huggingface.co/blog/how-to-train) to train the model.
I chose to follow the following training set-up: 28k training steps with batches of 64 sequences of length 512 with an initial learning rate 5e-5. The model acheived a training loss of 3.6 on the MLM task over 10 epochs.
