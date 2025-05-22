# Continious batching

## 1. What is CB (Continiuous Batching)
Continious Batching enables very fast model evalutation when you are training a model! It is based on (paged-attention)[https://arxiv.org/abs/2309.06180]. It's a very efficient way of preparing your inputs for the model, to optimize the token throuput of your model. Meaning for each forward pass of your model, you generate as many tokens as possible. 

Let me explain what I mean by that.
A normal model that uses `batched` generation will have inputs of size (batch_size, max_input_length). As an AI model is basically a `matrix multiplication`, it needs static shape, so you have to pad the inputs to the maximum length of the input in your batch.

Let's use an example:
![batched image](image.png)

When you pack you sequences for efficiency, you end up with a batch_size of 1, and just `max_input_length`. 
![ragged inputs](image-1.png)
The above is actually not "accurate", it's missing padding tokens to fit into the `max_input_length`. 
As otherwise you have a `dynamic` running lenght for your model, which prevents it from being optimized.

When the model finishes generating and you are in a batched setting, you usually never remove the finished sequence from the batch. You would just wait for all requests to be done. This would give something like this:
![batched-generation](image-2.png)
as you can see, the red + the blue tokens are all tokens that you process, and waste compute on.

With **Dynamic Batching** you remove the sequence as soon as it is generated. The next graph shows the input to the 5 forward pass the model has to run.
![cb-generation](image-3.png)


## 2. How it works?

### Chunked prefill:
![chunked-prefill](image-4.png)