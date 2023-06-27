<!--版权2022年HuggingFace团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证 LICENSE ”）获得许可；除非符合许可证的要求，否则不得使用此文件。您可以在以下网址获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“原样 AS IS”分发，不附带任何明示或暗示的担保或条件。有关许可证下的特定语言的权限和限制，请参阅许可证。
⚠️请注意，此文件虽然是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# 使用🤗 Accelerate 进行分布式训练
随着模型越来越大，并行性已成为在有限硬件上训练更大模型的策略，通过数个数量级加速训练速度。在 Hugging Face，我们创建了 [🤗 Accelerate](https://huggingface.co/docs/accelerate) 库，以帮助用户在任何类型的分布式环境中轻松训练🤗 Transformers 模型，无论是一台机器上的多个 GPU 还是多台机器上的多个 GPU。在本教程中，了解如何自定义原生 PyTorch 训练循环以启用分布式环境中的训练。
## 设置 Setup

首先安装🤗 Accelerate：
```bash
pip install accelerate
```

然后导入并创建 [`~accelerate.Accelerator`] 对象。[`~accelerate.Accelerator`] 将自动检测您的分布式设置类型，并初始化所有必要的训练组件。您不需要显式地将模型放置在设备上。
```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## 准备加速 Prepare to accelerate
下一步是将所有相关的训练对象传递给 [`~accelerate.Accelerator.prepare`] 方法。这包括您的训练和评估 DataLoader，模型和优化器：
```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## 回退（Backward）
最后一步是将训练循环中典型的 `loss.backward()` 替换为🤗 Accelerate 的 [`~accelerate.Accelerator.backward`] 方法：
```py
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         outputs = model(**batch)
...         loss = outputs.loss
...         accelerator.backward(loss)

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

如下面的代码所示，您只需要在训练循环中添加四行额外的代码即可启用分布式训练！
```diff
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

## 训练
在添加相关代码行后，可以在脚本或笔记本（如 Colaboratory）中启动训练。

### 使用脚本进行训练

如果要从脚本中运行训练，请运行以下命令以创建并保存配置文件：

```bash
accelerate config
```

然后使用以下命令启动训练：
```bash
accelerate launch train.py
```

### 使用笔记本进行训练
如果您计划使用 Colaboratory 的 TPU，🤗 Accelerate 也可以在笔记本中运行。将负责训练的所有代码封装在一个函数中，并将其传递给 [`~accelerate.notebook_launcher`]：
```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

有关🤗 Accelerate 及其丰富功能的更多信息，请参阅 [文档](https://huggingface.co/docs/accelerate)。