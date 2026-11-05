<!--版权2023年HuggingFace团队保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则按“按原样”分发的软件，无论是明示还是暗示的，都没有任何担保或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️ 请注意，本文件虽然使用Markdown编写，但包含了特定的语法，适用于我们的doc-builder（类似于MDX），可能无法在您的Markdown查看器中正常渲染。

-->

# 🤗 加速分布式训练

随着模型变得越来越大，并行性已经成为在有限硬件上训练更大模型和加速训练速度的策略，增加了数个数量级。在Hugging Face，我们创建了[🤗 加速](https://huggingface.co/docs/accelerate)库，以帮助用户在任何类型的分布式设置上轻松训练🤗 Transformers模型，无论是在一台机器上的多个GPU还是在多个机器上的多个GPU。在本教程中，了解如何自定义您的原生PyTorch训练循环，以启用分布式环境中的训练。

## 设置

通过安装🤗 加速开始:

```bash
pip install accelerate
```

然后导入并创建[`~accelerate.Accelerator`]对象。[`~accelerate.Accelerator`]将自动检测您的分布式设置类型，并初始化所有必要的训练组件。您不需要显式地将模型放在设备上。

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## 准备加速

下一步是将所有相关的训练对象传递给[`~accelerate.Accelerator.prepare`]方法。这包括您的训练和评估DataLoader、一个模型和一个优化器:

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## 反向传播

最后一步是用🤗 加速的[`~accelerate.Accelerator.backward`]方法替换训练循环中的典型`loss.backward()`:

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

如您在下面的代码中所见，您只需要添加四行额外的代码到您的训练循环中即可启用分布式训练！

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

在添加了相关代码行后，可以在脚本或笔记本（如Colaboratory）中启动训练。

### 用脚本训练

如果您从脚本中运行训练，请运行以下命令以创建和保存配置文件:

```bash
accelerate config
```

然后使用以下命令启动训练:

```bash
accelerate launch train.py
```

### 用笔记本训练

🤗 加速还可以在笔记本中运行，如果您计划使用Colaboratory的TPU，则可在其中运行。将负责训练的所有代码包装在一个函数中，并将其传递给[`~accelerate.notebook_launcher`]:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

有关🤗 加速及其丰富功能的更多信息，请参阅[文档](https://huggingface.co/docs/accelerate)。