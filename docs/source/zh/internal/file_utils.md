<!--版权所有2021年HuggingFace团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在许可证网站上获取许可证的副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以了解权限和限制的特定语言。⚠️请注意，此文件为 Markdown 格式，但包含我们文档生成器的特定语法（类似于 MDX），在 Markdown 查看器中可能无法正确渲染。
-->

# 通用实用程序


此页面列出了在 `utils.py` 文件中找到的 Transformers 通用实用函数。
如果您正在研究库中的通用代码，这些函数大多数情况下才会有用。

## 枚举和命名元组

## Enums and namedtuples

[[autodoc]] utils.ExplicitEnum
[[autodoc]] utils.PaddingStrategy
[[autodoc]] utils.TensorType

## 特殊装饰器
[[autodoc]] utils.add_start_docstrings
[[autodoc]] utils.add_start_docstrings_to_model_forward
[[autodoc]] utils.add_end_docstrings
[[autodoc]] utils.add_code_sample_docstrings
[[autodoc]] utils.replace_return_docstrings
## 特殊属性
[[autodoc]] utils.cached_property

## 其他实用程序
[[autodoc]] utils._LazyModule