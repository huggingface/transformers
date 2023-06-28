from pydantic import BaseModel
from transformers.training_args import TrainingArguments

class MyTrainingArguments(TrainingArguments):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_arg = "my_arg"


class MyModel(BaseModel):
    training_args: MyTrainingArguments


model = MyModel(training_args=MyTrainingArguments(output_dir=""))
