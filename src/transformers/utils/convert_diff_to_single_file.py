"""

running this script on `src/transformers/models/**_diff.py` should produce the equivalent single model single files

1. Iterate though `**_diff.py` files
2. How to handle the imports? 
    a. `model_type` should always be present? 
    b. `ConfigClass` should be defined as well?

3. Copy each class and function one by one.
    a. if there is a class registered for this file like `@__file__.register(MyNewClass, OldClass)`
    then copy the content of `OldClass`, replacing all names of `Old` with `MyNew`. 
    Also copy the decorators that are on top of this class. 
    b. if there is inheritance, copy non-overloaded functions from base, and overloaded from non base.

4. Register things?
new = type("new_class", (torch.nn.Linear,),{})
new__main__.new_class
new(10,10)
new_class(in_features=10, out_features=10, bias=True)

CohereConverter = ModelConverter(__file__)
CohereMLP = CohereConverter.register("CohereMLP", LlamaMLP) 

CohereMLP
<class 'transformers.models.cohere.modeling_cohere.CohereMLP'>
CohereMLP(LlamaConfig())
CohereMLP(
  (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
  (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
  (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
  (act_fn): SiLU()
)

>>> CohereMLP(LlamaConfig())(torch.ones(1,1,4096))
How to deal with submodules? 

CohereSdpaAttention(
  (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (rotary_emb): LlamaRotaryEmbedding()
)

"""
import regex as re
class ModelConverter:

    def __init__(self, file):
        self.diff_file = file
        self.model_name = re.search(r'models/(.*?)/diff', self.diff_file).group(1)
        self.modeling_file = file.replace("diff", "modeling")
        self.registered_classes = {}
        self.modules_to_import = []

    def register(self, new_class, old_class):
        # registering. Returns the old class to be usable with a new name
        self.registered_classes[new_class] = old_class
        self.modules_to_import.append([old_class, old_class.__module__])
        new_class = type(new_class, (old_class,), {})
        base_model_name = re.search(r'models\.(.*?)\.modeling', old_class.__module__).group(1)

        new_class.__module__ = re.sub(base_model_name, self.model_name, old_class.__module__)
        
        return new_class

    def __repr__(self) -> str:
        return f"ModelConverter({self.diff_file}, {self.model_name}, {self.registered_classes})"
    