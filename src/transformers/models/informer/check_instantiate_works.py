from transformers import InformerModel, InformerConfig

if __name__ == '__main__':
    model = InformerModel(InformerConfig())
    print(model)
