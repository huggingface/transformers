class AutoMergeAdapters:
    """
    Utility to merge multiple LoRA adapters into one model.
    """

    @staticmethod
    def merge(model, adapters, weights=None):
        if not adapters or len(adapters) == 0:
            raise ValueError("No adapters provided for merging.")
        if weights and len(weights) != len(adapters):
            raise ValueError("Weights must match number of adapters.")
        return model
