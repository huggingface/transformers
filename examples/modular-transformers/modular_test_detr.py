from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrModel


# Here, the old and new model have by essence a common "detr" suffix. Make sure everything is correctly named
# in this case
class TestDetrModel(DeformableDetrModel):
    pass
