from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrModel


# Here, the old and new model have by essence a common "detr" suffix. Make sure everything is correctly named
# in this case (i.e., we do not wrongly detect `Detr` as part of a suffix to remove)
class TestDetrModel(DeformableDetrModel):
    pass
