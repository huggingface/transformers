import torch

from transformers import UdopConfig, UdopForConditionalGeneration


def get_visual_bbox(image_size=224):
    image_feature_pool_shape = [image_size // 16, image_size // 16]
    visual_bbox_x = (
        torch.arange(
            0,
            1.0 * (image_feature_pool_shape[1] + 1),
            1.0,
        )
        / image_feature_pool_shape[1]
    )
    visual_bbox_y = (
        torch.arange(
            0,
            1.0 * (image_feature_pool_shape[0] + 1),
            1.0,
        )
        / image_feature_pool_shape[0]
    )
    visual_bbox_input = torch.stack(
        [
            visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
            visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
            visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
            visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
        ],
        dim=-1,
    ).view(-1, 4)
    return visual_bbox_input


config = UdopConfig()
model = UdopForConditionalGeneration(config)

# for name, param in model.named_parameters():
#     print(name, param.shape)

# let's test a forward pass
input_ids = torch.tensor([[101, 102]])
seg_data = torch.tensor([[[0, 0, 0, 0], [1, 2, 3, 4]]]).float()
image = torch.randn(1, 3, 224, 224)
visual_seg_data = get_visual_bbox().unsqueeze(0)
decoder_input_ids = torch.tensor([[101]])


print("Shape of seg_data: ", seg_data.shape)
print("Shape of visual_seg_data: ", visual_seg_data.shape)

outputs = model(
    input_ids=input_ids,
    seg_data=seg_data,
    image=image,
    visual_seg_data=visual_seg_data,
    decoder_input_ids=decoder_input_ids,
)

print("Outputs:", outputs.keys())
