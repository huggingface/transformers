import torch
from torch import nn

batch_size = 2
channels = 3
h, w = 12, 12
image = torch.randn(batch_size, channels, h, w) # input image

kh, kw = 3, 3 # kernel size
dh, dw = 3, 3 # stride

# Create conv
conv = nn.Conv2d(3, 7, (kh, kw), stride=(dh, dw), bias=False)
filt = conv.weight

patches = image.view(batch_size, channels, h // kh, kh, w // kw, kw)
patches = patches.permute(0, 1, 2, 4, 3, 5).reshape(batch_size, channels, -1, kh, kw)
patches = patches.permute(0, 2, 1, 3, 4).contiguous() # nb_windows, channels, kh, kw
print(patches.shape)


# Manual approach
# patches = image.unfold(2, kh, dh).unfold(3, kw, dw)
# print(patches.shape) # batch_size, channels, h_windows, w_windows, kh, kw
# patches = patches.contiguous().view(batch_size, channels, -1, kh, kw)
# print(patches.shape) # batch_size, channels, windows, kh, kw
# 
# nb_windows = patches.size(2)
# # Now we have to shift the windows into the batch dimension.
# # Maybe there is another way without .permute, but this should work
# patches = patches.permute(0, 2, 1, 3, 4)
# print(patches.shape) # batch_size, nb_windows, channels, kh, kw

res1 = (patches[0].unsqueeze(1) * filt.unsqueeze(0)).sum([2, 3, 4]).unsqueeze(0)
res1 = res1.permute(0, 2, 1) # batch_size, out_channels, output_pixels
h = w = int(res1.size(2)**0.5)
res1 = res1.reshape(1, -1, h, w)


# (bs, h*w, channels, 1, 1) -> (bs, h*w, 1, in_chan, 1, 1)
# filt (in_chan, out_chan, 1, 1) -> (1, 1, in_chan, out_chan, 1, 1)


# Calculate the conv operation manually
res = (patches.unsqueeze(2) * filt.unsqueeze(0).unsqueeze(1)).sum([3, 4, 5])
print(res.shape) # batch_size, output_pixels, out_channels
res = res.permute(0, 2, 1) # batch_size, out_channels, output_pixels
# assuming h = w
h = w = int(res.size(2)**0.5)
res = res.view(batch_size, -1, h, w)
print(res.shape, res1.shape)

# Module approach
out = conv(image)


print('max abs error ', (out - res).abs().max())
print('max abs error ', (res1 - res[:1]).abs().max())




im1 = torch.randn(1, channels, h, w)
im2 = torch.randn(1, channels, 15, 15)


patches_list = []
for image in [im1, im2]:
    patches = image.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(1, channels, -1, kh, kw)
    nb_windows = patches.size(2)
    patches = patches.permute(0, 2, 1, 3, 4)
    print(patches.shape) # batch_size, nb_windows, channels, kh, kw
    patches_list.append(patches)


patches = torch.cat(patches_list, dim=1)
print(patches.shape)

res = (patches.unsqueeze(2) * filt.unsqueeze(0).unsqueeze(1)).sum([3, 4, 5])
print(res.shape) # batch_size, output_pixels, out_channels


# Module approach
out1 = conv(im1).flatten(2).permute(0, 2, 1)
out2 = conv(im2).flatten(2).permute(0, 2, 1)
out_all = torch.cat([out1, out2], dim=1)
print(out_all.shape)

print('max abs error ', (out_all - res).abs().max())

