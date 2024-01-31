
import torch
from torch import nn, Tensor
from torchvision.ops import roi_align
from torchvision.ops.roi_align import RoIAlign, _roi_align, _bilinear_interpolate

def boundary(
    input,  # [N, C, H, W]
    y,  # [K, PH, IY]
    x,  # [K, PW, IX]
   ):
   _, channels, height, width = input.size()

   # deal with inverse element out of feature map boundary
   y = y.clamp(min=0)
   x = x.clamp(min=0)
   y_low = y.int()
   x_low = x.int()
   y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
   y_low = torch.where(y_low >= height - 1, height - 1, y_low)
   y = torch.where(y_low >= height - 1, y.to(input.dtype), y)

   x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
   x_low = torch.where(x_low >= width - 1, width - 1, x_low)
   x = torch.where(x_low >= width - 1, x.to(input.dtype), x)

   ly = y - y_low
   lx = x - x_low
   hy = 1.0 - ly
   hx = 1.0 - lx
   
   return ly, lx, hy, hx

def masked_index(
        input,  # [N, C, H, W]
        roi_batch_ind,  # [K]
        y,  # [K, PH, IY]
        x,  # [K, PW, IX]
        ymask,  # [K, IY]
        xmask,  # [K, IX]
        ):
    _, channels, height, width = input.size()
    if ymask is not None:
        assert xmask is not None
        y = torch.where(ymask[:, None, :], y, 0)
        x = torch.where(xmask[:, None, :], x, 0)
        return input[
                roi_batch_ind[:, None, None, None, None, None],
                torch.arange(channels, device=input.device)[None, :, None, None, None, None],
                y[:, None, :, None, :, None],  # prev [K, PH, IY]
                x[:, None, None, :, None, :],  # prev [K, PW, IX]
                ]  # [K, C, PH, PW, IY, IX]

def outer_prod(y, x):
    return y[:, None, :, None, :, None] * x[:, None, None, :, None, :]

def interpolate(w1, w2, w3, w4, v1, v2, v3, v4):
    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val

def roi(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    orig_dtype = input.dtype

#    input = maybe_cast(input)
#    rois = maybe_cast(rois)

    _, _, height, width = input.size()

    ph = torch.arange(pooled_height, device=input.device)  # [PH]
    pw = torch.arange(pooled_width, device=input.device)  # [PW]

    # input: [N, C, H, W]
    # rois: [K, 5]

    roi_batch_ind = rois[:, 0].int()  # [K]
    offset = 0.5 if aligned else 0.0
    roi_start_w = rois[:, 1] * spatial_scale - offset  # [K]
    roi_start_h = rois[:, 2] * spatial_scale - offset  # [K]
    roi_end_w = rois[:, 3] * spatial_scale - offset  # [K]
    roi_end_h = rois[:, 4] * spatial_scale - offset  # [K]

    # return roi_start_w, roi_start_h, roi_end_w, roi_end_h
    roi_width = roi_end_w - roi_start_w  # [K]
    roi_height = roi_end_h - roi_start_h  # [K]
    if not aligned:
        roi_width = torch.clamp(roi_width, min=1.0)  # [K]
        roi_height = torch.clamp(roi_height, min=1.0)  # [K]
#    return roi_width, roi_height
    bin_size_h = roi_height / pooled_height  # [K]
    bin_size_w = roi_width / pooled_width  # [K]

    exact_sampling = sampling_ratio > 0

    roi_bin_grid_h = sampling_ratio if exact_sampling else torch.ceil(roi_height / pooled_height)  # scalaror [K]
    roi_bin_grid_w = sampling_ratio if exact_sampling else torch.ceil(roi_width / pooled_width)  # scalar or [K]
    # return roi_bin_grid_h, roi_bin_grid_w

#    if exact_sampling:
#    count = max(roi_bin_grid_h * roi_bin_grid_w, 1)  # scalar
#    iy = torch.arange(roi_bin_grid_h, device=input.device)  # [IY]
#    ix = torch.arange(roi_bin_grid_w, device=input.device)  # [IX]
#        ymask = None
#        xmask = None
#    else:
    count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)  # [K]
        # When doing adaptive sampling, the number of samples we need to do
        # is data-dependent based on how big the ROIs are.  This is a bit
        # awkward because first-class dims can't actually handle this.
        # So instead, we inefficiently suppose that we needed to sample ALL
        # the points and mask out things that turned out to be unnecessary
    iy = torch.arange(height, device=input.device)  # [IY]
    ix = torch.arange(width, device=input.device)  # [IX]
    ymask = iy[None, :] < roi_bin_grid_h[:, None]  # [K, IY]
    xmask = ix[None, :] < roi_bin_grid_w[:, None]  # [K, IX]
    # return count, iy, ix

    y = (
            roi_start_h[:, None, None]
            + ph[None, :, None] * bin_size_h[:, None, None]
            + (iy[None, None, :] + 0.5).to(input.dtype) * (bin_size_h / roi_bin_grid_h)[:, None, None]
        )  # [K, PH, IY]
    x = (
            roi_start_w[:, None, None]
          + pw[None, :, None] * bin_size_w[:, None, None]
          + (ix[None, None, :] + 0.5).to(input.dtype) * (bin_size_w / roi_bin_grid_w)[:, None, None]
         )  # [K, PW, IX]
    return x, y

a = torch.Tensor([[i * 6 + j for j in range(6)] for i in range(6)])
print(a)
a = a.unsqueeze(dim=0)

boxes = [torch.Tensor([[0, 2, 2, 4]])]
a = a.unsqueeze(dim=0)

n = RoIAlign(output_size=2,spatial_scale=1.0,sampling_ratio=-1,aligned=False)
aligned_rois = n.forward(input=a, rois=boxes)
#torch.jit.trace(n.forward, (a, boxes))
scripted_fn = torch.jit.script(roi)
# scripted_fn = torch.jit.script(masked_index)
# scripted_fn = torch.jit.script(boundary)
# scripted_fn = torch.jit.script(outer_prod)
# scripted_fn = torch.jit.script(interpolate)
print(scripted_fn.graph)

# aligned_rois = roi_align(input=a, boxes=boxes, output_size=2)

print(aligned_rois.shape)
print("aligned_rois:", aligned_rois)
