import cv2,torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.signal
import torch.fft
import imageio
import os
import scipy
import kornia
from PIL import Image
from typing import Tuple, Optional
from skimage import metrics

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def get_gaussian(ksize=5):
    xx, yy = np.meshgrid(np.arange(ksize), np.arange(ksize))
    grid = torch.from_numpy(np.stack([xx,yy])).permute(1,2,0) - (ksize // 2)
    grid = grid**2 / 2
    grid = grid.sum(-1) * (-1)
    grid = torch.exp(grid)
    return grid # ksize x ksize

def SML_torch(img, kx, ky, conv, ksize):
    '''
    img: BxCxHxW
    kx, ky:  1x3x3
    ksize: For BoxBlur

    return: BxCxHxW

    '''

    mx = abs(kornia.filters.filter2d(img, kx, normalized=False))
    my = abs(kornia.filters.filter2d(img, ky, normalized=False))
    ml_img = mx + my
    
    sml = conv(ml_img) * ksize * ksize
    return sml

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)




__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''

def rgb_ssim_nerf(im1t: torch.Tensor, im2t: torch.Tensor,
                       metric="mse", margin=0, mask=None):
    """
    im1t, im2t: torch.tensors with batched imaged shape, range from (0, 1)
    """
    photometric= metrics.structural_similarity

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1:
            mask = mask.expand(-1, 3, -1, -1)
        mask = mask.permute(0, 2, 3, 1).numpy()
        batchsz, hei, wid, _ = mask.shape
        if margin > 0:
            marginh = int(hei * margin) + 1
            marginw = int(wid * margin) + 1
            mask = mask[:, marginh:hei - marginh, marginw:wid - marginw]

    # convert from [0, 1] to [-1, 1]
    im1t = (im1t * 2 - 1).clamp(-1, 1)
    im2t = (im2t * 2 - 1).clamp(-1, 1)

    if im1t.dim() == 3:
        im1t = im1t.unsqueeze(0)
        im2t = im2t.unsqueeze(0)
    im1t = im1t.detach().cpu()
    im2t = im2t.detach().cpu()

    if im1t.shape[-1] == 3:
        im1t = im1t.permute(0, 3, 1, 2)
        im2t = im2t.permute(0, 3, 1, 2)

    im1 = im1t.permute(0, 2, 3, 1).numpy()
    im2 = im2t.permute(0, 2, 3, 1).numpy()
    batchsz, hei, wid, _ = im1.shape
    if margin > 0:
        marginh = int(hei * margin) + 1
        marginw = int(wid * margin) + 1
        im1 = im1[:, marginh:hei - marginh, marginw:wid - marginw]
        im2 = im2[:, marginh:hei - marginh, marginw:wid - marginw]
    values = []

    for i in range(batchsz):
        value, ssimmap = photometric(im1[i], im2[i], multichannel=True, full=True, channel_axis=-1)
        if mask is not None:
            value = (ssimmap * mask[i]).sum() / mask[i].sum()

    return value

import torch.nn as nn
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
