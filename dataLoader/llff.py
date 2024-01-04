import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
import math
from itertools import product
from .ray_utils import *


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


class LLFFDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8, lof=2, focus=0, datatype="real", preprocess_path="./preprocess", 
                 grouping="clustering", aabb=0, fmo="SML", tag=None, valid_lof=300, patch_size=8, pad=6, patch_batch=64):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        if downsample == 1.0: # synthetic
            self.hold_every = 8
        else:
            self.hold_every = 0
            filelist = os.listdir(self.root_dir)
            for f in filelist:
                if f.startswith("hold"):
                    self.hold_every = int(f.split("=")[-1])
            assert self.hold_every, "Wrong testset"
        
        self.is_stack = is_stack
        self.define_transforms()

        self.datatype = datatype

        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.focus = focus  # window size for focus measure
        self.lof = lof    # level of focus -> how many levels to use
        self.read_meta()
        self.white_bg = False

        self.near_far = [0.0, 1.0]

        if 'scene_bbox' not in dir(self):
            if aabb == 0:
                self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
            else:
                self.scene_bbox = torch.tensor([[-aabb, -aabb, -aabb], [aabb, aabb, aabb]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        
        if split == "train" and self.focus > 0:
            self.preprocess_path = preprocess_path
            self.read_focus(grouping, fmo, tag=tag, lof=lof)
            self.focus_lv = self.focus_lv.clamp(0, valid_lof)
            self.allfocus_lv_train = self.focus_lv
            self.make_windows(patch_size, pad, patch_batch)

    def make_windows(self, patch_size=8, pad=6, patch_batch=64):
        W, H = self.img_wh
        min_w = W // patch_size
        min_h = H // patch_size
        patch_batch = patch_batch
        cor_size = (patch_size + pad) ** 2
        edge_size = (patch_size + pad) * (patch_size + pad*2) 
        center_size = (patch_size + pad*2) * (patch_size + pad*2) 
        windows = []
        windows_gt = []
        for h in range(min_h):
            for w in range(min_w):
                if h == 0:
                    if w == 0:  # TL corner
                        windows.append([[0, patch_size + pad, 0, patch_size + pad], ["corner1", cor_size, None, None]])    # TL
                        windows_gt.append([0, patch_size, 0, patch_size])    # TL
                    elif w == min_w - 1: # TR corner
                        windows.append([[W - patch_size - pad, W, 0, patch_size + pad], ["corner2", cor_size, None, None]]) # TR
                        windows_gt.append([W - patch_size, W, 0, patch_size]) # TR
                    else: # T edge
                        windows.append([[w*patch_size - pad, (w+1)*patch_size + pad, 0, patch_size + pad], ["edge5", edge_size, None, None]])
                        windows_gt.append([w*patch_size, (w+1)*patch_size, 0, patch_size])
                elif h == min_h -1:
                    if w == 0: # BL corner
                        windows.append([[0, patch_size + pad, H - patch_size - pad, H], ["corner4", cor_size, None, None]])    # BL
                        windows_gt.append([0, patch_size, H - patch_size, H])    # BL
                    elif w == min_w - 1:    # BR corner
                        windows.append([[W - patch_size - pad, W, H - patch_size - pad, H], ["corner3", cor_size, None, None]]) # BR
                        windows_gt.append([W - patch_size, W, H - patch_size, H]) # BR
                    else: # B_edge
                        windows.append([[w*patch_size - pad, (w+1)*patch_size + pad, H - patch_size - pad, H], ["edge7", edge_size, None, None]])
                        windows_gt.append([w*patch_size, (w+1)*patch_size, H - patch_size, H])
                elif w == 0:    # L edge
                    windows.append([[0, patch_size + pad, h * patch_size - pad, (h+1)*patch_size + pad], ["edge8", edge_size, None, None]])
                    windows_gt.append([0, patch_size, h * patch_size, (h+1)*patch_size])
                elif w == min_w - 1 :   # R edge
                    windows.append([[W - patch_size - pad, W, h * patch_size - pad, (h+1)*patch_size + pad], ["edge6", edge_size, None, None]])
                    windows_gt.append([W - patch_size, W, h * patch_size, (h+1)*patch_size])
                else:   # center
                    windows.append([[w*patch_size - pad, (w+1)*patch_size + pad, h*patch_size - pad, (h+1)*patch_size + pad], ["center9", center_size, None, None]])
                    windows_gt.append([w*patch_size, (w+1)*patch_size, h*patch_size, (h+1)*patch_size])
        
        img_perm = np.arange(self.N_train_img)
        self.patches = list(product(*[img_perm, windows]))
        self.gts = list(product(*[img_perm, windows_gt]))


    def read_focus(self, grouping="clustering", fmo="AIF", wsize=4, tag=None, lof=128):
        if fmo == "AIF":
            self.focus_path = os.path.join(self.root_dir,f"focus_map_{fmo}_{self.focus}.npy")
        else:
            raise NotImplementedError
            
        focus_map = torch.from_numpy(np.load(self.focus_path)) # (N, H, W) or (N, H, W, 3)
        if fmo  == "AIF":
            focus_map = 1 - focus_map
        
        focus_map_train = focus_map[self.img_list]
        self.focus_map = focus_map_train

        focus_lv = torch.zeros_like(focus_map_train).to(torch.int32) # N_tr, H,W
        N_tr_img = focus_lv.shape[0]
        scene_name = self.root_dir.split("/")[-1]

        if grouping == "quantile":
            W, H = self.img_wh
            focus_lv = focus_lv.view(N_tr_img, -1).to(torch.int64)
            _, idx = focus_map_train.view(N_tr_img, -1).sort(dim=-1)
            split = math.ceil(H*W // self.lof)
            for i in range(self.lof):
                if i == self.lof-1:     # for the case where H*W / self.lof does not produce integer value.
                    mask = idx[:, i*split : ]
                else:
                    mask = idx[:, i*split : (i+1)*split]
                for j in range(N_tr_img):
                    focus_lv[j, mask[j]] = i
            self.focus_lv = focus_lv.view(-1, H, W) if fmo != "SML_seperate" else focus_lv.view(-1,H,W,3)

        elif grouping == "depthQ":
            self.focus_lv = torch.from_numpy(np.load(f"./preprocess/sharpness_level/{scene_name}.npy"))
            
            self.focus_lv = self.focus_lv[self.img_list]
            self.lof = self.focus_lv.unique().shape[0]
        else:
            assert False, "Invalid grouping. Please choose quantile or depthQ"
        self.focus_gt = focus_map[self.img_list]

    def read_meta(self, ksize=0):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        down_factor = 1 if self.datatype == "synthetic" else 4
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, f'images_{down_factor}/*')))
        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        hwf = poses[:, :, -1]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor

        # build rendering path
        N_views, N_rots = 120, 2
        tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)

        self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)

        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
        i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]
        img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))

        self.img_list = img_list
        # use first N_images-1 to train, the LAST is val
        self.all_rays = []
        self.all_rgbs = []

        # bbox
        min_bound = [100, 100, 100]
        max_bound = [-100, -100, -100]
        points = []
        near = 0; far = 1

        for i in img_list:
            image_path = self.image_paths[i]

            c2w = torch.FloatTensor(self.poses[i])

            img = Image.open(image_path).convert('RGB')
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)

            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            self.all_rgbs += [img]
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)

            def find_min_max(pt):
                for i in range(3):
                    if(min_bound[i] > pt[i]):
                        min_bound[i] = pt[i]
                    if(max_bound[i] < pt[i]):
                        max_bound[i] = pt[i]
                return
            for j in [0, W-1, H*W-W, H*W-1]:
                min_point = rays_o[j] + near*rays_d[j]
                max_point = rays_o[j] + far*rays_d[j]
                points += [min_point, max_point]
                find_min_max(min_point)
                find_min_max(max_point)

            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
            self.N_train_img = len(img_list)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

        print(self.all_rgbs.shape, self.all_rays.shape, self.split)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        cur = self.patches[idx]
        cur_gt = self.gts[idx]
        img_idx, (coord, ray_info) = cur
        w, ww, h, hh = coord 
        chunk = ray_info[1]
        img_list = img_idx
        rays_train = self.all_rays[img_idx, h:hh, w:ww].reshape(-1,6)

        w, ww, h, hh = cur_gt[1]
        rgbs_train= self.all_rgbs[img_idx, h:hh, w:ww]
        focus_lv_train = self.allfocus_lv_train[img_idx, h:hh, w:ww]

        return rays_train, rgbs_train, chunk, img_list, ray_info, focus_lv_train