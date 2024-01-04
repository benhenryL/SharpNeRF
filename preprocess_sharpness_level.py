import torch
import numpy as np
import os
import sys


def build_sharpness_level(obj, dQ=10, fQ=40):
    N_lv = dQ * fQ
    depth_map = torch.from_numpy(np.load(f"./preprocess/depth_map/defocus{obj}.npy"))
    focus_map = torch.from_numpy(np.load(f"./preprocess/focus_map/defocus{obj}.npy"))
    focus_map = 1 - focus_map
    _max = depth_map.amax(dim=-1).amax(dim=-1)
    _min = depth_map.amin(dim=-1).amin(dim=-1)
    norm_depth = (depth_map - _min[:,None,None]) / (_max[:,None,None] - _min[:,None,None])

    _max = focus_map.amax(dim=-1).amax(dim=-1)
    _min = focus_map.amin(dim=-1).amin(dim=-1)
    norm_focus = (focus_map - _min[:,None,None]) / (_max[:,None,None] - _min[:,None,None])

    N = norm_depth.shape[0]
    focus_lv = torch.ones_like(depth_map, dtype=torch.int64) * (N_lv + 1)
    focus_lv = focus_lv.view(N,-1)
    
    dS = (400*600) // dQ
    fS = dS // fQ
    norm_depth = norm_depth.view(N,-1)
    norm_focus = norm_focus.view(N,-1)
    val, idx = norm_depth.sort(dim=-1)
    for i in range(dQ):
        if i != dQ-1:
            depth_mask = idx[:, i*dS : (i+1)*dS]
        else:
            depth_mask = idx[:, i*dS :]
        tar = torch.gather(norm_focus, -1, depth_mask)
        f_val, f_idx = tar.sort(dim=-1)
        for j in range(fQ):
            if j != fQ-1:
                _focus_mask = f_idx[:, j*fS: (j+1)*fS]
            else:
                _focus_mask = f_idx[:, j*fS:]
            focus_mask =  torch.gather(depth_mask, -1, _focus_mask)
            focus_lv.scatter_(-1,focus_mask,i*fQ + j)
    if N_lv + 1 in focus_lv.unique().tolist():
        raise NotImplementedError

    N = focus_lv.shape[0]
    owow = []
    focus_map = focus_map.view(N,-1)
    for i in range(N_lv):
        mask = focus_lv == i
        cur_focus = focus_map[mask].view(N,-1)
        cur_focus = cur_focus.mean(dim=-1)    # N
        owow.append(cur_focus)
    
    owow2 = torch.stack(owow).permute(1,0)   # N, N_lv
    val,idx = owow2.sort(dim=-1)
    focus_lv_sort = torch.ones_like(focus_lv, dtype=torch.int64) * (N_lv + 1)
    for i in range(N_lv):
        cur = idx[:, i]
        mask = focus_lv == cur[:,None]
        focus_lv_sort[mask] = i
    focus_lv_sort = focus_lv_sort.view(N,400,600)
    if N_lv + 1 in focus_lv_sort.unique().tolist():
        raise NotImplementedError
    
    np.save(f"./preprocess/sharpness_level/defocus{obj}.npy", focus_lv_sort)


os.makedirs("./preprocess/sharpness_level", exist_ok=True)
obj = sys.argv[1]

if obj == "all":
    for obj in ["cake", "caps", "cisco", "coral", "cups", "cupcake", "daisy", "sausage", "seal", "tools", "cozy2room", "factory", "pool", "tanabata", "wine"]:
        print(obj)
        build_sharpness_level(obj)
else:
    build_sharpness_level(obj)