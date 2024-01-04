from .tensorBase import *
import torch.nn.functional as F
import numpy as np
import os

def get_gaussian(ksize=5, sigma=1, tar_ksize=13):
    xx, yy = np.meshgrid(np.arange(ksize), np.arange(ksize))
    grid = torch.from_numpy(np.stack([xx,yy])).permute(1,2,0) - (ksize // 2)
    _const = 1 / (2 * np.pi * sigma**2)
    _exp = torch.exp((-1) * (((grid[...,0]) ** 2 + (grid[...,1]) **2) / (2 * sigma ** 2)))
    grid = _const * _exp

    if tar_ksize != 0:
        if tar_ksize != ksize:
            print("different kernel size")
            pad = (tar_ksize - ksize) // 2
            grid = F.pad(grid, (pad,pad,pad,pad))
    return grid


def normalize(tensor):
    _min = tensor.amin()
    _max = tensor.amax()
    return (tensor - _min) / (_max - _min)


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume_coarse(self, device):
        self.density_plane_coarse, self.density_line_coarse = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane_coarse, self.app_line_coarse = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat_coarse = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def init_kernel(self, device, N_train_img, channel_wise=True, img_wise=True, kcoef=1, wsize=4, sigma=1.0, focus_map=None, focus_lv=None, patch_batch=64):
        self.kernel_C = 3 if channel_wise else 1
        self.kernel_N = N_train_img if img_wise else 1
        kernel = get_gaussian(self.ksize, sigma, self.ksize)  # k, k
        kernel = kernel.repeat(self.kernel_C, 1, 1)  # C k k
        kernel = kernel.repeat(self.valid_lof, 1, 1, 1)    # lof-top_off, C, k, k
        if img_wise:
            kernel = kernel.repeat(N_train_img, 1, 1, 1, 1)  # N lof C k k
            kernel = kernel.permute(0,1,3,4,2).contiguous()  # N lof k k C
        self.kernel = torch.nn.ParameterList([torch.nn.Parameter(kernel * kcoef)]).to(device)
        self.identity = torch.zeros((self.ksize, self.ksize)).to(device)
        self.identity[self.ksize // 2, self.ksize // 2] = 1.0
        self.identity = self.identity.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        self.identity = self.identity.repeat(patch_batch, 1, 1, 1, self.kernel_C)


    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001, lr_kernel=0.05, lr_crf=0.05):
        grad_vars = [{'params': self.density_line_coarse, 'lr': lr_init_spatialxyz}, {'params': self.density_plane_coarse, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line_coarse, 'lr': lr_init_spatialxyz}, {'params': self.app_plane_coarse, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat_coarse.parameters(), 'lr':lr_init_network}]
        grad_vars += [{'params':self.renderModule_coarse.parameters(), 'lr':lr_init_network}]

        if "tonemapping" in dir(self):
            grad_vars += [{'params':self.tonemapping.parameters(), 'lr':lr_crf}]

        if self.focus > 0:
            grad_vars += [{'params': self.kernel, 'lr': lr_kernel}]
        return grad_vars


    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane_coarse)):
            total = total + torch.mean(torch.abs(self.density_plane_coarse[idx])) + torch.mean(torch.abs(self.density_line_coarse[idx]))# + torch.mean(torch.abs(self.app_plane_coarse[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane_coarse)):
            total = total + reg(self.density_plane_coarse[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane_coarse)):
            total = total + reg(self.app_plane_coarse[idx]) * 1e-2 #+ reg(self.app_line_coarse[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane_coarse)):
            plane_coef_point = F.grid_sample(self.density_plane_coarse[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line_coarse[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane_coarse)):
            plane_coef_point.append(F.grid_sample(self.app_plane_coarse[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line_coarse[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat_coarse((plane_coef_point * line_coef_point).T)


    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target_coarse):
        self.app_plane_coarse, self.app_line_coarse = self.up_sampling_VM(self.app_plane_coarse, self.app_line_coarse, res_target_coarse)
        self.density_plane_coarse, self.density_line_coarse = self.up_sampling_VM(self.density_plane_coarse, self.density_line_coarse, res_target_coarse)


        self.update_stepSize(res_target_coarse)
        print(f'upsamping to {res_target_coarse}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line_coarse[i] = torch.nn.Parameter(
                self.density_line_coarse[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line_coarse[i] = torch.nn.Parameter(
                self.app_line_coarse[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane_coarse[i] = torch.nn.Parameter(
                self.density_plane_coarse[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane_coarse[i] = torch.nn.Parameter(
                self.app_plane_coarse[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask_coarse.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))