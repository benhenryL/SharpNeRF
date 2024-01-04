
import os
import time
import torch.nn.functional as F
import gc
import json, random
import sys
import datetime
import imageio
import math

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from opt import config_parser
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
from dataLoader import dataset_dict
from itertools import product
from dataLoader import ray_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast

@torch.no_grad()
def render_test(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    img_wise = args.grouping != "noimgwise"
    kwargs.update({'img_wise': img_wise})

    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    datatype = args.datadir.split("/")[-2].split("_")[0]    # real or synthetic
    if datatype == "real":
        args.downsample_train = 4.0
    elif datatype == "synthetic":
        args.downsample_train = 1.0
    else:
        assert False, "Invalid dataset"

    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, datatype=datatype)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    # logfolder = os.path.dirname(args.ckpt)
    logfolder = "./videos"
    os.makedirs(logfolder, exist_ok=True)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, 
        savePath= f'{logfolder}/imgs_train_all/', N_vis=-1, N_samples_coarse=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')
    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, 
        savePath=f'{logfolder}/{args.expname}/imgs_test_all/', N_vis=-1, N_samples_coarse=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples_coarse=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    datatype = args.datadir.split("/")[-2].split("_")[0]    # real or synthetic
    if datatype == "real":
        args.downsample_train = 4.0
    elif datatype == "synthetic":
        args.downsample_train = 1.0
    else:
        assert False, "Invalid dataset"

    aabb = 0

    assert args.patch_size ** 2 * args.patch_batch == args.batch_size, "batch size must be patch_size **2 * patch_batch"
        
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, lof=args.lof, focus=args.focus, datatype=datatype, grouping=args.grouping, aabb=aabb, fmo=args.fmo, tag=args.tag, valid_lof=args.lof-args.top_off, patch_size=args.patch_size, pad=args.ksize//2, patch_batch=args.patch_batch)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, lof=args.lof, focus=args.focus, datatype=datatype, grouping=args.grouping, fmo=args.fmo)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)



    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    print(aabb)
    reso_coarse = N_to_reso(args.N_voxel_init_coarse, aabb)

    nSamples_coarse = min(1e6, cal_n_samples(reso_coarse,args.step_ratio)) 
    N_train_img = train_dataset.N_train_img

    img_wise = args.grouping != "noimgwise"
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
        print("model loaded")
    else:
        reso_coarse = N_to_reso(args.N_voxel_init_coarse, aabb)
        tensorf_coarse = eval(args.model_name)(aabb, reso_coarse, device=device, ksize=args.ksize, lof=args.lof, focus=args.focus, patch_batch=args.patch_batch, tone_mapping=args.tone_mapping, kernel_type=args.kernel_type,
                    N_train_img = N_train_img, top_off=args.top_off, patch_size=args.patch_size, kernel_coef=args.kernel_coef, kernel_sigma=args.kernel_sigma, 
                    img_wise=img_wise, focus_map=train_dataset.focus_map, focus_lv=train_dataset.focus_lv, 
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)

    if tensorf_coarse.focus > 0:
        allfocus_lv = train_dataset.focus_lv
        valid_lof = args.lof - args.top_off
        del train_dataset.focus_lv
    
    print("kernel size: ", tensorf_coarse.kernel[0].shape, args.lof)
    grad_vars = tensorf_coarse.get_optparam_groups(args.lr_init, args.lr_basis, args.lr_kernel, args.lr_crf)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    if args.tv_decay_end == 50000:
        args.tv_decay_end = args.n_iters
    if args.lr_kernel_decay_end == 50000:
        args.lr_kernel_decay_end = args.n_iters

    tv_lr_factor = args.tv_decay_target_ratio**(1/(args.tv_decay_end - args.tv_decay_start))
    kernel_lr_factor = args.lr_kernel_decay_target_ratio**(1/(args.lr_kernel_decay_end - args.lr_kernel_decay_start))

    if args.lr_kernel_decay_target_ratio == 0:
        kernel_lr_factor = 0.0
        print(f"kernel freeze at {args.lr_kernel_decay_end} ~ {args.lr_kernel_decay_start}")

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)



    #linear in logrithmic space
    N_voxel_list_coarse = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init_coarse), np.log(args.N_voxel_final_coarse), len(upsamp_list)+1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs


    sample_mode = args.sample_mode
    patch_batch = args.patch_batch

    if not args.ndc_ray:
        allrays, allrgbs = tensorf_coarse.filtering_rays(allrays, allrgbs, bbox_only=True)

    patch_pad = (args.ksize - 1) // 2
    patch_pad2 = patch_pad*2
    padded_patch_size = args.patch_size + patch_pad2

    pad_mode = args.pad_mode

    W, H = train_dataset.img_wh

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)


    logging = True
    if logging:
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(logfolder + "/log.log")
        file_handler.setFormatter(formatter)
        try:
            lhStdout = logger.handlers[0]
            logger.removeHandler(lhStdout)
        except IndexError:
            pass
        logger.addHandler(file_handler)

        logger.info(args)
    
    patch_sampling = True
    blurring = True
    batch_size = args.batch_size
    test_psnrs =[]


    
    def batch_collate(batch):
        transposed = zip(*batch)
        it = iter(transposed)
        rays_train = next(it)
        rays_train = torch.concat(rays_train)
        rgbs_train = next(it)
        rgbs_train = torch.stack(rgbs_train)
        chunk = next(it)
        chunk = sum(chunk)
        img_list = next(it)
        img_list = list(img_list)
        ray_info = next(it)
        ray_info = list(ray_info)
        focus_lv_train = next(it)
        focus_lv_train = torch.stack(focus_lv_train)

        return rays_train, rgbs_train, chunk, img_list, ray_info, focus_lv_train

    train_dataset.all_rays = train_dataset.all_rays.view(train_dataset.N_train_img, H, W, 6)
    train_dataset.all_rgbs = train_dataset.all_rgbs.view(train_dataset.N_train_img, H, W, 3)
    dataloader = DataLoader(train_dataset, batch_size=args.patch_batch, num_workers=0, shuffle=True, collate_fn=batch_collate, drop_last=True)
    batch_iterator = iter(dataloader)

    res_idx = []
    st = time.time()
    for iteration in pbar:
        rays_info = []
        img_list = []
        focus_lv_train = []
        rgb_train = []
        rays_train = []
        chunk = 0
        try:
            rays_train, rgb_train, chunk, img_list, rays_info, focus_lv_train = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(dataloader)
            rays_train, rgb_train, chunk, img_list, rays_info, focus_lv_train = next(batch_iterator)
        rays_train = rays_train.to(device)
        rgb_train = rgb_train.view(-1,3).to(device)
        focus_lv_train = focus_lv_train.to(device)

        rgb_map, depth_map, sharp_rgb_map = renderer(rays_train, tensorf_coarse, blurring=blurring, img_list=img_list,            
                focus_lv=focus_lv_train, chunk=chunk, N_samples_coarse=nSamples_coarse, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True, pad_mode=pad_mode, rays_info=rays_info, top_off=args.top_off, gt=rgb_train)

        loss_coarse = torch.mean((rgb_map - rgb_train) ** 2)
        loss = loss_coarse
        loss_psnr = loss_coarse

        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf_coarse.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf_coarse.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            if iteration >= args.tv_decay_start and iteration <= args.tv_decay_end:
                TV_weight_density *= tv_lr_factor
            loss_tv = tensorf_coarse.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            if iteration >= args.tv_decay_start and iteration <= args.tv_decay_end:
                TV_weight_app *= tv_lr_factor
            loss_tv = tensorf_coarse.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss_psnr = loss_psnr.detach().item()

        
        PSNRs.append(-10.0 * np.log(loss_psnr) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss_psnr, global_step=iteration)

        for param_group in optimizer.param_groups[:-1]:
            param_group['lr'] = param_group['lr'] * lr_factor

        if iteration >= args.lr_kernel_decay_start and iteration <= args.lr_kernel_decay_end:
            optimizer.param_groups[-1]["lr"] = optimizer.param_groups[-1]["lr"] * kernel_lr_factor
        
        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss_psnr:.6f}'
            )
            if logging and iteration % 100 == 0:
                logger.info(f"Iteration {iteration:05d}: train_psnr = {float(np.mean(PSNRs)):.2f}\ttest_psnr = {float(np.mean(PSNRs_test)):.2f}")
            PSNRs = []


        if args.N_vis!=0 and (iteration in [5000, 10000, 20000]):
            if iteration >= 24000 and iteration <= 26000:
                PSNRs_test = evaluation(test_dataset,tensorf_coarse, args, renderer, savePath=f'{logfolder}/imgs_vis/', N_vis=-1,
                                        prtx=f'{iteration:06d}_', N_samples_coarse=nSamples_coarse, white_bg = white_bg, ndc_ray=ndc_ray)
            else:
                PSNRs_test = evaluation(test_dataset,tensorf_coarse, args, renderer, savePath=f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                        prtx=f'{iteration:06d}_', N_samples_coarse=nSamples_coarse, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
            test_psnrs.append(f"{float(np.mean(PSNRs_test)):.2f}")
            res_idx.append(iteration)
            tensorf_coarse.save(f'{logfolder}/{args.expname}_{iteration}.th')

        if iteration in update_AlphaMask_list:
            if reso_coarse[0] * reso_coarse[1] * reso_coarse[2]<256**3:# update volume resolution
                reso_mask = tuple(reso_coarse)

            new_aabb = tensorf_coarse.updateAlphaMask(reso_mask)
            logger.info(f"aabb: {aabb} -> {new_aabb}")
            print(f"aabb: {aabb} -> {new_aabb}")
            if iteration == update_AlphaMask_list[0]:
                tensorf_coarse.shrink(new_aabb)     
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


        if iteration in upsamp_list:
            n_voxels_coarse = N_voxel_list_coarse.pop(0)
            reso_cur_coarse = N_to_reso(n_voxels_coarse, tensorf_coarse.aabb)
            nSamples_coarse = min(1e6, cal_n_samples(reso_cur_coarse,args.step_ratio))
            tensorf_coarse.upsample_volume_grid(reso_cur_coarse)
            logger.info(f"nSamples: {nSamples_coarse}")
            print(f"iteration {iteration} - nSamples: {nSamples_coarse}")


            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf_coarse.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale, args.lr_kernel*lr_scale, args.lr_crf*lr_scale)
            if args.kernel_type in ["argmin", "argmin_patch"]:
                grad_vars = grad_vars[:-1]
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
            torch.cuda.empty_cache()

            gc.collect()
    


    ed = time.time()
    runtime = (ed - st) // 60
    logger.info(f"Training time (include testing for every 5000th epoch: {runtime} mins.")
    tensorf_coarse.save(f'{logfolder}/{args.expname}.th')


    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf_coarse, args, renderer, savePath=f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples_coarse=nSamples_coarse, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
        if logging:
            logger.info(f"test all psnr: {np.mean(PSNRs_test):.3f}")

    

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    
    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

