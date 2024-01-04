import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import sys
import os
from PIL import Image

class VGG19_down(nn.Module):
    def __init__(self):
        super(VGG19_down, self).__init__()
        self.VGG_MEAN = [103.939, 116.779, 123.68]

        # Convolutional layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0) # input = N,3,H,W . o/p = N,64,H,W
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)

        # Padding layers
        self.pad = nn.ReflectionPad2d(1)
        self.rep_pad = nn.ReplicationPad2d(1)

    def forward(self, imgs):
        rgb_scaled = imgs * 255.0
        red, green, blue = torch.split(rgb_scaled, 1, dim=1)
        bgr = torch.cat([blue - self.VGG_MEAN[0],
                         green - self.VGG_MEAN[1],
                         red - self.VGG_MEAN[2]], dim=1)

        # Input layer
        net_in = bgr

        # Conv1
        network = F.relu(self.conv1_1(self.pad(net_in)))
        network = F.relu(self.conv1_2(self.pad(network)))
        d0 = network
        network = F.max_pool2d(network, kernel_size=2, stride=2)

        # Conv2
        network = F.relu(self.conv2_1(self.pad(network)))
        network = F.relu(self.conv2_2(self.pad(network)))
        d1 = network
        network = F.max_pool2d(network, kernel_size=2, stride=2)

        # Conv3
        network = F.relu(self.conv3_1(self.pad(network)))
        network = F.relu(self.conv3_2(self.pad(network)))
        network = F.relu(self.conv3_3(self.pad(network)))
        network = F.relu(self.conv3_4(self.pad(network)))
        d2 = network
        network = F.max_pool2d(network, kernel_size=2, stride=2)

        # Conv4
        network = F.relu(self.conv4_1(self.pad(network)))
        network = F.relu(self.conv4_2(self.pad(network)))
        network = F.relu(self.conv4_3(self.pad(network)))
        network = F.relu(self.conv4_4(self.pad(network)))
        d3 = network
        network = F.max_pool2d(network, kernel_size=2, stride=2)

        # Conv5
        network = F.relu(self.conv5_1(self.rep_pad(network)))
        network = F.relu(self.conv5_2(self.rep_pad(network)))
        network = F.relu(self.conv5_3(self.rep_pad(network)))
        d4 = F.relu(self.conv5_4(self.rep_pad(network)))

        return d0, d1, d2, d3, d4



def lrelu(x, negative_slope=0.2):
    return nn.LeakyReLU(negative_slope)(x)
def UpSampling2dLayer_(input, scale, mode='bilinear', align_corners=False):
    return nn.Upsample(scale_factor=scale, mode=mode, align_corners=align_corners)(input)

class UNet_up(nn.Module):
    def __init__(self):
        super(UNet_up, self).__init__()

        # # Constants
        # w_init_relu = nn.init.variance_scaling_
        # w_init_sigmoid = nn.init.xavier_normal_

        # UpSampling2dLayer


        # Layers and Blocks
        self.pad = nn.ReflectionPad2d(1)
        self.rep_pad = nn.ReplicationPad2d(1)

        # d4_aux
        self.u4_aux_c1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.u4_aux_b1 = nn.BatchNorm2d(256)
        self.u4_aux_c2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0)
        self.u4_aux_b2 = nn.BatchNorm2d(1)
        self.u4_aux_act2 = nn.Sigmoid()

        self.u3_c1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=0)
        self.u3_b1 = nn.BatchNorm2d(256)
        self.u3_c2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.u3_b2 = nn.BatchNorm2d(256)
        self.u3_c3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.u3_b3 = nn.BatchNorm2d(256)

        self.u3_aux_c1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.u3_aux_b1 = nn.BatchNorm2d(128)
        self.u3_aux_c2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=0)
        self.u3_aux_b2 = nn.BatchNorm2d(1)
        self.u3_aux_act2 = nn.Sigmoid()

        self.u2_c1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=0)
        self.u2_b1 = nn.BatchNorm2d(128)
        self.u2_c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.u2_b2 = nn.BatchNorm2d(128)
        self.u2_c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.u2_b3 = nn.BatchNorm2d(128)

        self.u2_aux_c1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.u2_aux_b1 = nn.BatchNorm2d(64)
        self.u2_aux_c2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=0)
        self.u2_aux_b2 = nn.BatchNorm2d(1)
        self.u2_aux_act2 = nn.Sigmoid()

        self.u1_c1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=0)
        self.u1_b1 = nn.BatchNorm2d(64)
        self.u1_c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u1_b2 = nn.BatchNorm2d(64)
        self.u1_c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u1_b3 = nn.BatchNorm2d(64)

        self.u1_aux_c1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.u1_aux_b1 = nn.BatchNorm2d(32)
        self.u1_aux_c2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=0)
        self.u1_aux_b2 = nn.BatchNorm2d(1)
        self.u1_aux_act2 = nn.Sigmoid()

        self.u0_c_init = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.u0_b_init = nn.BatchNorm2d(64)

        self.u0_aux_c1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.u0_aux_b1 = nn.BatchNorm2d(32)
        self.u0_aux_c2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=0)
        self.u0_aux_b2 = nn.BatchNorm2d(1)
        self.u0_aux_act2 = nn.Sigmoid()

        self.u0_c_res0 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.u0_c_res1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.u0_c_res2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.u0_c_res3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.u0_c_res4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.u0_c_res5 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.u0_c_res6 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.u0_b_res0 = nn.BatchNorm2d(64)
        self.u0_b_res1 = nn.BatchNorm2d(64)
        self.u0_b_res2 = nn.BatchNorm2d(64)
        self.u0_b_res3 = nn.BatchNorm2d(64)
        self.u0_b_res4 = nn.BatchNorm2d(64)
        self.u0_b_res5 = nn.BatchNorm2d(64)
        self.u0_b_res6 = nn.BatchNorm2d(64)
        self.u0_c0_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c5_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c6_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_b0_1 = nn.BatchNorm2d(64)
        self.u0_b1_1 = nn.BatchNorm2d(64)
        self.u0_b2_1 = nn.BatchNorm2d(64)
        self.u0_b3_1 = nn.BatchNorm2d(64)
        self.u0_b4_1 = nn.BatchNorm2d(64)
        self.u0_b5_1 = nn.BatchNorm2d(64)
        self.u0_b6_1 = nn.BatchNorm2d(64)
        self.u0_c0_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_c6_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.u0_b0_2 = nn.BatchNorm2d(64)
        self.u0_b1_2 = nn.BatchNorm2d(64)
        self.u0_b2_2 = nn.BatchNorm2d(64)
        self.u0_b3_2 = nn.BatchNorm2d(64)
        self.u0_b4_2 = nn.BatchNorm2d(64)
        self.u0_b5_2 = nn.BatchNorm2d(64)
        self.u0_b6_2 = nn.BatchNorm2d(64)

        self.uf_c1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.uf_b1 = nn.BatchNorm2d(64)
        self.uf_c2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.uf_b2 = nn.BatchNorm2d(32)
        self.uf_c3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=0)


    def forward(self, feats, is_train=False):
        d0 = feats[0]
        d1 = feats[1]
        d2 = feats[2]
        d3 = feats[3]
        d4 = feats[4]

        u4 = d4 # 512
        u4 = self.u4_aux_c1(self.rep_pad(u4))   
        u4 = lrelu(self.u4_aux_b1(u4))
        u4 = self.u4_aux_act2(self.u4_aux_b2(self.u4_aux_c2(self.rep_pad(u4)))) 

        n = UpSampling2dLayer_(d4, scale=(2, 2), mode='bilinear', align_corners=True)
        n = torch.cat([n, d3], dim=1)
        n = lrelu(self.u3_b1(self.u3_c1(self.pad(n))))  
        n = lrelu(self.u3_b2(self.u3_c2(self.pad(n))))
        n = lrelu(self.u3_b3(self.u3_c3(self.pad(n))))

        u3 = lrelu(self.u3_aux_b1(self.u3_aux_c1(self.pad(n))))
        u3 = self.u3_aux_act2(self.u3_aux_b2(self.u3_aux_c2(self.pad(u3))))

        
        n = UpSampling2dLayer_(n, scale=(2, 2), mode='bilinear', align_corners=True)    
        n = torch.cat([n, d2], dim=1)   
        n = lrelu(self.u2_b1(self.u2_c1(self.pad(n))))
        n = lrelu(self.u2_b2(self.u2_c2(self.pad(n))))
        n = lrelu(self.u2_b3(self.u2_c3(self.pad(n))))

        u2 = lrelu(self.u2_aux_b1(self.u2_aux_c1(self.pad(n))))
        u2 = self.u2_aux_act2(self.u2_aux_b2(self.u2_aux_c2(self.pad(u2))))

        n = UpSampling2dLayer_(n, scale=(2, 2), mode='bilinear', align_corners=True)
        n = torch.cat([n, d1], dim=1)   
        n = lrelu(self.u1_b1(self.u1_c1(self.pad(n))))
        n = lrelu(self.u1_b2(self.u1_c2(self.pad(n))))
        n = lrelu(self.u1_b3(self.u1_c3(self.pad(n))))

        u1 = lrelu(self.u1_aux_b1(self.u1_aux_c1(self.pad(n))))
        u1 = self.u1_aux_act2(self.u1_aux_b2(self.u1_aux_c2(self.pad(u1))))

        n = UpSampling2dLayer_(n, scale=(2, 2), mode='bilinear', align_corners=True)
        n = torch.cat([n, d0], dim=1)   
        n = lrelu(self.u0_b_init(self.u0_c_init(self.pad(n))))
        
        u0 = lrelu(self.u0_aux_b1(self.u0_aux_c1(self.pad(n))))
        u0 = self.u0_aux_act2(self.u0_aux_b2(self.u0_aux_c2(self.pad(u0)))) 
        

        for i in range(7):
            n_res = n
            n_res = eval(f"self.u0_c_res{i}")(n_res)
            n_res = lrelu(eval(f"self.u0_b_res{i}")(n_res))

            n = lrelu(eval(f"self.u0_b{i}_1")(eval(f"self.u0_c{i}_1")(self.pad(n))))
            n = lrelu(eval(f"self.u0_b{i}_2")(eval(f"self.u0_c{i}_2")(self.pad(n))))
            n = n + n_res
        n = lrelu(self.uf_b1(self.uf_c1(self.pad(n))))
        n = lrelu(self.uf_b2(self.uf_c2(self.pad(n))))
        n = F.sigmoid(self.uf_c3(self.pad(n)))

        return n


def init_dme():
    vgg = VGG19_down()
    unet = UNet_up()
    W = np.load("./preprocess/dmenet_pretrained.npz")
    vgg_sd = {}
    unet_sd = {}
    for name in W.files:
        sp = name.split('/')
        if sp[0] != "defocus_net":
            continue
        if sp[1] == "encoder":
            layer = sp[2]
            if layer.startswith("conv") and layer in dir(vgg):
                _type = sp[3].split(":")[0]
                if _type == "kernel":
                    vgg_sd[f"{layer}.weight"] = torch.from_numpy(W[name]).permute(3,2,0,1)
                else:
                    vgg_sd[f"{layer}.bias"] = torch.from_numpy(W[name])
        elif sp[1] == "decoder":   # Unet
            layer = sp[2] + "_" + sp[3]
            _type = sp[4].split(":")[0]
            if _type == "kernel":
                unet_sd[f"{layer}.weight"] = torch.from_numpy(W[name]).permute(3,2,0,1)
            elif _type =="bias":
                unet_sd[f"{layer}.bias"] = torch.from_numpy(W[name])
            elif _type == "beta":
                unet_sd[f"{layer}.bias"] = torch.from_numpy(W[name])
            elif _type == "moving_variance":
                unet_sd[f"{layer}.running_var"] = torch.from_numpy(W[name])
            elif _type == "moving_mean":
                unet_sd[f"{layer}.running_mean"] = torch.from_numpy(W[name])
                unet_sd[f"{layer}.weight"] = torch.ones_like(torch.from_numpy(W[name]))

    unet.load_state_dict(unet_sd)
    vgg.load_state_dict(vgg_sd)

    return vgg, unet

def build_focus_map(vgg, unet, basedir, obj, ds):
    file_path = f"{basedir}/defocus{obj}/images_{ds}"
    files = os.listdir(file_path)
    files.sort()
    imgs = []
    for file in files:
        imgs.append(torch.from_numpy(np.array(Image.open(os.path.join(file_path, file)))))
    imgs = torch.stack(imgs)
    imgs = imgs / 255
    imgs = F.pad(imgs, (0,0,4,4), "reflect")
    imgs = imgs.permute(0,3,1,2)
    
    feat = vgg(imgs)
    focus_map = unet(feat)
    focus_map = focus_map[..., 4:-4]
    focus_map = focus_map.squeeze()

    np.save(f"./preprocess/focus_map/defocus{obj}.npy", focus_map.detach().numpy())
    

if __name__ == "__main__":
    basedir = sys.argv[1] # /home/blee/nfs/DCT/data/deblur/
    obj = sys.argv[2]

    vgg, unet = init_dme()
    vgg.eval()
    unet.eval()
    print("==Model loaded (cpu)==")

    os.makedirs("./preprocess/focus_map", exist_ok=True)
    real_dir = os.path.join(basedir, "real_defocus_blur")
    syn_dir = os.path.join(basedir, "synthetic_defocus_blur")
    real_obj = ["cake", "caps", "cisco", "coral", "cups", "cupcake", "daisy", "sausage", "seal", "tools"]
    syn_obj = ["cozy2room", "factory", "pool", "tanabata", "wine"]

    if obj == "all":
        for obj in real_obj:
            print(obj)
            build_focus_map(vgg, unet, real_dir, obj, 4)

        for obj in syn_obj:
            print(obj)
            build_focus_map(vgg, unet, syn_dir, obj, 1)

    else:
        if obj in real_obj:
            build_focus_map(vgg, unet, real_dir, obj, 4)
        elif obj in syn_obj:
            build_focus_map(vgg, unet, syn_dir, obj, 1)
        else:
            print("wrong dataset")
            raise NotImplementedError