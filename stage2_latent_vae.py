import os
import torch
from os.path import join as pjoin
import common.paramUtil as paramUtil
from options.train_vae_options import TrainOptions
import networks.networks as Net
from networks.trainer import LatentVAETrainer
from data.dataset import MotionDataset, MotionBfaCMUTrainDataset
from torch.utils.data import DataLoader
from common.paramUtil import bfa_style_enumerator, bfa_style_inv_enumerator, kinematic_chain
import numpy as np
import torch.optim as optim
import time
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_
from train_utils import reparametrize
from loss import kl_criterion_unit, kl_criterion
from utils.utils import print_current_loss

def swap(x):
    "Swaps the ordering of the minibatch"
    shape = x.shape
    assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
    new_shape = [shape[0]//2, 2] + list(shape[1:])
    x = x.view(*new_shape)
    x = torch.flip(x, [1])
    return x.view(*shape)

def creat_ae_models(opt):
    en_channels = [dim_pose - 4, 384, 512]
    de_channels = [opt.dim_z, 512, 384]

    encoder = Net.MotionEncoder(en_channels, opt.dim_z, vae_encoder=opt.use_vae)
    decoder = Net.MotionDecoder(de_channels, output_size=dim_pose)
    encoder.to(opt.device)
    decoder.to(opt.device)

    if not opt.is_continue:
        if opt.use_vae:
            load_name = "MVAE_KLDE3_DZ512_DOWN2"
        else:
            load_name = "MAE_SMSE3_SPAE3_DZ512_DOWN2"
        checkpoint = torch.load('/data/chenkerui1/genmo/checkpoints/cmu/MAE_SMSE3_SPAE3_DZ512_DOWN2/model/best.tar', map_location=opt.device)
        # checkpoint = torch.load(pjoin(opt.checkpoints_dir, "cmu", load_name, "model", "best.tar"),
                                # map_location=opt.device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
    return encoder, decoder


def create_models(opt):
    encoder = Net.StyleContentEncoder(e_mid_channels, e_sp_channels, e_st_channels)
    generator = Net.Generator(n_conv, n_up, opt.dim_z, g_channels, dim_style)

    return encoder, generator


parser = TrainOptions()
opt = parser.parse()

# opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:%d"%(opt.gpu_id) )
opt.device = torch.device('cuda:1')
torch.autograd.set_detect_anomaly(True)
if opt.gpu_id != -1:
    torch.cuda.set_device(opt.gpu_id)

opt.checkpoints_dir = './checkpoints'
opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
opt.model_dir = pjoin(opt.save_root, 'model')
opt.meta_dir = pjoin(opt.save_root, "meta")
opt.log_dir = pjoin("./log", opt.dataset_name, opt.name)

os.makedirs(opt.model_dir, exist_ok=True)
os.makedirs(opt.meta_dir, exist_ok=True)
# os.makedirs(opt.eval_dir, exist_ok=True)
os.makedirs(opt.log_dir, exist_ok=True)

if opt.dataset_name in ['bfa', "cmu"]:
    opt.data_root = '/data/chenkerui1/motion_transfer_data/processed_bfa'
    opt.cmu_data_root = '/data/chenkerui1/motion_transfer_data/processed_cmu'
    opt.bfa_data_root = opt.data_root


    opt.use_action = False
    opt.num_of_action = 1
    style_enumerator = bfa_style_enumerator
    opt.num_of_style = len(bfa_style_inv_enumerator)
    # opt.motion_length = 96

opt.topology = paramUtil.parents
action_dim = opt.num_of_action if opt.use_action else 0
style_dim = opt.num_of_style if opt.use_style else 0

# opt.use_skeleton = True
opt.joint_num = 21
kinematic_chain = kinematic_chain.copy()
# opt.joint_num = len(kinematic_chain)
radius = 40
fps = 30
dim_pose = 260

ae_encoder, ae_decoder = creat_ae_models(opt)
# Encoder
# 96 -> 48 -> 24
e_mid_channels = [opt.dim_z, 768]
e_sp_channels = [768+action_dim, 512]
e_st_channels = [768+style_dim, 768, 512]

dim_style = e_st_channels[-1] + style_dim
# Generator
n_conv = 2
n_up = len(e_mid_channels) - 1
g_channels = [e_sp_channels[-1]+action_dim, 768, 1024, 768, opt.dim_z]
encoder, decoder = create_models(opt)

all_params = 0
pc_enc = sum(param.numel() for param in encoder.parameters())
print(encoder)
print("Total parameters of encoder net: {}".format(pc_enc))
all_params += pc_enc

pc_gen = sum(param.numel() for param in decoder.parameters())
print(decoder)
print("Total parameters of decoder: {}".format(pc_gen))
all_params += pc_gen

pc_ae_de = sum(param.numel() for param in ae_decoder.parameters())
print(ae_decoder)
print("Total parameters of decoder: {}".format(pc_ae_de))
all_params += pc_ae_de
print('Total parameters of all models: {}'.format(all_params))

mean = np.load(pjoin(opt.cmu_data_root, "Mean.npy"))
std = np.load(pjoin(opt.cmu_data_root, "Std.npy"))

train_data_path = pjoin(opt.data_root, "train_data.npy")
test_data_path = pjoin(opt.data_root, "test_data.npy")
# trainer = LatentVAETrainer(opt, encoder, decoder, ae_encoder, ae_decoder)
if opt.dataset_name == "cmu":
    cmu_train_data_path = pjoin(opt.cmu_data_root, "train_data.npy")
    bfa_train_data_path = pjoin(opt.bfa_data_root, "train_data.npy")
    cmu_test_data_path = pjoin(opt.cmu_data_root, "test_data.npy")
    bfa_test_data_path = pjoin(opt.bfa_data_root, "test_data.npy")
    train_dataset = MotionBfaCMUTrainDataset(opt, mean, std, cmu_train_data_path, bfa_train_data_path)
    val_dataset = MotionBfaCMUTrainDataset(opt, mean, std, cmu_test_data_path, bfa_test_data_path)
else:
    train_data_path = pjoin(opt.data_root, "train_data.npy")
    test_data_path = pjoin(opt.data_root, "test_data.npy")
    train_dataset = MotionDataset(opt, mean, std, train_data_path)
    val_dataset = MotionDataset(opt, mean, std, test_data_path)
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4,
                        drop_last=True, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4,
                        drop_last=True, shuffle=True, pin_memory=True)
# trainer.train(train_loader, val_loader)

opt_encoder = optim.Adam(encoder.parameters(), lr = opt.lr)
opt_decoder = optim.Adam(decoder.parameters(), lr = opt.lr)

l1_criterion = torch.nn.SmoothL1Loss()
mse_criterion = torch.nn.MSELoss()
iter = 0

start_time = time.time()
total_iters = opt.max_epoch * len(train_loader)
print("Iters Per Epoch, Training: %04d, Validation: %03d" % (len(train_loader), len(val_loader)))
min_val_loss = np.inf

logs = OrderedDict()
for epoch in range(opt.max_epoch):
    ae_decoder.train()
    ae_encoder.train()
    encoder.train()
    decoder.train()
    ae_decoder = ae_decoder.to(opt.device)
    ae_encoder = ae_encoder.to(opt.device)
    encoder = encoder.to(opt.device)
    decoder = decoder.to(opt.device)
    for i, batch_data in enumerate(train_loader):
        if (opt.dataset_name == 'cmu'):
            M1, M2, A1, S1, SID1 = batch_data
        else:
            M1, M2, _, _, A1, S1, SID1, _, _ = batch_data
        
        A2, S2 = A1, S1   # M1 M2来自同一段motion的不同切片

        M1 = M1.permute(0, 2, 1).to(opt.device).float().detach()
        M2 = M2.permute(0, 2 ,1).to(opt.device).float().detach()
        M3 = swap(M2) # 交错排列M2,得到与M1 M2风格不同的参考motion
        SID3 = swap(SID1)

        if (opt.use_action):
            A1 = A1.to(opt.device).float().detach()
            A2 = A2.to(opt.device).float().detach()
            A3 = swap(A2)
        
        else:
            A1, A2, A3 = None, None, None

        if (opt.use_style):
            S1 = S1.to(opt.device).float().detach()
            S2 = S2.to(opt.device).float().detach()
            S3 = swap(S2)
        else:
            S1, S2, S3 = None, None, None
        
        LM1, _, _ = ae_encoder(M1[:, :-4]) # 使用的是AE
        LM2, _, _ = ae_encoder(M2[:, :-4])
        LM3 = swap(LM2)

        LM1, LM2, LM3 = LM1.detach(), LM2.detach(), LM3.detach() # 梯度阶段, 让stage2独立于stage1

        sp1, gl_mu1, gl_logvar1 = encoder(LM1, A1, S1) #sp1->content (mu,logvar)->style
        sp2, gl_mu2, gl_logvar2 = encoder(LM2, A2, S2)
        sp3, gl_mu3, gl_logvar3 = swap(sp2), swap(gl_mu2), swap(gl_logvar2)

        z_sp1 = sp1 # content2 motion1
        z_gl1 = reparametrize(gl_mu1, gl_logvar1) # style2 motion1

        z_sp2 = sp2 # content2 motion2
        z_gl2 = reparametrize(gl_mu2, gl_logvar2) # style2 motion2

        z_sp3 = z_sp2.detach() # content2
        z_gl3 = reparametrize(gl_mu3, gl_logvar3) # style3

        RLM1 = decoder(z_sp1, z_gl1, A1, S1) # 重建
        RLM2 = decoder(z_sp2, z_gl3, A2, S2) # 重建
        RLM3 = decoder(z_sp3, z_gl3, A2, S3) # 迁移

        # 循环重建
        sp4, gl_mu4, gl_logvar4 = encoder(RLM3, A2, S3) # 从迁移结果中提取content style
        z_sp4 = sp4
        z_gl4 = reparametrize(gl_mu2.detach(), gl_logvar2.detach()) #因为content是motion2 的content, 所以这里取style2, 希望重建motion2
        
        z_sp5 = sp3.detach()
        z_gl5 = reparametrize(gl_mu4, gl_logvar4)

        RRLM2 = decoder(z_sp4, z_gl4, A2, S2) # 循环重建M2
        RRLM3 = decoder(z_sp5, z_gl5, A3, S3) # 重建迁移后的风格

        # latent motion -> motion
        RM1 = ae_decoder(RLM1)
        RM2 = ae_decoder(RLM2)
        RM3 = ae_decoder(RLM3)
        RRM2 = ae_decoder(RRLM2)
        RRM3 = ae_decoder(RRLM3)

        loss_rec_lm1 = l1_criterion(LM1, RLM1)
        loss_rec_lm2 = l1_criterion(LM2, RLM2)
        loss_rec_m1 = l1_criterion(M1, RM1)
        loss_rec_m2 = l1_criterion(M2, RM2)

        loss_rec_rlm2 = l1_criterion(LM2, RRLM2)
        loss_rec_rlm3 = l1_criterion(LM3, RRLM3)
        loss_rec_rm2 = l1_criterion(M2, RRM2)
        loss_rec_rm3 = l1_criterion(M3, RRM3)

        loss_kld_gl_m1 = kl_criterion_unit(gl_mu1, gl_logvar1)
        loss_kld_gl_m2 = kl_criterion_unit(gl_mu2, gl_logvar2)
        loss_kld_gl_m4 = kl_criterion_unit(gl_mu4, gl_logvar4)
        loss_kld_gl_m12 = kl_criterion(gl_mu1, gl_logvar1, gl_mu2, gl_logvar2)

        loss = (loss_rec_lm1 + loss_rec_lm2 + loss_rec_m1 + loss_rec_m2) * opt.lambda_rec + \
                    (loss_rec_rlm2 + loss_rec_rlm3 + loss_rec_rm2 + loss_rec_rm3) * opt.lambda_rec_c + \
                    (loss_kld_gl_m1 + loss_kld_gl_m2 + loss_kld_gl_m4) * opt.lambda_kld_gl + \
                    loss_kld_gl_m12 * opt.lambda_kld_gl12

        opt_encoder.zero_grad()
        opt_decoder.zero_grad()
        loss.backward()
        clip_grad_norm_(encoder.parameters(), 0.5)
        clip_grad_norm_(decoder.parameters(), 0.5)
        opt_encoder.step()
        opt_decoder.step()

        loss_dict = OrderedDict({})
        loss_dict["loss"] = loss.item()
        loss_dict["loss_rec_lm1"] = loss_rec_lm1.item()
        loss_dict["loss_rec_lm2"] = loss_rec_lm2.item()
        loss_dict["loss_rec_m1"] = loss_rec_m1.item()
        loss_dict["loss_rec_m2"] = loss_rec_m2.item()

        loss_dict["loss_rec_rlm2"] = loss_rec_rlm2.item()
        loss_dict["loss_rec_rlm3"] = loss_rec_rlm3.item()
        loss_dict["loss_rec_rm2"] = loss_rec_rm2.item()
        loss_dict["loss_rec_rm3"] = loss_rec_rm3.item()

        loss_dict["loss_kld_gl_m1"] = loss_kld_gl_m1.item()
        loss_dict["loss_kld_gl_m2"] = loss_kld_gl_m2.item()
        loss_dict["loss_kld_gl_m4"] = loss_kld_gl_m4.item()
        loss_dict["loss_kld_gl_m12"] = loss_kld_gl_m12.item()

        iter += 1
        for k,v in loss_dict.items():
            if (k not in logs):
                logs[k] = v
            else:
                logs[k] += v

        if (iter % opt.log_every == 0):
            mean_loss = OrderedDict()
            for tag, value in logs.items():
                mean_loss[tag] = value / opt.log_every
            
            logs = OrderedDict()
            print_current_loss(start_time, iter, total_iters, mean_loss, epoch, i)

        if (iter % opt.save_latest == 0):
            state = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "ae_encoder": ae_encoder.state_dict(),
                "ae_decoder": ae_decoder.state_dict(),

                "opt_encoder": opt_encoder.state_dict(),
                "opt_decoder": opt_decoder.state_dict(),

                "ep": epoch,
                "total_it": iter,
            }
            torch.save(state, f'{opt.model_dir}/latest.tar')
    print('Validation time:')
    val_loss = None
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        ae_encoder.eval()
        ae_decoder.eval()
        for i, batch_data in enumerate(val_loader):
            if (opt.dataset_name == 'cmu'):
                M1, M2, A1, S1, SID1 = batch_data
            else:
                M1, M2, _, _, A1, S1, SID1, _, _ = batch_data
            
            A2, S2 = A1, S1   # M1 M2来自同一段motion的不同切片

            M1 = M1.permute(0, 2, 1).to(opt.device).float().detach()
            M2 = M2.permute(0, 2 ,1).to(opt.device).float().detach()
            M3 = swap(M2) # 交错排列M2,得到与M1 M2风格不同的参考motion
            SID3 = swap(SID1)

            if (opt.use_action):
                A1 = A1.to(opt.device).float().detach()
                A2 = A2.to(opt.device).float().detach()
                A3 = swap(A2)
            
            else:
                A1, A2, A3 = None, None, None

            if (opt.use_style):
                S1 = S1.to(opt.device).float().detach()
                S2 = S2.to(opt.device).float().detach()
                S3 = swap(S2)
            else:
                S1, S2, S3 = None, None, None
            
            LM1, _, _ = ae_encoder(M1[:, :-4]) # 使用的是AE
            LM2, _, _ = ae_encoder(M2[:, :-4])
            LM3 = swap(LM2)

            LM1, LM2, LM3 = LM1.detach(), LM2.detach(), LM3.detach() # 梯度阶段, 让stage2独立于stage1

            sp1, gl_mu1, gl_logvar1 = encoder(LM1, A1, S1) #sp1->content (mu,logvar)->style
            sp2, gl_mu2, gl_logvar2 = encoder(LM2, A2, S2)
            sp3, gl_mu3, gl_logvar3 = swap(sp2), swap(gl_mu2), swap(gl_logvar2)

            z_sp1 = sp1 # content2 motion1
            z_gl1 = reparametrize(gl_mu1, gl_logvar1) # style2 motion1

            z_sp2 = sp2 # content2 motion2
            z_gl2 = reparametrize(gl_mu2, gl_logvar2) # style2 motion2

            z_sp3 = z_sp2.detach() # content2
            z_gl3 = reparametrize(gl_mu3, gl_logvar3) # style3

            RLM1 = decoder(z_sp1, z_gl1, A1, S1) # 重建
            RLM2 = decoder(z_sp2, z_gl3, A2, S2) # 重建
            RLM3 = decoder(z_sp3, z_gl3, A2, S3) # 迁移

            # 循环重建
            sp4, gl_mu4, gl_logvar4 = encoder(RLM3, A2, S3) # 从迁移结果中提取content style
            z_sp4 = sp4
            z_gl4 = reparametrize(gl_mu2.detach(), gl_logvar2.detach()) #因为content是motion2 的content, 所以这里取style2, 希望重建motion2
            
            z_sp5 = sp3.detach()
            z_gl5 = reparametrize(gl_mu4, gl_logvar4)

            RRLM2 = decoder(z_sp4, z_gl4, A2, S2) # 循环重建M2
            RRLM3 = decoder(z_sp5, z_gl5, A3, S3) # 重建迁移后的风格

            # latent motion -> motion
            RM1 = ae_decoder(RLM1)
            RM2 = ae_decoder(RLM2)
            RM3 = ae_decoder(RLM3)
            RRM2 = ae_decoder(RRLM2)
            RRM3 = ae_decoder(RRLM3)

            loss_rec_lm1 = l1_criterion(LM1, RLM1)
            loss_rec_lm2 = l1_criterion(LM2, RLM2)
            loss_rec_m1 = l1_criterion(M1, RM1)
            loss_rec_m2 = l1_criterion(M2, RM2)

            loss_rec_rlm2 = l1_criterion(LM2, RRLM2)
            loss_rec_rlm3 = l1_criterion(LM3, RRLM3)
            loss_rec_rm2 = l1_criterion(M2, RRM2)
            loss_rec_rm3 = l1_criterion(M3, RRM3)

            loss_kld_gl_m1 = kl_criterion_unit(gl_mu1, gl_logvar1)
            loss_kld_gl_m2 = kl_criterion_unit(gl_mu2, gl_logvar2)
            loss_kld_gl_m4 = kl_criterion_unit(gl_mu4, gl_logvar4)
            loss_kld_gl_m12 = kl_criterion(gl_mu1, gl_logvar1, gl_mu2, gl_logvar2)

            loss = (loss_rec_lm1 + loss_rec_lm2 + loss_rec_m1 + loss_rec_m2) * opt.lambda_rec + \
                        (loss_rec_rlm2 + loss_rec_rlm3 + loss_rec_rm2 + loss_rec_rm3) * opt.lambda_rec_c + \
                        (loss_kld_gl_m1 + loss_kld_gl_m2 + loss_kld_gl_m4) * opt.lambda_kld_gl + \
                        loss_kld_gl_m12 * opt.lambda_kld_gl12

            loss_dict = OrderedDict({})
            loss_dict["loss"] = loss.item()
            loss_dict["loss_rec_lm1"] = loss_rec_lm1.item()
            loss_dict["loss_rec_lm2"] = loss_rec_lm2.item()
            loss_dict["loss_rec_m1"] = loss_rec_m1.item()
            loss_dict["loss_rec_m2"] = loss_rec_m2.item()

            loss_dict["loss_rec_rlm2"] = loss_rec_rlm2.item()
            loss_dict["loss_rec_rlm3"] = loss_rec_rlm3.item()
            loss_dict["loss_rec_rm2"] = loss_rec_rm2.item()
            loss_dict["loss_rec_rm3"] = loss_rec_rm3.item()

            loss_dict["loss_kld_gl_m1"] = loss_kld_gl_m1.item()
            loss_dict["loss_kld_gl_m2"] = loss_kld_gl_m2.item()
            loss_dict["loss_kld_gl_m4"] = loss_kld_gl_m4.item()
            loss_dict["loss_kld_gl_m12"] = loss_kld_gl_m12.item()
            if val_loss is None:
                val_loss = loss_dict
            else:
                for k, v in loss_dict.items():
                    val_loss[k] += v

    print_str = "Validation Loss:"
    for k, v in val_loss.items():
        val_loss[k] /= len(val_loader)
        print_str += ' %s: %.4f ' % (k, val_loss[k])
    print(print_str)

    if val_loss["loss"] < min_val_loss:
        min_val_loss = val_loss["loss"]
        min_val_epoch = epoch
        state = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "ae_encoder": ae_encoder.state_dict(),
            "ae_decoder": ae_decoder.state_dict(),

            "opt_encoder": opt_encoder.state_dict(),
            "opt_decoder": opt_decoder.state_dict(),

            "ep": epoch,
            "total_it": iter,
        }
        torch.save(state, f'{opt.model_dir}/best.tar')
        print("Best Validation Model So Far!~")
    