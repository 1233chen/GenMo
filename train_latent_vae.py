import os
from os.path import join as pjoin
import common.paramUtil as paramUtil
from options.train_vae_options import TrainOptions

from utils.plot_script import *

import networks.networks as Net
from networks.trainer import LatentVAETrainer
from data.dataset import MotionDataset, MotionBfaCMUTrainDataset
from scripts.motion_process_bvh import *
from torch.utils.data import DataLoader

def animation(data, save_dir, styles):
    if opt.use_skeleton:
        data = train_dataset.deskeletonize(data)
    data =  train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        style_label = style_enumerator[styles[i]]
        joint = recover_pos_from_rot(torch.from_numpy(joint_data).float(),
                                 opt.joint_num, skeleton).numpy()
        # joint = recover_pos_from_ric(torch.from_numpy(joint_data).float(),
        #                              opt.joint_num).numpy()
        save_path = pjoin(save_dir, "%02d.mp4" %(i))
        plot_3d_motion(save_path, kinematic_chain, joint, title=style_label, fps=fps, radius=radius)

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
        checkpoint = torch.load('/home/chenkerui/GenMo/checkpoints/bfa/MAE_SMSE3_SPAE3_DZ512_DOWN2/model/best.tar', map_location=opt.device)
        # checkpoint = torch.load(pjoin(opt.checkpoints_dir, "cmu", load_name, "model", "best.tar"),
                                # map_location=opt.device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
    return encoder, decoder


def create_models(opt):
    encoder = Net.StyleContentEncoder(e_mid_channels, e_sp_channels, e_st_channels)
    generator = Net.Generator(n_conv, n_up, opt.dim_z, g_channels, dim_style)

    return encoder, generator

if __name__ == "__main__":
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
    opt.eval_dir = pjoin(opt.save_root, "animation")
    opt.log_dir = pjoin("./log", opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name in ['bfa', "cmu"]:
        opt.data_root = '/data/chenkerui1/motion_transfer_data/processed_bfa'
        opt.cmu_data_root = '/data/chenkerui1/motion_transfer_data/processed_cmu'
        opt.bfa_data_root = opt.data_root


        opt.use_action = False
        opt.num_of_action = 1
        style_enumerator = bfa_style_enumerator
        opt.num_of_style = len(bfa_style_inv_enumerator)
        anim = BVH.load(pjoin(opt.data_root, "bvh", "Hurried_02.bvh"))
        skeleton = Skeleton(anim.offsets, anim.parents, "cpu")
        # opt.motion_length = 96
    else:
        raise Exception("Unsupported data type !~")

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
    trainer = LatentVAETrainer(opt, encoder, decoder, ae_encoder, ae_decoder)
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
    trainer.train(train_loader, val_loader, animation)