import os
from os.path import join as pjoin
import common.paramUtil as paramUtil
from options.train_vae_options import TrainOptions

from utils.plot_script import *

import networks.networks as Net
from networks.trainer import MotionAEKLTrainer
from data.dataset import MotionBfaCMUTrainDataset
from scripts.motion_process_bvh import *
from torch.utils.data import DataLoader

def animation(data, save_dir):
    data =  train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        # style_label = style_enumerator[styles[i]]
        joint = recover_pos_from_rot(torch.from_numpy(joint_data).float(),
                                 opt.joint_num, skeleton).numpy()
        global_quat, local_quat, r_pos = recover_bvh_from_rot(torch.from_numpy(joint_data).float(),
                                                              opt.joint_num, skeleton)

        save_path = pjoin(save_dir, "%02d.mp4" %(i))
        bvh_path = pjoin(save_dir, "%02d.bvh" % (i))
        bvh_writer.write(bvh_path, local_quat.numpy(), r_pos.numpy(), order="zyx")
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


if __name__ == "__main__":
    parser = TrainOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:%d"%(opt.gpu_id) )
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)
    
    opt.name = 'MAE_SMSE3_SPAE3_DZ512_DOWN2'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.checkpoints_dir, "bfa", "VAEV3_KGLE010_YL_ML180", "meta")
    opt.eval_dir = pjoin(opt.save_root, "animation")
    opt.log_dir = pjoin("./log", opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name in ['bfa', 'cmu']:
        # opt.data_root = "../data/motion_transfer/processed_bfa"
        # opt.cmu_data_root = "../data/motion_transfer/processed_cmu"
        opt.data_root = '/data/chenkerui1/motion_transfer_data/processed_bfa'
        opt.cmu_data_root = '/data/chenkerui1/motion_transfer_data/processed_cmu'
        opt.bfa_data_root = opt.data_root
        opt.use_action = False
        opt.num_of_action = 1
        style_enumerator = bfa_style_enumerator
        opt.num_of_style = len(bfa_style_inv_enumerator)
        anim = BVH.load(pjoin(opt.data_root, "bvh", "Hurried_02.bvh"))
        skeleton = Skeleton(anim.offsets, anim.parents, "cpu")
    else:
        raise Exception("Unsupported data type !~")

    bvh_writer = BVH.WriterWrapper(anim.parents, anim.frametime, anim.offsets, anim.names)

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

    target_channel = dim_pose
    # Encoder
    # 96 -> 48 -> 24
    en_channels = [dim_pose-4, 384, 512]
    de_channels = [opt.dim_z, 512, 384]

    encoder = Net.MotionEncoder(en_channels, opt.dim_z, vae_encoder=opt.use_vae)
    decoder = Net.MotionDecoder(de_channels, output_size=dim_pose)

    all_params = 0
    pc_enc = sum(param.numel() for param in encoder.parameters())
    print(encoder)
    print("Total parameters of encoder net: {}".format(pc_enc))
    all_params += pc_enc

    pc_gen = sum(param.numel() for param in decoder.parameters())
    print(decoder)
    print("Total parameters of decoder: {}".format(pc_gen))
    all_params += pc_gen

    print('Total parameters of all models: {}'.format(all_params))

    mean = np.load(pjoin(opt.cmu_data_root, "Mean.npy"))
    std = np.load(pjoin(opt.cmu_data_root, "Std.npy"))

    trainer = MotionAEKLTrainer(opt, encoder, decoder)

    cmu_train_data_path = pjoin(opt.cmu_data_root, "train_data.npy")
    bfa_train_data_path = pjoin(opt.bfa_data_root, "train_data.npy")
    cmu_test_data_path = pjoin(opt.cmu_data_root, "test_data.npy")
    bfa_test_data_path = pjoin(opt.bfa_data_root, "test_data.npy")
    train_dataset = MotionBfaCMUTrainDataset(opt, mean, std, cmu_train_data_path, bfa_train_data_path)
    val_dataset = MotionBfaCMUTrainDataset(opt, mean, std, cmu_test_data_path, bfa_test_data_path)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4,
                              drop_last=True, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4,
                            drop_last=True, shuffle=True, pin_memory=True)
    trainer.train(train_loader, val_loader, animation)