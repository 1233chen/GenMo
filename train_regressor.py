import os
from os.path import join as pjoin
import common.paramUtil as paramUtil
from options.train_vae_options import TrainRegressorOptions

from utils.plot_script import *

import networks.networks as Net
from networks.trainer import GMRTrainer
from data.dataset import MotionRegressorDataset, MotionBfaCMUTrainDataset
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
    parser = TrainRegressorOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:%d"%(opt.gpu_id) )
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.name = 'GLR_CV3_NP5_NS5_FT1'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin("/home/chenkerui/GenMo/data/bfa/meta")
    opt.eval_dir = pjoin(opt.save_root, "animation")
    opt.log_dir = pjoin("./log", opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name in ['bfa', 'cmu']:
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
    elif opt.dataset_name == "xia":
        opt.data_root = "../data/motion_transfer/processed_xia/"
        opt.num_of_action = len(xia_action_inv_enumerator)
        opt.num_of_style = len(xia_style_inv_enumerator)
        style_enumerator = xia_style_enumerator
        anim = BVH.load(pjoin(opt.data_root, "bvh", "angry_transitions_001.bvh"))
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
    channels = [dim_pose-4-2, 512, 256, 128, 64]

    regressor = Net.GlobalRegressor(opt.n_conv, 2, channels)

    all_params = 0
    pc_enc = sum(param.numel() for param in regressor.parameters())
    print(regressor)
    print("Total parameters of regressor net: {}".format(pc_enc))
    all_params += pc_enc
    print('Total parameters of all models: {}'.format(all_params))

    if opt.dataset_name == "cmu":
        mean = np.load(pjoin(opt.cmu_data_root, "Mean.npy"))
        std = np.load(pjoin(opt.cmu_data_root, "Std.npy"))
    else:
        mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
        std = np.load(pjoin(opt.meta_dir, "std.npy"))

    trainer = GMRTrainer(opt, regressor)
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
        train_dataset = MotionRegressorDataset(opt, mean, std, train_data_path)
        val_dataset = MotionRegressorDataset(opt, mean, std, test_data_path)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4,
                              drop_last=True, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4,
                            drop_last=True, shuffle=True, pin_memory=True)
    trainer.train(train_loader, val_loader, animation)