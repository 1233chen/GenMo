import os
from os.path import join as pjoin
import common.paramUtil as paramUtil
from options.train_vae_options import TrainRegressorOptions

from utils.plot_script import *

import networks.networks as Net
from networks.trainer import ActionClassifierTrainer
from data.dataset import MotionXiaDataset
from scripts.motion_process_bvh import *
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = TrainRegressorOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:%d"%(opt.gpu_id) )
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin("./data/bfa/meta")
    opt.eval_dir = pjoin(opt.save_root, "animation")
    opt.log_dir = pjoin("./log", opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 'bfa':
        opt.data_root = "../data/motion_transfer/processed_bfa"
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
    classifier = Net.GRUClassifier(dim_pose-4, opt.num_of_action, 512)

    all_params = 0
    pc_enc = sum(param.numel() for param in classifier.parameters())
    print(classifier)
    print("Total parameters of classifier net: {}".format(pc_enc))
    all_params += pc_enc
    print('Total parameters of all models: {}'.format(all_params))

    opt.batch_size =1
    mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(opt.meta_dir, "std.npy"))
    train_data_path = pjoin(opt.data_root, "train_data.npy")
    test_data_path = pjoin(opt.data_root, "test_data.npy")
    trainer = ActionClassifierTrainer(opt, classifier)
    train_dataset = MotionXiaDataset(opt, mean, std, train_data_path, fix_bias=True)
    val_dataset = MotionXiaDataset(opt, mean, std, test_data_path, fix_bias=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=1,
                              drop_last=True, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=1,
                            drop_last=True, shuffle=True, pin_memory=True)
    trainer.train(train_loader, val_loader)