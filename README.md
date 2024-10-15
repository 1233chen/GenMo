Train


python stage2_latent_vae.py


eval


python eval_cmu.py --name LVAE_AE_RCE1_KGLE1_121_YL_ML160 --gpu_id 1 --dataset_name bfa --motion_length 160 --ext cmu_NSP_IK --use_style --batch_size 12 --use_ik --niters 1
