# -*- coding:utf-8 -*-
# @file  :zylRecoverCKPT.py
# @time  :2024/01/13
# @author:ylZhang
import os
import torch
from datetime import datetime

""" 断点续训功能 """

""" meta-learning 断点续训 """


def save_meta_checkpoint(iter, model, optimizer, scheduler, scheduler_optimizer, args, moving_avg_reward):
    checkpoint = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scheduler_optimizer_state_dict': scheduler_optimizer.state_dict(),
        'args': args,
        'moving_avg_reward': moving_avg_reward
    }
    # current_time = datetime.now()
    # formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    save_path = args.continue_ckpt_dir + 'continue_meta_model_' + args.dataset + f'_iter_{iter}_{args.continue_train_id}.ckpt'
    copy_save_path = args.continue_ckpt_dir + 'continue_meta_model_' + args.dataset + f'_iter_{iter}_{args.continue_train_id}_copy.ckpt'
    if os.path.exists(copy_save_path):
        os.remove(copy_save_path)
    os.rename(save_path, copy_save_path)
    torch.save(checkpoint, save_path)
    print(f"[Meta-learning] Continue Backup Checkpoint saved at iter {iter}: {save_path}")


def load_meta_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    else:
        return 0  # 如果检查点不存在，则从头开始训练

