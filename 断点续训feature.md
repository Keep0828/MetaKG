# 断点续训新feature

2024年1月13日开发。

# 1. 需求分析

meta-learning和finetune阶段都存在断点续训的需求。

## 1.1 meta-learning阶段需要保存的变量


    model = Recommender(n_params, args, graph, user_pre_embed, item_pre_embed).to(device)

    scheduler = Scheduler(len(names_weights_copy), grad_indexes=indexes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_update_lr)
    scheduler_optimizer = torch.optim.Adam(scheduler.parameters(), lr=args.scheduler_lr)
    
    # 因为被shuffle了，这个也需要加载进来一下。
    support_meta_set = support_meta_set[index]
    query_meta_set = query_meta_set[index]

    iter_num = math.floor(len(support_meta_set) / args.batch_size)
    train_s_t = time()
    batch_size # 只是防止batch_size发生变化
    meta_batch_size 

    #######
    meta-learning阶段的loss没有log，是不是不用加到Checkpoint里了，感觉没必要，meta-learning的loss不关心。


## 1.2 finetune阶段需要保存的变量


# 2. API设计


## 2.1 meta-learning 断点续训的plugin

需要：

1. 检查点保存函数
2. 检查点加载函数（epoch和iter的更新）
3. 先封装train函数；
4. 封装meta-learning和finetune的函数；

### 2.1.1 实施

将meta-learning模块进行封装。

```python
    if args.use_meta_model:
        model.load_state_dict(torch.load('./model_para/meta_model_{}.ckpt'.format(args.dataset), map_location=device))
    else:
        print("start meta training ...")
        """meta training"""

        if args.continue_train_ckpt:
            last_checkpoint = load_meta_checkpoint(args.continue_train_ckpt)
            if not last_checkpoint:
                raise ValueError("没有找到合法的Checkpoint，无法断点续传。考虑将args.continue_train置为空以从头训练。")
            start_iter = last_checkpoint['iter']
            model_state_dict = last_checkpoint['model_state_dict']
            optimizer_state_dict = last_checkpoint['optimizer_state_dict']
            scheduler_state_dict = last_checkpoint['scheduler_state_dict']
            scheduler_optimizer_state_dict = last_checkpoint['scheduler_optimizer_state_dict']
            args = last_checkpoint['args']
            moving_avg_reward = last_checkpoint['moving_avg_reward']

            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
            scheduler.load_state_dict(scheduler_state_dict)
            scheduler_optimizer.load_state_dict(scheduler_optimizer_state_dict)

            # 还要重新加载指定Checkpoint的support_meta_set, query_meta_set
            support_meta_set = np.load(f'{args.continue_ckpt_dir}support_meta_set_{args.dataset}_{args.continue_train_id}.npy')
            query_meta_set = np.load(f'{args.continue_ckpt_dir}query_meta_set_{args.dataset}_{args.continue_train_id}.npy')
            print(f"断点续训（{start_iter}iteration）开始...")
        else:
            # meta-training ui_interaction
            interact_mat = convert_to_sparse_tensor(mean_mat_list)
            model.interact_mat = interact_mat
            moving_avg_reward = 0
            start_iter = 0
```

support_meta_set, query_meta_set, 这两个最好在一开始就保存对应的，否则的话每一个检查点就save，太耗时。

因此，此次设计的断点续训的逻辑是在一开始判断是否执行断点续训。

由如下几个parser参数配合使用：

```python
    parser.add_argument("--continue_ckpt_dir", type=str, default="./continue_backup_ckpt/", help="continue_backup_ckpt directory for meta learning")
    parser.add_argument("--continue_train_id", type=int, default=0, help="目前用来作为保存和加载query support set的标识，理论上改变了训练参数就应该有所调整")
    parser.add_argument("--continue_train_ckpt", type=str, default="", help="这里如果指定为空，则不进行断点续训，从0开始，若指定断点保存的ckpt文件，则从此处开始进行断点续训，需搭配continue_train_id使用。")
    parser.add_argument("--need_continue_train_step", type=int, default=50, help="这里如果指定为0，则不为断点续训保存ckpt，否则每隔need_continue_train_step步保存一次ckpt。")
```

必要参数为：`continue_train_ckpt`，如果不指定，则
