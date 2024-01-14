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
