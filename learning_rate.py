def adjust_learning_rate(optimizer, epoch,configs):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = configs['lr'] * (0.1 ** (epoch // configs['lr_decaying_period']))
    print('Learning rate:', lr)
    # for param_group in optimizer.param_groups:
    #     if args.retrain and ('mask' in param_group['key']): # retraining
    #         param_group['lr'] = 0.0
    #     elif args.prune_target and ('mask' in param_group['key']):
    #         if args.prune_target in param_group['key']:
    #             param_group['lr'] = lr
    #         else:
    #             param_group['lr'] = 0.0
    #     else:
    #         param_group['lr'] = lr
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    return lr