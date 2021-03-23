# Hub for all the networks available
def get_network(name_net, config, num_features):

    if name_net == 'MLP' or name_net == "MLP+pgd":
        from networks.MLP import robustMLP as MLP
        return MLP(config, num_features)
    elif name_net == 'CNN' or name_net == 'Madry' or name_net == "CNN+pgd" or name_net == "CNN+clipping":
        from networks.CNN import robustCNN as CNN
        return CNN(config)
    elif name_net == 'ResNet':
        from networks.ResNet import robustResNet as ResNet
        return ResNet(config)

