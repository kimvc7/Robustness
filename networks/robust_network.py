# Hub for all the networks available
def get_network(name_net, config, num_features):

    if name_net == 'MLP' or name_net == "MLP+pgd":
        from networks.MLP import robustMLP as MLP
        return MLP(config, num_features)
    elif name_net == 'CNN' or name_net == 'Madry' or name_net == "CNN+pgd" or name_net == "CNN+clipping":
        from networks.CNN import robustCNN as CNN
        return CNN(config, num_features)
    elif name_net == 'ResNet':
        from networks.ResNet import robustResNet as ResNet
        return ResNet(config)
    elif name_net == 'OneLayer' or name_net == "OneLayer+pgd":
        from networks.OneLayer import robustOneLayer as OneLayer
        return OneLayer(config, num_features)
    elif name_net == 'ThreeLayer' or name_net == "ThreeLayer+pgd":
        from networks.OneLayer import robustOneLayer as ThreeLayer
        return ThreeLayer(config, num_features)
