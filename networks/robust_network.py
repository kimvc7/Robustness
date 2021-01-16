# Hub for all the networks available
def get_network(name_net, config, num_features):

    if name_net == 'MLP':
        from networks.MLP import robustMLP as MLP
        return MLP(config, num_features)
    elif name_net == 'CNN':
        from networks.CNN import robustCNN as CNN
        return CNN(config)

