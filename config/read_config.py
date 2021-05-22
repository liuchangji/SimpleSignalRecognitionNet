def parse_cfg():
    """
    将yaml文件转为字典类型
    """
    cfgfile = "/home/tuxiang/theCode/SignalRecognitionNet/CNN/config/my_net.yaml"
    import yaml
    with open(cfgfile, 'r') as f:
        net_structure_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for i in net_structure_cfg:
        print(i,":",net_structure_cfg[str(i)])

    return net_structure_cfg
