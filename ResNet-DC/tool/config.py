class config():
    # dataName = 'BRT/'
    # dataName = 'mall_dataset/'
    # dataName = 'UCF_CC_50/'
    dataName = 'ShangHai_PartA/'
    # dataName = 'ShangHai_PartB'
    model = 'train'

    dir = 'E:/CS/'    # 项目目录
    result = 'E:/result/'                 # 结果输出目录
    my_data = dir + 'my_data/'    # 处理后数据目录
    path = my_data + dataName   # 对应数据集文件夹

    train_iFname = path + 'train/img/'
    train_dFname = path + 'train/den/'
    train_pFname = path + 'train/peak/'
    train_cFname = path + 'train/cor/'
    train_mFname = path + 'train/mask/'

    val_iFname = path + 'val/img/'
    val_mFname = path + 'val/mask/'
    val_dFname = path + 'val/den/'
    val_pFname = path + 'val/peak/'

    test_iFname = path + 'test/img/'
    test_dFname = path + 'test/den/'
    test_pFname = path + 'test/peak/'
    test_cFname = path + 'test/cor/'
    test_mFname = path + 'test/mask/'

    # pre_ckpt 文件
    vgg16 = dir + 'pre_ckpt/vgg16-397923af.pth'
    vgg16_bn = dir + 'pre_ckpt/vgg16_bn-6c64b313.pth'
    resnet18 = dir + 'pre_ckpt/resnet18-5c106cde.pth'
    resnet34 = dir + 'pre_ckpt/resnet34-333f7ec4.pth'
    resnet50 = dir + 'pre_ckpt/resnet50-19c8e357.pth'
    resnet101 = dir + 'pre_ckpt/resnet101-5d3b4d8f.pth'

    # 训练参数设置
    lr = 5e-5
    lr_decay = 0.96
    lr_decay_start = 5
    num_epoch_lr_decay = 10
    max_epoch = 100
    weight_decay = 1e-4
    batch_size = 8
    dbatch_size = 8
    optimizer = 'Adam'   # 'Adam', 'SGD'
    momentum = 0.95

    val_freq = 1
    val_dense_start = 10

    # 峰值密度图非极大抑制阈值
    threl = 0.1

    # ssim_loss 权重
    delta = 0.001

    # tensorboardX 目录
    logdir = dir + 'logdir/'

    # 可视化频率/epoch
    vis_fre = 1

    # 交叉模型保存文件夹
    best_ckpt_dir = dir + 'best_ckpt/'
    best_ckpt = best_ckpt_dir + '74_129.97_202.04_0.58_0.59.pth'

    # 训练模型保存文件夹
    train_model = dir + 'ckpt/'
    # 是否继续训练
    reuse = False
    reuse_model = None

    # 模型选择
    m_size = '18'
    if dataName == 'ShangHai_PartA/':
        amplif = 100
    else:
        amplif = 1000


opt = config()
