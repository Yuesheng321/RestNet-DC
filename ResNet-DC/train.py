from model.PDenseNet import PDenseNet18, PDenseNet34, PDenseNet50, PDenseNet101, PVGG
# from model.CSRNet import CSRNet
from tool.config import opt
from torch.utils.data import DataLoader
from torchvision import transforms
from tool.Loader import Loader, Normalize, ToTensor, my_collate, RandomHorizontallyFlip
from Solver import Solver


def load_net(cuda=True):
    m_size = opt.m_size
    if m_size == '18':
        model = PDenseNet18()
    elif m_size == '34':
        model = PDenseNet34()
    elif m_size == '50':
        model = PDenseNet50()
    elif m_size == '101':
        model = PDenseNet101()
    else:
        model = PVGG()
    if cuda:
        model.cuda()
    return model


def load_data():
    train_dataset = Loader(opt.train_iFname, opt.train_pFname,
                           transform=transforms.Compose([
                               RandomHorizontallyFlip(),
                               Normalize(),
                               ToTensor()])
                           )
    val_dataset = Loader(opt.val_iFname, opt.val_pFname,
                         transform=transforms.Compose([
                             Normalize(),
                             ToTensor()])
                         )
    # if opt.batch_size == 1:
    trainLoader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    # else:
    #     trainLoader = DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn=my_collate, shuffle=True)
    valLoader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
    return [trainLoader, valLoader]


if __name__ == '__main__':
    net = load_net()
    print('网络加载完成')
    loader = load_data()
    print('数据加载完成')
    solver = Solver(net, loader, model='train')
    solver.forward2()
