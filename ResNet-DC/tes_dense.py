from model.PDenseNet import PDenseNet18, PDenseNet34, PDenseNet50, PDenseNet101
# from model.SANet import SANet
from tool.config import opt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tool.Loader import Loader, Normalize, ToTensor, my_collate
from SolverDen import SolverDen
from model.CSRNet import CSRNet


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
    elif m_size == 'CSRNet':
        model = CSRNet().cuda()
    if cuda:
        model.cuda()
    return model


def load_data():
    test_dataset = Loader(opt.test_iFname, opt.test_dFname,
                          transform=transforms.Compose([
                              Normalize(),
                              ToTensor()])
                          )
    testLoader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return testLoader


if __name__ == '__main__':
    net = load_net()
    # print(net)
    print('网络加载完成')
    loader = load_data()
    print('数据加载完成')
    solver = SolverDen(net, loader, model='test')
    solver.test()
