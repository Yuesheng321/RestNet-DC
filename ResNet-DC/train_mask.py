from model.Mask import PDenseNet18
from tool.config import opt
from torch.utils.data import DataLoader
from torchvision import transforms
from tool.Loader import Loader, Normalize, ToTensor, my_collate, RandomHorizontallyFlip
from SolverMask import SolverMask


def load_net(cuda=True):
    model = PDenseNet18()
    if cuda:
        model.cuda()
    return model


def load_data():
    train_dataset = Loader(opt.train_iFname, opt.train_pFname, opt.train_mFname,
                           transform=transforms.Compose([
                               RandomHorizontallyFlip(),
                               Normalize(),
                               ToTensor()])
                           )
    val_dataset = Loader(opt.val_iFname, opt.val_pFname, opt.val_mFname,
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
    solver = SolverMask(net, loader, model='train')
    solver.forward()
