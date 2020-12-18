from tool.config import opt
from tool.Meter import AverageMeter
from tool import trans
from tensorboardX import SummaryWriter
from tool.vis import vis, vis_heat
from tool.Loader import deNormalize
import numpy as np
import torch.nn as nn
import torch


class SolverDen(object):
    def __init__(self, net, loader, model='train'):
        self.net = net
        self.writer = SummaryWriter(opt.logdir)
        self.model = model
        self.amplif = opt.amplif
        if self.model == 'train':
            self.epoch = 0
            self.vis_fre = opt.vis_fre
            self.val_freq = opt.val_freq
            self.val_dense_start = opt.val_dense_start
            self.lr_decay_start = opt.lr_decay_start
            self.max_epoch = opt.max_epoch
            self.train_loader, self.val_loader = loader
            if opt.optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.num_epoch_lr_decay,
                                                             gamma=opt.lr_decay)
            self.criteria = torch.nn.MSELoss(reduction='mean')

            self.record = {'best_mae': 1e20, 'best_mse': 1e20}
            self.best_ckpt_dir = opt.best_ckpt_dir
            # self.writer.add_graph(self.net, input_to_model=torch.randn(1, 3, 1024, 768).cuda())

            if opt.reuse:
                self.load_model(path=opt.reuse_model)
        elif self.model == 'test':
            self.test_loader = loader
            self.load_model(path=opt.best_ckpt, model=self.model)

    def forward(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch += 1
            self.train()

            if self.epoch >= self.lr_decay_start:
                self.scheduler.step()
            if self.epoch % self.val_freq == 0 or self.epoch > self.val_dense_start:
                self.val()

    def train(self):
        self.net.train()
        mae_meter, mse_meter = AverageMeter(), AverageMeter()
        loss_meter = AverageMeter()
        # cors_meter = Meter()

        print("开始第{}次训练".format(self.epoch))
        for index, sample in enumerate(self.train_loader):
            # 输入图像和真实标签
            imgs = trans.tovariable(sample['image'])
            gt_denses = trans.tovariable(sample['dense']) * self.amplif

            # 预测标签
            pre_denses = self.net(imgs)  # shape[N, 1, H, W]

            # 计算损失
            loss = self.criteria(input=pre_denses, target=gt_denses)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss)
            print('loss:{}'.format(loss.data))

            pre_denses = trans.tonumpy(pre_denses/self.amplif)
            gt_denses = trans.tonumpy(gt_denses/self.amplif)

            pre_number = np.sum(pre_denses, axis=(1, 2, 3))     # 预测人数
            gt_number = np.sum(gt_denses, axis=(1, 2, 3))       # 真实人数
            # print(pre_number, gt_number)

            for idx in range(gt_number.shape[0]):
                error = pre_number[idx] - gt_number[idx]
                mae_meter.update(np.abs(error))
                mse_meter.update(np.square(error))

        if self.epoch % self.vis_fre == 0:
            vis(self.writer, pre_denses[0, 0], gt_denses[0, 0], self.epoch, model='train')
            np.savetxt(opt.result + 'train_map/' + str(self.epoch) + '_' + 'pre.csv', pre_denses[0, 0], fmt='%.4f',
                       delimiter=',')
        self.writer.add_scalar('train_loss', loss_meter.avg, self.epoch)
        self.writer.add_scalar('train_mae', mae_meter.avg, self.epoch)
        self.writer.add_scalar('train_mse', np.sqrt(mse_meter.avg), self.epoch)
        print('训练epoch:{}, mae:{}, mse:{}'.format(self.epoch, mae_meter.avg, np.sqrt(mse_meter.avg)))

    def val(self):
        self.net.eval()
        mae_meter, mse_meter = AverageMeter(), AverageMeter()
        loss_meter = AverageMeter()
        print("开始第{}次交叉".format(self.epoch))
        for index, sample in enumerate(self.val_loader):
            with torch.no_grad():
                # 输入图像和真实标签
                imgs = trans.tovariable(sample['image'])
                gt_denses = trans.tovariable(sample['dense'])*self.amplif

                # 预测标签
                pre_denses = self.net(imgs)  # shape[N, 1, H, W]

                # 计算损失
                loss = self.criteria(input=pre_denses, target=gt_denses)

                loss_meter.update(loss)
                print('loss:{}'.format(loss.data))

                pre_denses = trans.tonumpy(pre_denses/self.amplif)
                gt_denses = trans.tonumpy(gt_denses/self.amplif)

                pre_number = np.sum(pre_denses, axis=(1, 2, 3))  # 预测人数
                gt_number = np.sum(gt_denses, axis=(1, 2, 3))  # 真实人数

                for idx in range(gt_number.shape[0]):
                    error = pre_number[idx] - gt_number[idx]
                    mae_meter.update(np.abs(error))
                    mse_meter.update(np.square(error))

        if self.epoch % self.vis_fre == 0:
            vis(self.writer, pre_denses[0, 0], gt_denses[0, 0], self.epoch, model='val')
            np.savetxt(opt.result + 'val_map/' + str(self.epoch) + '_' + 'pre.csv', pre_denses[0, 0], fmt='%.4f',
                       delimiter=',')
        if mae_meter.avg < self.record['best_mae'] or np.sqrt(mse_meter.avg) < self.record['best_mse']:
            self.record['best_mae'] = mae_meter.avg
            self.record['best_mse'] = np.sqrt(mse_meter.avg)
            self.save_model()

        self.writer.add_scalar('val_loss', loss_meter.avg, self.epoch)
        self.writer.add_scalar('val_mae', mae_meter.avg, self.epoch)
        self.writer.add_scalar('val_mse', np.sqrt(mse_meter.avg), self.epoch)
        print('交叉epoch:{}, mae:{}, mse:{}'.format(self.epoch, mae_meter.avg, np.sqrt(mse_meter.avg)))

    def test(self):
        self.net.eval()
        mae_meter, mse_meter = AverageMeter(), AverageMeter()
        for index, sample in enumerate(self.test_loader):
            with torch.no_grad():
                # 输入图像和真实标签
                imgs = trans.tovariable(sample['image'])
                gt_denses = trans.tovariable(sample['dense'])

                # 预测标签
                pre_denses = self.net(imgs)  # shape[N, 1, H, W]

                pre_denses = trans.tonumpy(pre_denses)/self.amplif
                gt_denses = trans.tonumpy(gt_denses)

                pre_number = np.sum(pre_denses, axis=(1, 2, 3))  # 预测人数
                gt_number = np.sum(gt_denses, axis=(1, 2, 3))  # 真实人数

                print('预测人数:{}'.format(pre_number.tolist()))
                print('实际人数:{}'.format(gt_number.tolist()))

                for idx in range(gt_number.shape[0]):
                    error = pre_number[idx] - gt_number[idx]
                    mae_meter.update(np.abs(error))
                    mse_meter.update(np.square(error))
                    # np.savetxt(opt.result + 'test_map/' + str(index) + '_' + str(idx) + 'pre.csv', pre_denses[idx, 0],
                    #            fmt='%.4f', delimiter=',')
                    vis_heat(pre_denses[idx, 0], path=opt.result + 'test_map/' + str(index) + '_' + str(idx) + 'pre.png')
                    vis_heat(gt_denses[idx, 0], path=opt.result + 'test_map/' + str(index) + '_' + str(idx) + 'gt.png')
        print('mae:{}, mse:{}'.format(mae_meter.avg, np.sqrt(mse_meter.avg)))

    def save_model(self):
        name = self.best_ckpt_dir + str(self.epoch) + '_' + str(round(self.record['best_mae'], 2)) + '_' + \
               str(round(self.record['best_mse'], 2)) + '.pth'
        best_state = {
            'record': self.record, 'net': self.net.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(), 'epoch': self.epoch
        }
        torch.save(best_state, name)

    def load_model(self, path, model='train'):
        best_state = torch.load(path)
        self.net.load_state_dict(best_state['net'])
        if model == 'train':
            self.record = best_state['record']
            self.epoch = best_state['epoch']
            self.optimizer.load_state_dict(best_state['optimizer'])
            self.scheduler.load_state_dict(best_state['scheduler'])
