from tool.config import opt
from tool.Meter import AverageMeter, Meter
from tool import trans, myLoss
from tensorboardX import SummaryWriter
from tool.mynms import localmax
from tool.eval import eval_acc
from tool import vis
import numpy as np
import torch
from tool.Loader import deNormalize
import torch.nn as nn
from torch.nn import functional as F


class SolverMask(object):
    def __init__(self, net, loader, model='train'):
        self.net = net
        self.writer = SummaryWriter(opt.logdir)
        self.model = model
        self.scale = 10
        # self.loader = loader
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
            elif opt.optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay,
                                                 momentum=opt.momentum)
            else:
                self.optimizer = None
                print('optimizer error')
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.num_epoch_lr_decay,
                                                             gamma=opt.lr_decay)
            self.record = {'best_mae': 1e20, 'best_mse': 1e20, 'pca': 0, 'gac': 0}
            self.best_ckpt_dir = opt.best_ckpt_dir
            self.loss2 = nn.MSELoss(reduction='mean')
            self.loss_mask = nn.NLLLoss(reduction='mean')
            self.writer.add_graph(self.net, input_to_model=torch.randn(1, 3, 240, 240).cuda())
            # loss = F.nll_loss(torch.log(mask), target, reduction='mean')
            if opt.reuse:
                self.load_model(path=opt.reuse_model)
        elif self.model == 'test':
            self.test_loader = loader
            self.load_model(opt.best_ckpt)

    def forward(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch += 1
            self.train()
            if self.epoch % self.val_freq == 0 or self.epoch > self.val_dense_start:
                self.val()
            if self.epoch % opt.num_epoch_lr_decay == 0:
                self.scheduler.step()
        self.writer.close()

    def train(self):
        self.net.train()
        mae_meter, mse_meter = AverageMeter(), AverageMeter()
        loss1_meter, loss2_meter, loss_meter = AverageMeter(), AverageMeter(), AverageMeter()
        cors_meter = Meter()

        print("开始第{}次训练".format(self.epoch))
        for index, sample in enumerate(self.train_loader):
            # 输入图像和真实标签
            imgs = trans.tovariable(sample['image'])
            gt_denses = trans.tovariable(sample['dense']) * self.scale
            gt_masks = trans.tovariable(sample['mask'])  # [N, H, W]
            # 预测标签
            pre_masks, pre_denses = self.net(imgs)  # shape[N, 2, H, W]
            # 计算损失
            # print(pre_masks.shape, gt_masks.shape)
            # print(pre_denses.shape, gt_denses.shape)
            loss_mask = self.loss_mask(torch.log(pre_masks), gt_masks)
            loss_mse = self.loss2(pre_denses, gt_denses)
            loss = loss_mask + loss_mse

            loss1_meter.update(loss_mask)
            loss2_meter.update(loss_mse)
            loss_meter.update(loss)
            print('loss_mask:{}, loss_mse:{}, loss:{}'.format(loss_mask.data, loss_mse.data, loss.data))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pre_masks = F.softmax(pre_masks, dim=1)
            pre_masks[pre_masks >= 0.5] = 1
            pre_masks[pre_masks < 0.5] = 0

            pre_denses = trans.tonumpy(pre_denses) / self.scale
            pre_masks = trans.tonumpy(pre_masks)
            gt_denses = trans.tonumpy(gt_denses) / self.scale
            gt_masks = trans.tonumpy(gt_masks)
            gt_masks = gt_masks[:, np.newaxis, :, :]
            # 预测位置和分数

            my_n_cors = localmax(pre_denses)
            # 真实位置和分数
            n_cors = localmax(gt_denses)

            for idx in range(len(my_n_cors)):
                error = len(my_n_cors[idx]) - len(n_cors[idx])
                mae_meter.update(np.abs(error))
                mse_meter.update(np.square(error))

            # 位置评测指标
            g_count, a_count, p_count = eval_acc(n_cors, my_n_cors, s=8)
            cors_meter.update(a_count, p_count, g_count)

        if self.epoch % self.vis_fre == 0:
            vis.vis(self.writer, pre_denses[0, 0], gt_denses[0, 0], self.epoch, model='train', mode='den')
            vis.vis(self.writer, pre_masks[0, 1], gt_masks[0, 0], self.epoch, model='train', mode='mask')

            # np.savetxt(opt.result + 'train_map/' + path + str(epoch + 1) + '_' + str(index + 1) + 'pre.csv',
            #         np_mydenses[0, 0, :, :], fmt='%.4f', delimiter=',')
        self.writer.add_scalar('train_loss1', loss1_meter.avg, self.epoch)
        self.writer.add_scalar('train_loss2', loss2_meter.avg, self.epoch)
        self.writer.add_scalar('train_loss', loss_meter.avg, self.epoch)
        self.writer.add_scalar('train_mae', mae_meter.avg, self.epoch)
        self.writer.add_scalar('train_mse', np.sqrt(mse_meter.avg), self.epoch)
        self.writer.add_scalar('train_gca', cors_meter.get_gca(), self.epoch)
        self.writer.add_scalar('train_pca', cors_meter.get_pca(), self.epoch)
        print('训练epoch:{}, mae:{}, mse:{}, gca{}, pac{}'.format(self.epoch, mae_meter.avg, np.sqrt(mse_meter.avg),
                                                                round(cors_meter.get_gca(), 2),
                                                                round(cors_meter.get_pca(), 2)))
        print('正确人数:{}, 预测人数:{}, 实际人数:{}, 平均损失:{}'.format(cors_meter.a_count, cors_meter.p_count,
                                                          cors_meter.g_count, loss_meter.avg))

    def val(self):
        self.net.eval()
        mae_meter, mse_meter = AverageMeter(), AverageMeter()
        loss1_meter, loss2_meter, loss_meter = AverageMeter(), AverageMeter(), AverageMeter()
        cors_meter = Meter()
        print("开始第{}次交叉".format(self.epoch))
        for index, sample in enumerate(self.val_loader):
            with torch.no_grad():
                # 输入图像和真实标签
                imgs = trans.tovariable(sample['image'])
                gt_denses = trans.tovariable(sample['dense']) * self.scale
                gt_masks = trans.tovariable(sample['mask'])     # [N, 1, H, W]

                # 预测标签
                pre_masks, pre_denses = self.net(imgs)  # shape[N, 2, H, W]
                # 计算损失
                loss_mask = self.loss_mask(torch.log(pre_masks), gt_masks)
                loss_mse = self.loss2(pre_denses, gt_denses)
                loss = loss_mask + loss_mse

                loss1_meter.update(loss_mask)
                loss2_meter.update(loss_mse)
                loss_meter.update(loss)
                print('loss_mask:{}, loss_mse:{}, loss:{}'.format(loss_mask.data, loss_mse.data, loss.data))

                pre_masks = F.softmax(pre_masks, dim=1)
                pre_masks[pre_masks >= 0.5] = 1
                pre_masks[pre_masks < 0.5] = 0

                pre_denses = trans.tonumpy(pre_denses) / self.scale
                pre_masks = trans.tonumpy(pre_masks)
                gt_denses = trans.tonumpy(gt_denses) / self.scale
                gt_masks = trans.tonumpy(gt_masks)
                gt_masks = gt_masks[:, np.newaxis, :, :]
                # 预测位置和分数
                my_n_cors = localmax(pre_denses)
                # 真实位置和分数
                n_cors = localmax(gt_denses)

                for idx in range(len(my_n_cors)):
                    error = len(my_n_cors[idx]) - len(n_cors[idx])
                    mae_meter.update(np.abs(error))
                    mse_meter.update(np.square(error))

                # 位置评测指标
                g_count, a_count, p_count = eval_acc(n_cors, my_n_cors, s=8)
                cors_meter.update(a_count, p_count, g_count)

        if self.epoch % self.vis_fre == 0:
            vis.vis(self.writer, pre_denses[0, 0], gt_denses[0, 0], self.epoch, model='val', mode='den')
            vis.vis(self.writer, pre_masks[0, 1], gt_masks[0, 0], self.epoch, model='val', mode='mask')
        if mae_meter.avg < self.record['best_mae']:
            self.record['best_mae'] = mae_meter.avg
            self.record['best_mse'] = np.sqrt(mse_meter.avg)
            self.record['pca'] = cors_meter.get_pca()
            self.record['gca'] = cors_meter.get_gca()
            self.save_model()
        self.writer.add_scalar('val_loss1', loss1_meter.avg, self.epoch)
        self.writer.add_scalar('val_loss2', loss2_meter.avg, self.epoch)
        self.writer.add_scalar('val_loss', loss_meter.avg, self.epoch)
        self.writer.add_scalar('val_mae', mae_meter.avg, self.epoch)
        self.writer.add_scalar('val_mse', np.sqrt(mse_meter.avg), self.epoch)
        self.writer.add_scalar('val_gca', cors_meter.get_gca(), self.epoch)
        self.writer.add_scalar('val_pca', cors_meter.get_pca(), self.epoch)
        print('交叉epoch:{}, mae:{}, mse:{}, gca{}, pac{}'.format(self.epoch, mae_meter.avg, np.sqrt(mse_meter.avg),
                                                                round(cors_meter.get_gca(), 2),
                                                                round(cors_meter.get_pca(), 2)))
        print('正确人数:{}, 预测人数:{}, 实际人数:{}, 平均损失:{}'.format(cors_meter.a_count, cors_meter.p_count,
                                                          cors_meter.g_count, loss_meter.avg))

    def test(self):
        self.net.eval()
        mae_meter, mse_meter = AverageMeter(), AverageMeter()
        cors_meter = Meter()
        for index, sample in enumerate(self.test_loader):
            with torch.no_grad():
                # 输入图像和真实标签
                imgs = trans.tovariable(sample['image'])
                gt_denses = trans.tovariable(sample['dense']) * self.scale

                # 预测标签
                pre_denses = self.net(imgs)  # shape[N, 1, H, W]

                pre_denses = trans.tonumpy(pre_denses) / self.scale
                gt_denses = trans.tonumpy(gt_denses) / self.scale

                # 预测位置和分数
                my_n_cors = localmax(pre_denses)
                # 真实位置和分数
                n_cors = localmax(gt_denses)

                my_munbers = [len(my_n_cors[idx]) for idx in range(len(my_n_cors))]
                numbers = [len(n_cors[idx]) for idx in range(len(n_cors))]
                print('预测人数:{}'.format(my_munbers))
                print('实际人数:{}'.format(numbers))

                imgs = deNormalize(np.array(imgs.cpu()))
                for idx in range(len(my_n_cors)):
                    error = len(my_n_cors[idx]) - len(n_cors[idx])
                    mae_meter.update(np.abs(error))
                    mse_meter.update(np.square(error))
                    np.savetxt(opt.result + 'test_map/' + str(index) + '_' + str(idx) + 'pre.csv', pre_denses[idx, 0],
                               fmt='%.4f', delimiter=',')
                    vis.vis_imgCor(imgs[idx], my_n_cors[idx], n_cors[idx],
                                   path=opt.result + 'test_img/' + str(index) + '_' + str(idx) + 'pre.png')

                # 位置评测指标
                g_count, a_count, p_count = eval_acc(n_cors, my_n_cors, s=8)
                cors_meter.update(a_count, p_count, g_count)
        print('mae:{}, mse:{}, gca{}, pac{}'.format(mae_meter.avg, np.sqrt(mse_meter.avg),
                                                    round(cors_meter.get_gca(), 2),
                                                    round(cors_meter.get_pca(), 2)))
        print('正确人数:{}, 预测人数:{}, 实际人数:{}'.format(cors_meter.a_count, cors_meter.p_count, cors_meter.g_count))

    def save_model(self):
        name = self.best_ckpt_dir + str(self.epoch) + '_' + str(round(self.record['best_mae'], 2)) + '_' + \
               str(round(self.record['best_mse'], 2)) + '_' + str(round(self.record['pca'], 2)) + '_' + \
               str(round(self.record['gca'], 2)) + '.pth'
        best_state = {
            'record': self.record, 'net': self.net.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(), 'epoch': self.epoch
        }
        torch.save(best_state, name)

    def load_model(self, path):
        best_state = torch.load(path)
        self.net.load_state_dict(best_state['net'])
        if self.model == 'train':
            self.record = best_state['record']
            self.epoch = best_state['epoch']
            self.optimizer.load_state_dict(best_state['optimizer'])
            self.scheduler.load_state_dict(best_state['scheduler'])
