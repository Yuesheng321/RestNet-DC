import torch
import torch.nn as nn
import numpy as np
from tool import trans
import torch.nn.functional as F


def loss1(pre_denses, gt_denses, gt_cors):
    '''
    峰值或者位置损失
    :param pre_denses:
    :param gt_denses:
    :param gt_cors:
    :return:
    '''
    copy_pre = pre_denses.cpu().data.numpy()
    copy_gt = gt_denses.cpu().data.numpy()

    gt_mask = np.zeros_like(copy_gt, dtype=bool)
    for i in range(copy_gt.shape[0]):
        gt_cor = gt_cors[i]
        gt_mask[i, 0, gt_cor[:, 1], gt_cor[:, 0]] = True

    pre_denses_with_borders = np.pad(copy_pre, [(0, 0), (0, 0), (2, 2), (2, 2)], mode='constant')
    pre_denses_center = pre_denses_with_borders[:, :, 1:pre_denses_with_borders.shape[2] - 1,
                        1:pre_denses_with_borders.shape[3] - 1]
    pre_denses_left = pre_denses_with_borders[:, :, 1:pre_denses_with_borders.shape[2] - 1,
                      2:pre_denses_with_borders.shape[3]]
    pre_denses_right = pre_denses_with_borders[:, :, 1:pre_denses_with_borders.shape[2] - 1,
                       0:pre_denses_with_borders.shape[3] - 2]
    pre_denses_up = pre_denses_with_borders[:, :, 2:pre_denses_with_borders.shape[2],
                    1:pre_denses_with_borders.shape[3] - 1]
    pre_denses_down = pre_denses_with_borders[:, :, 0:pre_denses_with_borders.shape[2] - 2,
                      1:pre_denses_with_borders.shape[3] - 1]
    pre_mask = (pre_denses_center > pre_denses_left) & \
               (pre_denses_center > pre_denses_right) & \
               (pre_denses_center > pre_denses_up) & \
               (pre_denses_center > pre_denses_down)

    pre_mask = pre_mask[:, :, 1:pre_denses_center.shape[2] - 1, 1:pre_denses_center.shape[3] - 1]

    C_head = 1
    # 预测峰值掩码异或真实峰值掩码
    mask = np.bitwise_xor(pre_mask, gt_mask)
    mask = mask.astype(np.float)

    mask = C_head * torch.tensor(mask, dtype=torch.float, device='cuda')
    error = (pre_denses - gt_denses) * (pre_denses - gt_denses)
    # print(mask)
    # print(gt_denses)
    loss1 = torch.mean(torch.sum(mask * error, dim=(1, 2, 3)))  # 1.对预测错误的峰值做出惩罚  # 2.对未预测到的真实峰值做出惩罚
    return loss1


def loss2(pre_denses, gt_denses):
    '''
    带权重的均方损失
    :param pre_denses:
    :param gt_denses:
    :return:
    '''
    # copy_pre = pre_denses.data.numpy()
    head_mask = gt_denses.cpu().data.numpy()
    head_mask[head_mask > 0] = 1
    head_mask[head_mask <= 0] = 0
    no_head_mask = 1 - head_mask

    C_head = 5
    C_no_head = 1

    mse_mask = (C_head * head_mask + C_no_head * no_head_mask)  # 降低非头部区域的梯度,提高头部区域损失
    mse_mask = mse_mask.astype(np.float)
    mse_mask = torch.tensor(mse_mask, dtype=torch.float, device='cuda')
    error = (pre_denses - gt_denses) * (pre_denses - gt_denses)
    loss2 = torch.mean(torch.sum(mse_mask * error, dim=(1, 2, 3)))  # 均方损失
    return loss2


def loss2_mean(pre_denses, gt_denses):
    '''
    带权重的均方损失
    :param pre_denses:
    :param gt_denses:
    :return:
    '''
    # copy_pre = pre_denses.data.numpy()
    head_mask = gt_denses.cpu().data.numpy()
    head_mask[head_mask > 0] = 1
    head_mask[head_mask <= 0] = 0
    no_head_mask = 1 - head_mask

    C_head = 5
    C_no_head = 1

    mse_mask = (C_head * head_mask + C_no_head * no_head_mask)  # 降低非头部区域的梯度,提高头部区域损失
    mse_mask = mse_mask.astype(np.float)
    mse_mask = torch.tensor(mse_mask, dtype=torch.float, device='cuda')
    error = (pre_denses - gt_denses) * (pre_denses - gt_denses)
    loss2 = torch.mean(mse_mask * error)     # 均方损失
    return loss2


def get_dists(pre_cors, gt_cors):
    '''
    :param pre_cors: list wiht
    :param gt_cors: list with length
    :return: shape；（n_pre_cors, n_gt_cors）
    '''
    pre_cors = np.array(pre_cors)
    gt_cors = np.array(gt_cors)
    dis = -2*np.dot(pre_cors, gt_cors.T) + np.sum(np.square(gt_cors), axis=1) + np.transpose([np.sum(np.square(pre_cors), axis = 1)])
    return dis


def get_mask(pre_cors, gt_cors, dis, shape):
    mask = np.zeros(shape=[len(dis), shape[0], shape[1]])
    for idx in range(len(dis)):
        # 预测到的位置阈值范围类不存在任何峰值
        pre_min = np.min(dis[idx], axis=1)
        pre_error_cor = np.array(pre_cors[idx])[pre_min > 1]
        mask[idx, pre_error_cor[:, 0], pre_error_cor[:, 1]] = 1

        # 将每个真实位置对应最小距离的预测位置与阈值比较，若大于阈值，则惩罚。表示该真实位置没有预测到
        gt_min = np.min(dis[idx], axis=0)
        # print(np.array(gt_cors[idx])[gt_min > 1])
        gt_error_cor = np.array(gt_cors[idx])[gt_min > 1]
        mask[idx, gt_error_cor[:, 0], gt_error_cor[:, 1]] = 1
    return mask


def loss1_rubust(pre_denses, gt_denses, gt_cors):
    '''
    峰值或者位置损失
    :param pre_denses:Tensor with shape[N, 1, H, W]
    :param gt_denses:Tensor with shape[N, 1, H, W]
    :param gt_cors:list with length n, each element is a list [list, list..., list] n
    :return:
    '''
    copy_pre = pre_denses.cpu().data.numpy()
    copy_gt = gt_denses.cpu().data.numpy()

    gt_mask = np.zeros_like(copy_gt, dtype=bool)
    for i in range(copy_gt.shape[0]):
        gt_cor = np.array(gt_cors[i])  # list to array
        gt_mask[i, 0, gt_cor[:, 1], gt_cor[:, 0]] = True

    pre_denses_with_borders = np.pad(copy_pre, [(0, 0), (0, 0), (2, 2), (2, 2)], mode='constant')
    pre_denses_center = pre_denses_with_borders[:, :, 1:pre_denses_with_borders.shape[2] - 1,
                        1:pre_denses_with_borders.shape[3] - 1]
    pre_denses_left = pre_denses_with_borders[:, :, 1:pre_denses_with_borders.shape[2] - 1,
                      2:pre_denses_with_borders.shape[3]]
    pre_denses_right = pre_denses_with_borders[:, :, 1:pre_denses_with_borders.shape[2] - 1,
                       0:pre_denses_with_borders.shape[3] - 2]
    pre_denses_up = pre_denses_with_borders[:, :, 2:pre_denses_with_borders.shape[2],
                    1:pre_denses_with_borders.shape[3] - 1]
    pre_denses_down = pre_denses_with_borders[:, :, 0:pre_denses_with_borders.shape[2] - 2,
                      1:pre_denses_with_borders.shape[3] - 1]
    pre_mask = (pre_denses_center > pre_denses_left) & \
               (pre_denses_center > pre_denses_right) & \
               (pre_denses_center > pre_denses_up) & \
               (pre_denses_center > pre_denses_down)

    pre_mask = pre_mask[:, :, 1:pre_denses_center.shape[2] - 1, 1:pre_denses_center.shape[3] - 1]
    pre_cors = [list(zip(np.nonzero(pre_mask[i, 0])[1], np.nonzero(pre_mask[i, 0])[0])) for i in
                range(pre_mask.shape[0])]

    # 计算预测位置与真实位置之间的距离
    dists = list()
    for idx in range(len(gt_cors)):
        dists.append(get_dists(pre_cors[idx], gt_cors[idx]))
    # dists = np.stack(dists, axis=0)     # shape(N, H, W)
    mask = get_mask(pre_cors, gt_cors, dists, shape=[gt_mask.shape[2], gt_mask.shape[3]])

    mask = mask[:, np.newaxis, :, :]
    C_head = 1.0
    mask = C_head * mask
    mask = torch.tensor(mask, dtype=torch.float, device='cuda')
    error = (pre_denses - gt_denses) * (pre_denses - gt_denses)
    print('误差矩阵')
    print(error)
    loss1_rubust = torch.mean(torch.sum(mask * error, dim=(1, 2, 3)))  # batch上均方损失
    return loss1_rubust


def loss1_rub(pre_denses, gt_denses, gt_cors):
    '''
    峰值或者位置损失
    :param pre_denses:Tensor with shape[N, 1, H, W]
    :param gt_denses:Tensor with shape[N, 1, H, W]
    :param gt_cors:list with length n, each element is a list [list, list..., list] n
    :return:
    '''
    copy_pre = pre_denses.cpu().data.numpy()
    copy_gt = gt_denses.cpu().data.numpy()

    gt_mask = np.zeros_like(copy_gt, dtype=bool)
    for i in range(copy_gt.shape[0]):
        gt_cor = np.array(gt_cors[i])  # list to array
        gt_mask[i, 0, gt_cor[:, 1], gt_cor[:, 0]] = True

    pre_denses_with_borders = np.pad(copy_pre, [(0, 0), (0, 0), (2, 2), (2, 2)], mode='constant')
    pre_denses_center = pre_denses_with_borders[:, :, 1:pre_denses_with_borders.shape[2] - 1,
                        1:pre_denses_with_borders.shape[3] - 1]
    pre_denses_left = pre_denses_with_borders[:, :, 1:pre_denses_with_borders.shape[2] - 1,
                      2:pre_denses_with_borders.shape[3]]
    pre_denses_right = pre_denses_with_borders[:, :, 1:pre_denses_with_borders.shape[2] - 1,
                       0:pre_denses_with_borders.shape[3] - 2]
    pre_denses_up = pre_denses_with_borders[:, :, 2:pre_denses_with_borders.shape[2],
                    1:pre_denses_with_borders.shape[3] - 1]
    pre_denses_down = pre_denses_with_borders[:, :, 0:pre_denses_with_borders.shape[2] - 2,
                      1:pre_denses_with_borders.shape[3] - 1]
    pre_mask = (pre_denses_center > pre_denses_left) & \
               (pre_denses_center > pre_denses_right) & \
               (pre_denses_center > pre_denses_up) & \
               (pre_denses_center > pre_denses_down)

    pre_mask = pre_mask[:, :, 1:pre_denses_center.shape[2] - 1, 1:pre_denses_center.shape[3] - 1]
    pre_cors = [list(zip(np.nonzero(pre_mask[i, 0])[1], np.nonzero(pre_mask[i, 0])[0])) for i in
                range(pre_mask.shape[0])]

    # 计算预测位置与真实位置之间的距离
    dists = list()
    for idx in range(len(gt_cors)):
        dists.append(get_dists(pre_cors[idx], gt_cors[idx]))
    # dists = np.stack(dists, axis=0)     # shape(N, H, W)
    mask = get_mask(pre_cors, gt_cors, dists, shape=[gt_mask.shape[2], gt_mask.shape[3]])

    mask = mask[:, np.newaxis, :, :]
    C_head = 1.0
    mask = C_head * mask
    mask = torch.tensor(mask, dtype=torch.float, device='cuda')
    error = (pre_denses - gt_denses) * (pre_denses - gt_denses)
    print('误差矩阵')
    print(error)
    loss1_rub = torch.mean(mask * error)  # batch上均方损失
    return loss1_rub


def loss4(pre_denses, gt_denses):
    '''
    带权重的均方损失
    :param pre_denses:
    :param gt_denses:
    :return:
    '''
    kernel = torch.tensor([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], dtype=torch.float, device='cuda', requires_grad=False)
    kernel = torch.reshape(kernel, shape=(1, 1, 3, 3))

    pre_ker = F.conv2d(pre_denses, kernel, padding=1)
    gt_ker = F.conv2d(gt_denses, kernel, padding=1)
    loss = F.mse_loss(pre_ker, gt_ker, reduction='mean')
    return loss


def loss6(pre_denses, gt_denses):
    '''
    带权重的均方损失
    :param pre_denses:
    :param gt_denses:
    :return:
    '''
    kernel = [[[0, -1, 0], [0, 1, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
              [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 1, -1], [0, 0, 0]]]
    kernel = torch.tensor(kernel, dtype=torch.float, device='cuda', requires_grad=False)
    kernel = torch.reshape(kernel, shape=(4, 1, 3, 3))

    pre_ker = F.conv2d(pre_denses, kernel, padding=1)
    gt_ker = F.conv2d(gt_denses, kernel, padding=1)

    torch.max(0, gt_ker - pre_ker + 0.5)
    return loss


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
        pass

    def forward(self, pre_denses, gt_denses, gt_cors):
        return loss1(pre_denses, gt_denses, gt_cors) + loss2(pre_denses, gt_denses)


if __name__ == '__main__':
    gt_cors = np.array([[[4, 4], [5, 5], [6, 6]], [[4, 4]]])
    # print(gt_cors)
    gt_denses = np.zeros(shape=(2, 1, 10, 10))
    gt_denses[0, 0, 4, 4] = 2
    gt_denses[0, 0, 5, 5] = 2
    gt_denses[0, 0, 6, 6] = 2
    gt_denses[1, 0, 4, 4] = 2

    pre_denses = np.zeros(shape=(2, 1, 10, 10))
    pre_denses[0, 0, 4, 5] = 2
    pre_denses[0, 0, 3, 3] = 2
    pre_denses[1, 0, 6, 6] = 2
    pre_denses[1, 0, 4, 4] = 2

    pre_denses = torch.tensor(pre_denses, requires_grad=True, dtype=torch.float, device='cuda')
    gt_denses = torch.tensor(gt_denses, requires_grad=True, dtype=torch.float, device='cuda')

    # pre_denses = pre_denses.type(torch.FloatTensor).cuda()
    # gt_denses = gt_denses.type(torch.FloatTensor).cuda()
    # gt_cors = trans.tovariable(gt_cors, cuda=F)

    print('预测值')
    print(pre_denses, gt_denses)
    # loss = loss1_rubust(pre_denses, gt_denses, gt_cors)
    loss = loss1_rubust(pre_denses, gt_denses, gt_cors)
    print(loss)
    print('pre梯度矩阵')
    print(pre_denses.grad)
    loss.backward()

    print('post梯度矩阵')
    print(pre_denses.grad)
    # print(gt_denses.grad)




