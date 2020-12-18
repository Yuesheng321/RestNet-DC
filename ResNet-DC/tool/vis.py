import numpy as np
import matplotlib.pyplot as plt
from colour import Color


def vis_heat(img, path=None, save=True):
    # 改自http://www.cnblogs.com/arkenstone/p/6932632.html
    linspace = 256
    img_width = img.shape[1]
    img_height = img.shape[0]
    color_map = np.empty([img_height, img_width, 3], dtype=int)
    # 使用Color来生成颜色梯度
    hex_colors = list(Color("blue").range_to(Color("red"), linspace))
    rgb_colors = [[rgb * 255 for rgb in color.rgb] for color in hex_colors]  # (256, 3)
    # 缩放
    max = np.max(img) + np.finfo(np.float32).eps
    img_scale = np.array(255*img/max, dtype=int)
    # print(img_scale.max(), img_scale.min())
    for row in range(img_height):
        for col in range(img_width):
            # print(img[row, col])
            try:
                color_map[row, col, :] = rgb_colors[img_scale[row, col]]
            except TypeError as e:
                print(e, max)
                return
            except IndexError as e:
                print(e, max)
                return
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(color_map)
    if save:
        if path is not None:
            plt.savefig(path)
        else:
            return fig
    else:
        plt.show()
    plt.close(fig)


def vis_img(img, path=None, save=True):
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(img)
    if save:
        if path is not None:
            plt.savefig(path)
        else:
            return fig
    else:
        plt.show()
    plt.close(fig)


def vis_mask(img, path=None, save=True):
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    if save:
        if path is not None:
            plt.savefig(path, cmap='gray')
        else:
            return fig
    else:
        plt.show()
    plt.close(fig)


def vis_imgCor(img, p_cor, g_cor, path=None, save=True):
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(img)
    # g_cor = np.array(g_cor)
    # p_cor = np.array(p_cor)
    # print(g_cor)
    if len(p_cor) != 0:
            plt.plot(p_cor[:, 0], p_cor[:, 1], 'b*')
    if len(g_cor) != 0:
        plt.plot(g_cor[:, 0], g_cor[:, 1], 'rx')
    plt.title('truth' + str(len(g_cor)) + 'pre' + str(len(p_cor)))
    if save:
        if path is not None:
            plt.savefig(path)
        else:
            return fig
    else:
        plt.show()
    plt.close(fig)


def vis_loss(loss_his, path=None, save=True):
    fig = plt.figure()
    plt.plot(loss_his)
    if save:
        if path is not None:
            plt.savefig(path)
        else:
            return fig
    else:
        plt.show()
    plt.close(fig)


def vis(writer, pre, gt, epoch, model='train', mode='den'):
    name = str(epoch) + '_' + model + '_' + mode + '_'
    if mode == 'den':
        gt_fig = vis_heat(gt)
        pre_fig = vis_heat(pre)
    else:
        gt_fig = vis_mask(gt)
        pre_fig = vis_mask(pre)
    writer.add_figure(name + 'org', gt_fig)
    writer.add_figure(name + 'pre', pre_fig)


if __name__ == '__main__':
    vis_loss([1, 3, 4, 10.0], save=False)
