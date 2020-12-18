from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
import numpy as np
from tool.config import opt
from operator import itemgetter
import copy
import cv2


def find_peaks(dense, threl=opt.threl):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """
    # 四领域最大值，且原始像素值大于阀值
    peaks_binary = (maximum_filter(dense, footprint=generate_binary_structure(2, 1)) == dense) * (dense > threl)
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y x]...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T


def nms(denses):
    """
    :param denses: 峰值密度图的list
    :return: 每个峰值密度图对应的人数list，和位置list
    """
    counts = list()
    cors = list()
    denses = np.array(denses)
    if denses.ndim == 2:
        cor = find_peaks(denses)
        cors.append(cor)
        counts.append(cor.shape[0])
    elif denses.ndim == 3:
        for dense in denses:
            cor = find_peaks(dense)
            cors.append(cor)
            counts.append(cor.shape[0])
    return np.array(counts), cors


def extract_cors(heatmap):
    heatmap[heatmap < 0.1] = 0
    # heatmap[:] = gaussian_filter(heatmap[:], sigma=1.0)  # 过滤掉一些错误的峰值
    #heatmap[:] = cv2.GaussianBlur(heatmap[:], ksize=(5, 5), sigmaX=0)
    confidmap = copy.deepcopy(heatmap)
    confidmap[confidmap < opt.threl] = 0        # 设置阈值
    heatmap_with_borders = np.pad(confidmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)

    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
    # print(heatmap_peaks)

    cors = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
    cors = sorted(cors, key=itemgetter(0))
    cors_with_score = []
    for i in range(len(cors)):
        cor_with_score = (cors[i][0], cors[i][1], heatmap[cors[i][1], cors[i][0]])
        cors_with_score.append(cor_with_score)
    cors_with_score = np.array(cors_with_score)

    # elif cors_with_score.ndim == 1:
    #     cors_with_score = cors_with_score[np.newaxis, :]
    return cors_with_score


def localmax(heatmap):
    n_cors = list()
    if heatmap.ndim == 2:
        n_cors.append(extract_cors(heatmap))
    elif heatmap.ndim == 3:
        n_cors.append(extract_cors(heatmap[0, :, :]))
    elif heatmap.ndim == 4:
        N, _, H, W = heatmap.shape
        if N == 1:
            n_cors.append(extract_cors(heatmap[0, 0, :, :]))
        elif N > 1:
            for idx in range(N):
                n_cors.append(extract_cors(heatmap[idx, 0, :, :]))
    return n_cors


if __name__ == '__main__':
    a = np.array([[0, 0, 0, 0],
                  [0, 0.6, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    print(extract_cors(a))
