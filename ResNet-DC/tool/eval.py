import numpy as np


def eval_one(cors, my_cors, s):
    """
    :param t_pos:一副图像中真实位置array
    :param p_pos:一副图像中预测位置array
    :param s: 缩放比例
    :return:
    """
    t_count = len(cors)
    p_count = len(my_cors)
    a_count = 0
    if t_count == 0 or p_count == 0:
        return t_count, a_count, p_count
    for te in cors:
        for pe in my_cors:
            # 存在预测位置位于真实位置的s*s邻域
            if te[0] - s/2 <= pe[0] <= te[0] + s/2 and te[1] - s/2 <= pe[1] <= te[1] + s/2:
                a_count += 1
                break
    return t_count, a_count, p_count


def eval_acc(n_cors, my_n_cors, s=8):
    t_counts, a_counts, p_counts = 0, 0, 0
    # abs_error, square_error = .0, .0
    for cors, my_cors in zip(n_cors, my_n_cors):
        t_count, a_count, p_count = eval_one(cors, my_cors, s)
        # 真实人数
        t_counts += t_count
        # 预测正确的人数
        a_counts += a_count
        # 预测的总人数
        p_counts += p_count
    return t_counts, a_counts, p_counts
    # return abs_error, square_error, t_counts, a_counts, p_counts


if __name__ == '__main__':
    t_poss = np.array([[[1, 1], [10, 10]],
                       [[15, 15], [10, 10]]])

    p_poss = np.array([[[1, 1], [11, 11], [20, 20]],
                       [[18, 18], [6, 6], [5, 5]]])

    print(eval_acc(t_poss, p_poss, 8))
