class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class Meter(object):
    def __init__(self):
        self.a_count = 0
        self.p_count = 1e-10
        self.g_count = 0
        self.pca = 0
        self.gca = 0

    def update(self, a_count, p_count, g_count):
        self.a_count += a_count
        self.p_count += p_count
        self.g_count += g_count

        self.pca = self.a_count/self.p_count
        self.gca = self.a_count/self.g_count

    def get_pca(self):
        return self.pca

    def get_gca(self):
        return self.gca
