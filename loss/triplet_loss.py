import torch
from torch import nn
import  torch.nn.functional as F
import numpy as np
# from visualize.tsne import thermal_pos


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist

def select_prefix_an(
    x: torch.Tensor,
    threshold: torch.Tensor,
    initial_s: int,
    max_s: int,
):
    """
    x:         [n, d]，每行从小到大排列
    threshold: [n, 1]
    """

    threshold = threshold.unsqueeze(1)


    # 前 max_s 个元素的前缀均值
    prefix_sum = x[:, :max_s].cumsum(dim=1)
    prefix_mean = prefix_sum / torch.arange(
        1, max_s + 1, device=x.device, dtype=x.dtype
    )

    # 从 initial_s 开始判断是否满足阈值
    meets = prefix_mean[:, initial_s - 1:] > threshold

    # 找到每行第一个满足条件的位置
    has_match = meets.any(dim=1)
    first_match = meets.to(torch.int64).argmax(dim=1) + initial_s

    # 没有满足条件的行直接取 max_s
    selected_s = torch.where(
        has_match,
        first_match,
        torch.full_like(first_match, max_s),
    )

    # 构造前 S 项的选择掩码
    # positions = torch.arange(max_s, device=x.device).unsqueeze(0)
    # selected_mask = positions < selected_s.unsqueeze(1)

    # 未选中的位置填 0
    # selected_values = x[:, :max_s] * selected_mask

    # 对应前 S 项的均值
    selected_mean = prefix_mean.gather(
        1, (selected_s - 1).unsqueeze(1)
    ).squeeze(1)


    # KK = torch.cat((selected_mean.view(1,-1),threshold.view(1,-1)),dim=0)
    # print(KK[:,0:9])

    return selected_s.float(), selected_mean
    # return selected_s, selected_mean, has_match, selected_mask, selected_values




def hard_example_mining(dist_mat, labels):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)


    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_pos.fill_diagonal_(False)
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())


    all_ap = dist_mat[is_pos].contiguous().view(N, -1)
    all_ap_sort, _ = torch.sort(all_ap, dim=-1, descending=True)
    dist_ap = all_ap_sort[:,0]
    num_ap = torch.ones_like(dist_ap).mean()




    all_an = dist_mat[is_neg].contiguous().view(N, -1)
    all_an_sort, _ = torch.sort(all_an,dim=-1,descending=False)
    dist_an = all_an_sort[:,0]
    num_an = torch.ones_like(dist_ap).mean()





    return dist_ap, dist_an, num_an, num_ap



def ddam_hard_example_mining(dist_mat, labels, APS, ANS):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)


    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_pos.fill_diagonal_(False)
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())


    all_ap = dist_mat[is_pos].contiguous().view(N, -1)
    all_ap_sort, _ = torch.sort(all_ap, dim=-1, descending=True)
    dist_ap = all_ap_sort[:,0:APS].mean(dim=1)
    num_ap = (APS * torch.ones_like(dist_ap)).mean()



    all_an = dist_mat[is_neg].contiguous().view(N, -1)
    all_an_sort, _ = torch.sort(all_an,dim=-1,descending=False)
    num_an, dist_an = select_prefix_an(all_an_sort,threshold=dist_ap,initial_s=1,max_s=ANS)
    num_an = num_an.mean()




    return dist_ap, dist_an, num_ap, num_an



class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        print("using SoftMarginLoss")
        self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an, num_ap, num_an = hard_example_mining(dist_mat, labels)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        score = dist_an - dist_ap

        loss = self.ranking_loss(score, y)

        return loss, dist_ap, dist_an, num_ap, num_an

class DDAMTripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0,APS=3, ANS=8):
        self.margin = margin
        self.hard_factor = hard_factor
        assert APS == 3
        self.ranking_loss = nn.SoftMarginLoss()
        self.APS = APS
        self.ANS = ANS
        print('dymaic APS:', self.APS, ', ANS:', self.ANS)

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an, num_ap, num_an = ddam_hard_example_mining(dist_mat, labels, APS=self.APS, ANS = self.ANS)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        score = dist_an - dist_ap
        loss = self.ranking_loss(score, y)

        return loss, dist_ap, dist_an, num_ap, num_an
