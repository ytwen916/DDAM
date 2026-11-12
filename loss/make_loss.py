# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss, DDAMTripletLoss
from .center_loss import CenterLoss
from .mmd_loss import MMDLoss, ZJYLoss, FDALoss
# from ..... import xxxxloss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    # mmdloss = MMDLoss()#DiscLoss()
    center_criterion = FDALoss() #MMDLoss() #CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    ncp = None
    
    if 'ddamtriplet' == cfg.MODEL.METRIC_LOSS_TYPE:
        triplet = DDAMTripletLoss(APS=cfg.MODEL.APS,ANS=cfg.MODEL.ANS)
        print("using ddam triplet loss for training")

    elif 'triplet' == cfg.MODEL.METRIC_LOSS_TYPE:
        print("using soft triplet loss for training")
        triplet = TripletLoss()
    else:
        print('eror got {} loss type'.format(cfg.MODEL.METRIC_LOSS_TYPE))
        assert 1==0




    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target,target_cam):
            return F.cross_entropy(score, target)


    elif 'triplet' in sampler:
        def loss_func(score, feat, target, target_cam): #score/feat not use enhance, the rest do enhance
            num_ap = torch.zeros((), device=target.device, dtype=torch.float32)
            num_an = torch.zeros((), device=target.device, dtype=torch.float32)
            ID_LOSS = F.cross_entropy(score, target)
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                TRI_LOSS, _, _, num_ap, num_an = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'ddamtriplet':
                TRI_LOSS, _, _, num_ap, num_an = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)

            else:
                print('error got {} loss type'.format(cfg.MODEL.METRIC_LOSS_TYPE))
                assert 2==0
            
            return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS , cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS , num_ap, num_an

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))




    return loss_func, center_criterion


