# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss, DAMPTripletLoss, RRSITRRobustTripletLoss, AGWTripletLoss
from .center_loss import CenterLoss
from .mmd_loss import MMDLoss, ZJYLoss, FDALoss
from .adasp_standalone import AdaSPLoss
from .NCPloss import NCPLoss
# from ..... import xxxxloss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    # mmdloss = MMDLoss()#DiscLoss()
    center_criterion = FDALoss() #MMDLoss() #CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    ncp = None
    
    if 'damptriplet' == cfg.MODEL.METRIC_LOSS_TYPE:
        triplet = DAMPTripletLoss(APS=cfg.MODEL.APS,ANS=cfg.MODEL.ANS)
        print("using damp triplet loss for training")

    elif 'triplet' == cfg.MODEL.METRIC_LOSS_TYPE:
        print("using soft triplet loss for training")
        triplet = TripletLoss()
    elif 'rrsitr' ==  cfg.MODEL.METRIC_LOSS_TYPE:
        triplet = RRSITRRobustTripletLoss()
        print("using RRSITRRobustTripletLoss loss from cvpr2026")
    elif 'adasp' in cfg.MODEL.METRIC_LOSS_TYPE:
        adasp = AdaSPLoss(
                temp=getattr(cfg.SOLVER, 'ADASP_TEMP', 0.05),       # 需要加配置项或用默认值
                loss_type=getattr(cfg.SOLVER, 'ADASP_TYPE', 'adasp')  # 'adasp' / 'sp-h' / 'sp-lh'
            )
        print(f"using AdaSP loss (type={adasp.loss_type}, temp={adasp.temp})")
    elif 'agw' in cfg.MODEL.METRIC_LOSS_TYPE:
        agw = AGWTripletLoss()
        print(f"using AGWTriplet loss from tpami2021")
    elif 'ncp' in cfg.MODEL.METRIC_LOSS_TYPE:
        ncp = NCPLoss(nb_classes=num_classes, sz_embedding=768)
        center_criterion.ncp = ncp 
        print(f"using NCP loss from 26ESA-DNCPH")
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

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'damptriplet':
                TRI_LOSS, _, _, num_ap, num_an = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)


            elif cfg.MODEL.METRIC_LOSS_TYPE == 'rrsitr':
                TRI_LOSS, _, _, num_ap, num_an = triplet(feat, target)

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'adasp':
                # AdaSP 内部已做 L2 normalize，直接传 feat 和 target
                with torch.cuda.amp.autocast(enabled=False):          # 防 AMP 溢出
                    TRI_LOSS = adasp(feat.float(), target)          # 强制 float32
                # 不再需要 normalize_feature 参数和 [0] 下标
            
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'agw':
                TRI_LOSS = agw(feat, target, normalize_feature=True)

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'ncp':
                # NCPLoss 需要 one-hot 标签（不是类别索引）；proxies 是 fp32，需关 autocast + feat 转 fp32 防 dtype 不匹配
                with torch.cuda.amp.autocast(enabled=False):
                    if ncp.proxies.device != feat.device:
                        ncp.to(feat.device)  # 模型在 do_train 里才 .to(cuda)，ncp 单独创建，第一次前向时搬到 GPU（原地移动，优化器引用仍有效）
                    TRI_LOSS = ncp(feat.float(), F.one_hot(target, num_classes).float())

            else:
                print('error got {} loss type'.format(cfg.MODEL.METRIC_LOSS_TYPE))
                assert 2==0
            
            return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS , cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS , num_ap, num_an

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))




    return loss_func, center_criterion


