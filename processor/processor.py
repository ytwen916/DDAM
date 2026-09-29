import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from loss.mmd_loss import MMDLoss

def do_train(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             scheduler_center,
             loss_fn,
             num_query, local_rank, output_dir):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_id_meter = AverageMeter()
    loss_tri_meter = AverageMeter()
    loss_disc_meter = AverageMeter()
    acc_meter = AverageMeter()
    loss_e_meter = AverageMeter()

    num_ap_meter = AverageMeter()
    num_an_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    best_mAP = 0
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_id_meter.reset()
        loss_tri_meter.reset()
        loss_disc_meter.reset()
        loss_e_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        num_ap_meter.reset()
        num_an_meter.reset()
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score_i, feat_i = model(img, target, cam_label=target_cam, view_label=target_view) #i-not ehance, # d do enhance

                idloss, triloss, num_ap, num_an = loss_fn(score_i, feat_i, target, target_cam)

                loss = idloss + triloss

                discloss = triloss



            # scaler.scale(loss_e).backward(retain_graph=True)
            scaler.scale(loss).backward()


            scaler.step(optimizer)
            # scaler.step(optimizer_center)
            scaler.update()


            acc = (score_i.max(1)[1] == target).float().mean()

            loss_id_meter.update(idloss.item(), img.shape[0])
            loss_tri_meter.update(triloss.item(), img.shape[0])
            loss_disc_meter.update(discloss.item(), img.shape[0])
            num_ap_meter.update(num_ap.item(),1)
            num_an_meter.update(num_an.item(), 1)
            # loss_e_meter.update(loss_e.item(),img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            # if cfg.MODEL.DIST_TRAIN:
            #     if dist.get_rank() == 0:
            #         if (n_iter + 1) % log_period == 0:
            #             base_lr = optimizer.param_groups[0]['lr'] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else optimizer.param_groups[0]['lr']
            #             e_lr = optimizer_center.param_groups[0]['lr'] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else optimizer_center.param_groups[0]['lr']
            #             logger.info("Epoch[{}] Iter[{}/{}] IDLoss: {:.3f}, TRILoss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, E Lr: {:.2e}"
            #                         .format(epoch, (n_iter + 1), len(train_loader), loss_id_meter.avg, loss_tri_meter.avg, acc_meter.avg, base_lr, e_lr))
            # else:
            if (n_iter + 1) % log_period == 0:
                base_lr = optimizer.param_groups[0]['lr'] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else \
                optimizer.param_groups[0]['lr']
                e_lr = optimizer_center.param_groups[0]['lr'] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else \
                optimizer_center.param_groups[0]['lr']
                logger.info(
                    "Epoch[{}] Iter[{}/{}] IDLoss: {:.3f}, TRILoss: {:.3f}, DISCLoss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, E Lr: {:.2e}, NumAP: {:.4f}, NumAN: {:.4f}"
                    .format(epoch, (n_iter + 1), len(train_loader), loss_id_meter.avg, loss_tri_meter.avg,
                            loss_disc_meter.avg, acc_meter.avg, base_lr, e_lr, num_ap_meter.avg, num_an_meter.avg))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
            # scheduler_center.step(epoch)
        else:
            scheduler.step()
            # scheduler_center.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0: #
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period ==0: #  == 0: #
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()

            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        imgg = torch.mean(img, dim=1, keepdim=True).expand_as(img).to(device)
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat_i= model(img, imgg, cam_label=camids, view_label=target_view) # d for do enhance
                        evaluator.update((feat_i, vid, camid))
                cmc_ls, mAP_ls, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                for i in range(len(cmc_ls)):
                    logger.info("mode: {}".format(i))
                    logger.info("mAP: {:.1%}".format(mAP_ls[i]))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_ls[i][r - 1]))
                if mAP_ls[-1]>best_mAP:
                    best_mAP = mAP_ls[-1]
                    torch.save(model.state_dict(), os.path.join(output_dir, cfg.MODEL.NAME + '_best_mAP.pth'))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            # feat_i, feat_d, feat_c = model(img, cam_label=camids, view_label=target_view)
            # evaluator.update((feat_i, feat_d, feat_c, pid, camid))
            feat_i = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat_i, pid, camid))
            img_path_list.extend(imgpath)

    cmc_ls, mAP_ls, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    for i in range(len(cmc_ls)):
        logger.info("mode: {}".format(i))
        logger.info("mAP: {:.1%}".format(mAP_ls[i]))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_ls[i][r - 1]))
    return cmc_ls[-1][0], cmc_ls[-1][4]