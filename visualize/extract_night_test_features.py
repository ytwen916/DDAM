import argparse
import glob
import os
import re

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader

from config import cfg
from datasets import make_dataloader
from datasets.bases import ImageDataset
from model import make_model


NIGHTREID_PROTOCOLS = {
    '528_wd': ('query', 'bounding_box_test'),
    '528_wod': ('query', 'bounding_box_test-528-noDistractors'),
    '1000_wd': ('query-1000', 'bounding_box_test_withDistractors'),
    '1000_wod': ('query-1000', 'bounding_box_test-1000-noDistractors'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract Query/Gallery features for NightReID or Night600')
    parser.add_argument('--config_file', required=True,
                        help='training/evaluation YAML, e.g. configs/nightreid/DDAM.yml')
    parser.add_argument('--weight', required=True,
                        help='checkpoint used by this model')
    parser.add_argument('--save_path', required=True,
                        help='output directory; features.mat will be written here')
    parser.add_argument('--file_name', default='features.mat',
                        help='MAT filename inside --save_path')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--nightreid_protocol', default='default',
                        choices=['default'] + list(NIGHTREID_PROTOCOLS),
                        help='NightReID test split: 528/1000 IDs with distractors (wd) or '
                             'without distractors (wod). default is 528_wd. Ignored for Night600.')
    parser.add_argument('--opts', nargs='*', default=[],
                        help='optional config overrides, e.g. DATALOADER.NUM_WORKERS 4')
    return parser.parse_args()


def configure(args):
    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.WEIGHT = args.weight
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)


def select_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda was requested, but CUDA is unavailable.')
    if name == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_nightreid_dir(dir_path):
    """Parse a NightReID query/gallery directory exactly as datasets/nightreid.py does."""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(
            f'NightReID protocol directory does not exist: {dir_path}\n'
            'Check that all four official protocol folders are present under DATASETS.ROOT_DIR/NightReID.')
    paths = sorted(glob.glob(os.path.join(dir_path, '*.jpg')))
    if not paths:
        raise RuntimeError(f'No JPG images found in NightReID protocol directory: {dir_path}')

    pattern = re.compile(r'(\d{4})([LR][123])C')
    cam_map = {'L1': 0, 'L2': 1, 'L3': 2, 'R1': 3, 'R2': 4, 'R3': 5}
    samples = []
    for path in paths:
        match = pattern.search(os.path.basename(path))
        if match is None:
            raise RuntimeError(f'Unexpected NightReID filename: {path}')
        pid_text, camera_text = match.groups()
        samples.append((path, int(pid_text), cam_map[camera_text], 1))
    return samples


def make_nightreid_protocol_loader(default_loader, protocol):
    """Reuse the repository validation transform/collate function for one official protocol."""
    query_name, gallery_name = NIGHTREID_PROTOCOLS[protocol]
    dataset_root = os.path.join(str(cfg.DATASETS.ROOT_DIR), 'NightReID')
    query_dir = os.path.join(dataset_root, query_name)
    gallery_dir = os.path.join(dataset_root, gallery_name)
    query = process_nightreid_dir(query_dir)
    gallery = process_nightreid_dir(gallery_dir)

    # Reuse the transform and collate function created by make_dataloader so preprocessing
    # remains byte-for-byte identical to the repository's normal evaluation pipeline.
    val_set = ImageDataset(query + gallery, default_loader.dataset.transform)
    loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=default_loader.collate_fn,
        pin_memory=bool(getattr(default_loader, 'pin_memory', False)),
    )
    info = {
        'protocol': protocol,
        'query_dir': query_dir,
        'gallery_dir': gallery_dir,
        'num_query': len(query),
        'num_gallery': len(gallery),
    }
    return loader, info


def load_weights_checked(model, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        obj = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location='cpu')

    if isinstance(obj, dict) and isinstance(obj.get('state_dict'), dict):
        obj = obj['state_dict']
    elif isinstance(obj, dict) and isinstance(obj.get('net'), dict):
        obj = obj['net']
    if not isinstance(obj, dict):
        raise TypeError(f'Checkpoint must contain a state_dict, got {type(obj)}')

    state = {str(k).replace('module.', ''): v for k, v in obj.items()
             if torch.is_tensor(v)}
    incompatible = model.load_state_dict(state, strict=False)
    missing = [k for k in incompatible.missing_keys
               if not k.endswith('num_batches_tracked')]
    if missing:
        raise RuntimeError(
            f'{len(missing)} model tensors are missing from the checkpoint, e.g. {missing[:5]}. '
            'Use the same YAML that trained this weight.')
    if incompatible.unexpected_keys:
        print(f'[Warning] ignored {len(incompatible.unexpected_keys)} unexpected checkpoint keys, '
              f'e.g. {incompatible.unexpected_keys[:5]}')
    print(f'==> Loaded checkpoint: {path}')


@torch.no_grad()
def extract(model, loader, device):
    """Extract exactly the eval embedding returned by model.forward()."""
    model.eval()
    features, pids, camids, paths = [], [], [], []
    done = 0
    for batch in loader:
        images, pid, camid, cam_label, view_label = batch[:5]
        images = images.to(device, non_blocking=True)
        cam_label = cam_label.to(device, non_blocking=True)
        view_label = view_label.to(device, non_blocking=True)
        feat = model(images, cam_label=cam_label, view_label=view_label)
        if isinstance(feat, (tuple, list)):
            raise RuntimeError('The model returned training outputs in eval mode; call model.eval() '
                               'or update extraction for this model definition.')
        features.append(feat.float().cpu())
        pids.extend(np.asarray(pid, dtype=np.int64).tolist())
        camids.extend(np.asarray(camid, dtype=np.int64).tolist())
        if len(batch) > 5:
            paths.extend(str(p) for p in batch[5])
        else:
            paths.extend('' for _ in range(len(pid)))
        done += int(images.shape[0])
        if done % 2048 < int(images.shape[0]):
            print(f'==> Extracted {done:,} images')

    return (torch.cat(features, dim=0).numpy().astype(np.float32, copy=False),
            np.asarray(pids, dtype=np.int64),
            np.asarray(camids, dtype=np.int64),
            np.asarray(paths, dtype=object))


def main():
    args = parse_args()
    print(f'==> Extract version: {EXTRACT_VERSION}')
    configure(args)
    device = select_device(args.device)

    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    configured_names = cfg.DATASETS.NAMES
    dataset_name = (str(configured_names[0]) if isinstance(configured_names, (list, tuple))
                    else str(configured_names)).lower()
    protocol_info = {
        'protocol': 'dataset_default',
        'query_dir': '',
        'gallery_dir': '',
        'num_query': int(num_query),
    }
    if dataset_name == 'nightreid':
        protocol = '528_wd' if args.nightreid_protocol == 'default' else args.nightreid_protocol
        val_loader, protocol_info = make_nightreid_protocol_loader(val_loader, protocol)
        num_query = protocol_info['num_query']
    elif args.nightreid_protocol != 'default':
        raise ValueError('--nightreid_protocol can only be used when DATASETS.NAMES is nightreid')

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    load_weights_checked(model, args.weight)
    model.to(device)

    print(f'==> Dataset: {cfg.DATASETS.NAMES} | protocol: {protocol_info["protocol"]} | device: {device}')
    all_features, all_pids, all_camids, all_paths = extract(model, val_loader, device)
    num_query = int(num_query)
    num_gallery = int(all_features.shape[0] - num_query)
    if num_query <= 0 or num_gallery <= 0:
        raise RuntimeError(f'Invalid split: query={num_query}, gallery={num_gallery}')

    query_fc = all_features[:num_query]
    gallery_fc = all_features[num_query:]
    query_label = all_pids[:num_query]
    gallery_label = all_pids[num_query:]
    query_cam = all_camids[:num_query]
    gallery_cam = all_camids[num_query:]

    os.makedirs(args.save_path, exist_ok=True)
    output = os.path.join(args.save_path, args.file_name)
    scipy.io.savemat(output, {
        'query_fc': query_fc,
        'gallery_fc': gallery_fc,
        'query_label': query_label,
        'gallery_label': gallery_label,
        'query_cam': query_cam,
        'gallery_cam': gallery_cam,
        'query_path': all_paths[:num_query],
        'gallery_path': all_paths[num_query:],
        'dataset_name': str(cfg.DATASETS.NAMES),
        'nightreid_protocol': protocol_info['protocol'],
        'query_dir': protocol_info['query_dir'],
        'gallery_dir': protocol_info['gallery_dir'],
        'weight_path': os.path.abspath(args.weight),
        'feature_definition': 'model.eval() output used by test.py',
    }, do_compression=True)

    print('\n' + '-' * 78)
    print(f'Feature file : {output}')
    print(f'Protocol     : {protocol_info["protocol"]}')
    print(f'Query images : {num_query:,} | IDs: {np.unique(query_label).size:,}')
    print(f'Gallery images: {num_gallery:,} | IDs: {np.unique(gallery_label).size:,}')
    print(f'Feature dim  : {query_fc.shape[1]:,}')
    print(f'QxG pairs    : {num_query:,} x {num_gallery:,} = {num_query * num_gallery:,}')
    print('-' * 78)


if __name__ == '__main__':
    main()
