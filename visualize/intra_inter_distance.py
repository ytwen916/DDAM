import argparse
import csv
import math
import os

import matplotlib
matplotlib.use('Agg')  # headless servers
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
import torch


PLOT_VERSION = 'v15-auto-annotation-layout'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Two-panel Triplet-vs-DDAM Query-Gallery intra/inter distance plot')
    parser.add_argument('--mat_files', nargs=2, required=True, metavar=('TRIPLET_MAT', 'DDAM_MAT'),
                        help='exactly two feature files, ordered as Triplet then DDAM')
    parser.add_argument('--titles', nargs=2, default=['Triplet', 'DDAM'],
                        metavar=('TRIPLET_TITLE', 'DDAM_TITLE'))
    parser.add_argument('--metric', default='euclidean', choices=['euclidean', 'cosine'])
    parser.add_argument('--num_bins', default=None, type=int,
                        help='default: 400 for --pair_mode all, 100 for --pair_mode hard')
    # ---------------- new options ----------------
    parser.add_argument('--pair_mode', default='all', choices=['all', 'hard'],
                        help='all : every Query-Gallery pair (original figure). '
                             'hard: one value per query -> its positive distance (see --hard_pos) vs. '
                             'its NEAREST negative distance. This is what retrieval ranking depends on.')
    parser.add_argument('--hard_pos', default='nearest', choices=['nearest', 'farthest', 'mean'],
                        help='hard mode: which positive distance to keep per query. nearest -> share of '
                             'queries with margin>0 equals Rank-1; farthest -> all positives before any negative')
    parser.add_argument('--positive_policy', default='all_same_id',
                        choices=['all_same_id', 'cross_camera'],
                        help='all_same_id: every same-PID Query-Gallery pair is intra-class '
                             '(the supplied complete-matrix sketch). cross_camera: additionally remove '
                             'same-PID same-camera pairs, following the standard ReID evaluation protocol')
    parser.add_argument('--query_domain', default='Query',
                        help='name printed for the MAT query split')
    parser.add_argument('--gallery_domain', default='Gallery',
                        help='name printed for the MAT gallery split')
    parser.add_argument('--sharey', action='store_true', help='same y-axis range for all panels')
    parser.add_argument('--legend_loc', default='upper left',
                        help='matplotlib legend location, e.g. "upper left", "center left"')
    parser.add_argument('--chunk', default=256, type=int,
                        help='number of Query rows per distance block')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--smooth_sigma', default=2.0, type=float,
                        help='Gaussian smoothing in histogram-bin units; 0 disables it')
    parser.add_argument('--range_pct', nargs=2, default=(5.0, 95.0), type=float,
                        metavar=('LOW', 'HIGH'), help='percentiles used for the intra-class range mark')
    parser.add_argument('--xlim', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'))
    parser.add_argument('--save_path', required=True, help='output path without extension')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'],
                        choices=['png', 'pdf', 'svg', 'eps'])
    parser.add_argument('--dpi', default=600, type=int)
    parser.add_argument('--font_path', default='', help='optional Times New Roman TTF path')
    parser.add_argument('--xlabel', default=None,
                        help='x-axis label; default "Feature Distance" / "Cosine Distance"')
    parser.add_argument('--ylabel', default='Density',
                        help='y-axis label; the plotted values are probability densities')
    parser.add_argument('--decimals', default=2, type=int,
                        help='decimals shown in the intra/inter means and their mean-distance gap')
    parser.add_argument('--fill_alpha', default=0.6, type=float,
                        help='opacity of the filled histograms')
    parser.add_argument('--show_curve', action='store_true',
                        help='also draw the Gaussian-smoothed curves on top of the histograms')
    parser.add_argument('--panel_size', nargs=2, default=(5.2, 3.8), type=float,
                        metavar=('W', 'H'), help='size of one subplot in inches')
    return parser.parse_args()


def choose_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda was requested, but CUDA is unavailable.')
    if name == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def first_key(data, names):
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f'None of these MAT keys exists: {names}')


def feature_rows(array, expected_rows, name):
    x = np.asarray(array, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f'{name} must be 2-D, got shape {x.shape}')
    if x.shape[0] == expected_rows:
        return x
    if x.shape[1] == expected_rows:
        print(f'[Info] transposing {name} from {x.shape} to {(x.shape[1], x.shape[0])}')
        return x.T.copy()
    raise ValueError(f'{name} shape {x.shape} does not match {expected_rows} labels')


def load_feature_mat(path, need_cam):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = scipy.io.loadmat(path)
    q_pid = np.asarray(first_key(data, ['query_label', 'query_pid'])).reshape(-1).astype(np.int64)
    g_pid = np.asarray(first_key(data, ['gallery_label', 'gallery_pid'])).reshape(-1).astype(np.int64)
    q_feat = feature_rows(first_key(data, ['query_fc', 'query_f', 'query_feature']),
                          q_pid.size, 'query feature')
    g_feat = feature_rows(first_key(data, ['gallery_fc', 'gallery_f', 'gallery_feature']),
                          g_pid.size, 'gallery feature')
    if q_feat.shape[1] != g_feat.shape[1]:
        raise ValueError(f'Feature dimensions differ: Query {q_feat.shape}, Gallery {g_feat.shape}')
    if need_cam:
        q_cam = np.asarray(first_key(data, ['query_cam', 'query_camid'])).reshape(-1).astype(np.int64)
        g_cam = np.asarray(first_key(data, ['gallery_cam', 'gallery_camid'])).reshape(-1).astype(np.int64)
        if q_cam.size != q_pid.size or g_cam.size != g_pid.size:
            raise ValueError('Camera labels do not match the number of images')
    else:
        q_cam, g_cam = np.zeros_like(q_pid), np.ones_like(g_pid)  # never "same camera"
    return q_feat, g_feat, q_pid, g_pid, q_cam, g_cam


def queries_with_positive(q_pid, g_pid, q_cam, g_cam, exclude_same_cam):
    """True for queries that keep >=1 intra pair; eval_func skips all other queries."""
    g_count, g_cam_count = {}, {}
    for pid, cam in zip(g_pid.tolist(), g_cam.tolist()):
        g_count[pid] = g_count.get(pid, 0) + 1
        g_cam_count[(pid, cam)] = g_cam_count.get((pid, cam), 0) + 1
    keep = []
    for pid, cam in zip(q_pid.tolist(), q_cam.tolist()):
        valid = g_count.get(pid, 0) - (g_cam_count.get((pid, cam), 0) if exclude_same_cam else 0)
        keep.append(valid > 0)
    return np.asarray(keep, dtype=bool)


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norm, 1e-12, None)).astype(np.float32, copy=False)


def pair_audit(q_pid, g_pid, q_cam, g_cam, exclude_same_cam):
    """Count pairs analytically. Positive = same PID (minus same-camera pairs when excluded),
    negative = different PID (all cameras, as in the standard evaluation)."""
    nq, ng = int(q_pid.size), int(g_pid.size)
    if nq == 0 or ng == 0:
        raise RuntimeError(f'Empty split: query={nq}, gallery={ng}')
    g_ids, g_counts = np.unique(g_pid, return_counts=True)
    g_map = dict(zip(g_ids.tolist(), g_counts.astype(np.int64).tolist()))
    g_cam_map = {}
    for pid, cam in zip(g_pid.tolist(), g_cam.tolist()):
        g_cam_map[(pid, cam)] = g_cam_map.get((pid, cam), 0) + 1
    q_map = {}
    for pid in q_pid.tolist():
        q_map[pid] = q_map.get(pid, 0) + 1

    same_pid = same_cam = 0
    positive_q = 0
    for pid, cam in zip(q_pid.tolist(), q_cam.tolist()):
        n_same = g_map.get(pid, 0)
        n_cam = g_cam_map.get((pid, cam), 0)
        same_pid += n_same
        same_cam += n_cam
        valid = n_same - n_cam if exclude_same_cam else n_same
        positive_q += int(valid > 0)
    total = nq * ng
    excluded = same_cam if exclude_same_cam else 0
    positive = same_pid - excluded
    negative = total - same_pid
    return {
        'query_images': nq, 'gallery_images': ng,
        'query_ids': len(q_map), 'gallery_ids': int(g_ids.size),
        'total_pairs': total, 'same_pid_pairs': same_pid, 'same_cam_pairs': same_cam,
        'excluded_pairs': excluded, 'positive_pairs': positive, 'negative_pairs': negative,
        'positive_query_images': positive_q,
        'exclude_same_cam': bool(exclude_same_cam),
    }


def print_audit(path, audit, query_domain, gallery_domain, positive_policy):
    line = '-' * 88
    print(f'\n{line}\nFEATURE FILE: {path}')
    print(f'  {query_domain:<27}: {audit["query_images"]:,} images / {audit["query_ids"]:,} IDs')
    print(f'  {gallery_domain:<27}: {audit["gallery_images"]:,} images / {audit["gallery_ids"]:,} IDs')
    print(f'  Complete QxG pairs         : {audit["query_images"]:,} x '
          f'{audit["gallery_images"]:,} = {audit["total_pairs"]:,}')
    print(f'  Distance matrix in code    : Q x G = {audit["query_images"]:,} x '
          f'{audit["gallery_images"]:,}')
    print(f'  Matrix drawn in the sketch : G x Q = {audit["gallery_images"]:,} x '
          f'{audit["query_images"]:,} (transpose; identical pair set)')
    if positive_policy == 'cross_camera':
        share = 100.0 * audit['same_cam_pairs'] / max(audit['same_pid_pairs'], 1)
        print(f'  Same-PID pairs             : {audit["same_pid_pairs"]:,} '
              f'(same camera: {audit["same_cam_pairs"]:,}, {share:.2f}%)')
        print('  Same-PID same-camera pairs : EXCLUDED (standard ReID protocol)')
    else:
        print(f'  Same-PID pairs             : {audit["same_pid_pairs"]:,}')
        print('  Camera labels              : ignored; every same-PID pair is intra-class')
    print(f'  Positive-pair policy       : {positive_policy}')
    print(f'  Positive / intra pairs     : {audit["positive_pairs"]:,}')
    print(f'  Negative / inter pairs     : {audit["negative_pairs"]:,} (different PID, all cameras)')
    print(f'  Pair partition check       : {audit["positive_pairs"]:,} + {audit["negative_pairs"]:,} + '
          f'{audit["excluded_pairs"]:,} excluded = {audit["total_pairs"]:,}')
    print(f'  Queries with >=1 intra pair: {audit["positive_query_images"]:,} / '
          f'{audit["query_images"]:,}\n{line}')


def histogram_pairs(q_feat, g_feat, q_pid, g_pid, q_cam, g_cam, exclude_same_cam,
                    metric, bins, chunk, device):
    q_feat = l2_normalize(q_feat)
    g_feat = l2_normalize(g_feat)
    g_tensor = torch.from_numpy(g_feat).to(device)
    hist_pos = np.zeros(bins.size - 1, dtype=np.int64)
    hist_neg = np.zeros(bins.size - 1, dtype=np.int64)
    pos_sum = neg_sum = pos_sumsq = neg_sumsq = 0.0
    # Per-query statistics (one value per query) for --pair_mode hard and the Rank-1 check.
    per_query = {'nearest': [], 'farthest': [], 'mean': [], 'neg_nearest': [],
                 'rank1_hit': []}

    nq = q_feat.shape[0]
    for start in range(0, nq, max(1, int(chunk))):
        stop = min(start + max(1, int(chunk)), nq)
        q_tensor = torch.from_numpy(q_feat[start:stop]).to(device)
        similarity = q_tensor @ g_tensor.T
        if metric == 'euclidean':
            distance = torch.sqrt(torch.clamp(2.0 - 2.0 * similarity, min=0.0, max=4.0))
        else:
            distance = torch.clamp(1.0 - similarity, min=0.0, max=2.0)
        distance = distance.cpu().numpy()
        same = q_pid[start:stop, None] == g_pid[None, :]
        valid_pos = same
        if exclude_same_cam:
            valid_pos = same & (q_cam[start:stop, None] != g_cam[None, :])
        pos = distance[valid_pos]
        neg = distance[~same]
        hist_pos += np.histogram(pos, bins=bins)[0]
        hist_neg += np.histogram(neg, bins=bins)[0]
        pos_sum += float(pos.sum(dtype=np.float64))
        neg_sum += float(neg.sum(dtype=np.float64))
        pos_sumsq += float(np.square(pos, dtype=np.float64).sum(dtype=np.float64))
        neg_sumsq += float(np.square(neg, dtype=np.float64).sum(dtype=np.float64))
        # Every kept query has >=1 valid positive (see queries_with_positive).
        per_query['nearest'].append(np.where(valid_pos, distance, np.inf).min(axis=1))
        per_query['farthest'].append(np.where(valid_pos, distance, -np.inf).max(axis=1))
        per_query['mean'].append((distance * valid_pos).sum(axis=1) / valid_pos.sum(axis=1))
        per_query['neg_nearest'].append(np.where(same, np.inf, distance).min(axis=1))
        # Rank-1 is checked from the actual sorted gallery list.  This handles the
        # filtering protocol directly and avoids using a strict min-positive <
        # min-negative shortcut at the reporting stage.
        for row in range(distance.shape[0]):
            query_index = start + row
            keep_gallery = np.ones(g_pid.size, dtype=bool)
            if exclude_same_cam:
                keep_gallery &= ~((g_pid == q_pid[query_index]) &
                                  (g_cam == q_cam[query_index]))
            kept_indices = np.flatnonzero(keep_gallery)
            if kept_indices.size == 0:
                raise RuntimeError(f'Query row {query_index} has no gallery candidate after filtering.')
            # Match the evaluator's operation order: sort the whole gallery, then
            # remove protocol-invalid samples.  This also defines how exact ties
            # are handled, instead of inferring Rank-1 from two scalar minima.
            order = np.argsort(distance[row])
            first = order[keep_gallery[order]][0]
            per_query['rank1_hit'].append(g_pid[first] == q_pid[query_index])
        print(f'==> distance rows: {stop:,}/{nq:,}')

    del g_tensor
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    per_query = {
        k: (np.asarray(v, dtype=np.float64) if k == 'rank1_hit'
            else np.concatenate(v).astype(np.float64))
        for k, v in per_query.items()
    }
    return hist_pos, hist_neg, {
        'pos_sum': pos_sum, 'neg_sum': neg_sum,
        'pos_sumsq': pos_sumsq, 'neg_sumsq': neg_sumsq,
    }, per_query


def density(counts, bins):
    total = int(counts.sum())
    if total == 0:
        raise RuntimeError('Cannot plot an empty pair distribution.')
    return counts.astype(np.float64) / (total * np.diff(bins))


def hist_percentile(counts, bins, percentile):
    total = int(counts.sum())
    target = np.clip(percentile / 100.0, 0.0, 1.0) * total
    cumulative = np.cumsum(counts, dtype=np.int64)
    idx = min(int(np.searchsorted(cumulative, target, side='left')), counts.size - 1)
    before = int(cumulative[idx - 1]) if idx else 0
    fraction = 0.0 if counts[idx] == 0 else (target - before) / counts[idx]
    return float(bins[idx] + fraction * (bins[idx + 1] - bins[idx]))


def smooth(y, sigma):
    return gaussian_filter1d(y, sigma) if sigma > 0 else y


def analyse(path, args, bins, device):
    exclude = args.positive_policy == 'cross_camera'
    q_feat, g_feat, q_pid, g_pid, q_cam, g_cam = load_feature_mat(path, need_cam=exclude)
    keep = queries_with_positive(q_pid, g_pid, q_cam, g_cam, exclude)
    dropped = int((~keep).sum())
    if dropped and args.positive_policy == 'all_same_id':
        raise RuntimeError(
            f'{path}: {dropped:,} queries have no same-PID gallery image. '
            'The complete Query-Gallery protocol requires every query identity to occur in the gallery; '
            'check the MAT split/labels instead of silently dropping those queries.')
    if dropped:
        print(f'[Info] {path}: dropping {dropped:,}/{keep.size:,} queries without any valid '
              f'intra-class gallery match (they are skipped by the evaluation as well)')
        q_feat, q_pid, q_cam = q_feat[keep], q_pid[keep], q_cam[keep]
    audit = pair_audit(q_pid, g_pid, q_cam, g_cam, exclude)
    audit['dropped_queries'] = dropped
    print_audit(path, audit, args.query_domain, args.gallery_domain, args.positive_policy)
    h_pos, h_neg, moments, per_query = histogram_pairs(
        q_feat, g_feat, q_pid, g_pid, q_cam, g_cam, exclude,
        args.metric, bins, args.chunk, device)
    if int(h_pos.sum()) != audit['positive_pairs']:
        raise AssertionError(f'Positive histogram count {int(h_pos.sum()):,} != '
                             f'audit count {audit["positive_pairs"]:,}')
    if int(h_neg.sum()) != audit['negative_pairs']:
        raise AssertionError(f'Negative histogram count {int(h_neg.sum()):,} != '
                             f'audit count {audit["negative_pairs"]:,}')

    n_pos, n_neg = audit['positive_pairs'], audit['negative_pairs']
    if args.pair_mode == 'hard':
        # One value per query: chosen positive distance vs. nearest negative distance.
        pos_vals, neg_vals = per_query[args.hard_pos], per_query['neg_nearest']
        if not np.all(np.isfinite(neg_vals)):
            raise RuntimeError('Some queries have no negative gallery image.')
        h_pos = np.histogram(pos_vals, bins=bins)[0]
        h_neg = np.histogram(neg_vals, bins=bins)[0]
        moments = {'pos_sum': float(pos_vals.sum()), 'neg_sum': float(neg_vals.sum()),
                   'pos_sumsq': float(np.square(pos_vals).sum()),
                   'neg_sumsq': float(np.square(neg_vals).sum())}
        n_pos, n_neg = int(pos_vals.size), int(neg_vals.size)
        print(f'  Hard mode                  : {args.hard_pos} positive vs. nearest negative, '
              f'{n_pos:,} queries')
    rank1 = float(np.mean(per_query['rank1_hit']))
    print(f'  Rank-1 from sorted gallery : {100 * rank1:.2f}% '
          f'(diagnostic only; the histogram uses all pairs)')

    centers = (bins[:-1] + bins[1:]) / 2.0
    y_pos = density(h_pos, bins)
    y_neg = density(h_neg, bins)
    sigma = args.smooth_sigma * (args.num_bins / 400.0 if args.pair_mode == 'hard' else 1.0)
    ys_pos = smooth(y_pos, sigma)
    ys_neg = smooth(y_neg, sigma)
    pos_peak = float(centers[int(np.argmax(ys_pos))])
    neg_peak = float(centers[int(np.argmax(ys_neg))])
    lo = hist_percentile(h_pos, bins, args.range_pct[0])
    hi = hist_percentile(h_pos, bins, args.range_pct[1])
    print(f'  Histogram verification     : {int(h_pos.sum()):,} + {int(h_neg.sum()):,} '
          f'= {int(h_pos.sum() + h_neg.sum()):,}')
    print(f'  Intra peak / Inter peak    : {pos_peak:.4f} / {neg_peak:.4f}')
    print(f'  Peak gap                   : {neg_peak - pos_peak:.4f}')
    return {
        'pos': h_pos, 'neg': h_neg, 'y_pos': y_pos, 'y_neg': y_neg,
        'ys_pos': ys_pos, 'ys_neg': ys_neg, 'pos_peak': pos_peak,
        'neg_peak': neg_peak, 'range_lo': lo, 'range_hi': hi,
        'audit': audit, 'moments': moments, 'bins': bins,
        'n_pos': n_pos, 'n_neg': n_neg, 'per_query': per_query,
        'pos_mean': moments['pos_sum'] / n_pos,
        'neg_mean': moments['neg_sum'] / n_neg,
    }


INTRA_FILL = '#F2BA02'
INTER_FILL = '#F6BC8F'
INTRA_TEXT = '#B8860B'
INTER_TEXT = '#D97841'
OVERLAP_COLOR = '#D62728'

def distribution_stats(result, bins):
    """Separability statistics computed from the exact pair moments and histograms."""
    audit, m = result['audit'], result['moments']
    n_pos, n_neg = result['n_pos'], result['n_neg']
    pq = result['per_query']
    top1_margin = pq['neg_nearest'] - pq['nearest']
    mu_pos, mu_neg = m['pos_sum'] / n_pos, m['neg_sum'] / n_neg
    var_pos = max(m['pos_sumsq'] / n_pos - mu_pos ** 2, 0.0)
    var_neg = max(m['neg_sumsq'] / n_neg - mu_neg ** 2, 0.0)
    pooled = math.sqrt(0.5 * (var_pos + var_neg))
    d_prime = (mu_neg - mu_pos) / pooled if pooled > 0 else float('nan')

    overlap = float(np.sum(np.minimum(result['y_pos'], result['y_neg']) * np.diff(bins)))
    p_pos = result['pos'] / float(n_pos)
    p_neg = result['neg'] / float(n_neg)
    cdf_pos = np.cumsum(p_pos)
    # AUC = P(inter distance > intra distance); ties inside one bin count as 0.5.
    auc = float(np.sum(p_neg * (cdf_pos - 0.5 * p_pos)))
    # Verification at threshold t: accept a pair when distance <= t.
    far = np.cumsum(p_neg)
    frr = 1.0 - cdf_pos
    idx = int(np.argmin(np.abs(far - frr)))
    gap = result['neg_peak'] - result['pos_peak']
    rng = result['range_hi'] - result['range_lo']
    return {
        'queries_used': audit['query_images'], 'dropped_queries': audit.get('dropped_queries', 0),
        'intra_pairs': n_pos, 'inter_pairs': n_neg,
        'intra_peak': result['pos_peak'], 'inter_peak': result['neg_peak'],
        'peak_gap': gap, 'intra_range': rng,
        'gap_over_range': gap / rng if rng > 0 else float('nan'),
        'intra_mean': mu_pos, 'intra_std': math.sqrt(var_pos),
        'inter_mean': mu_neg, 'inter_std': math.sqrt(var_neg),
        'd_prime': d_prime, 'overlap': overlap, 'auc': auc,
        'eer': max(0.0, float(0.5 * (far[idx] + frr[idx]))), 'eer_threshold': float(bins[idx + 1]),
        'rank1_check': float(np.mean(pq['rank1_hit'])),
        'mean_top1_margin': float(np.mean(top1_margin)),
    }


def report_stats(results, titles, args):
    keys = ['queries_used', 'dropped_queries', 'intra_pairs', 'inter_pairs',
            'intra_peak', 'inter_peak', 'peak_gap', 'intra_range', 'gap_over_range',
            'intra_mean', 'intra_std', 'inter_mean', 'inter_std',
            'd_prime', 'overlap', 'auc', 'eer', 'eer_threshold',
            'rank1_check', 'mean_top1_margin']
    better = {'peak_gap': 'higher', 'intra_range': 'lower', 'gap_over_range': 'higher',
              'd_prime': 'higher', 'overlap': 'lower', 'auc': 'higher', 'eer': 'lower',
              'rank1_check': 'higher', 'mean_top1_margin': 'higher'}
    rows = [distribution_stats(r, r['bins']) for r in results]

    width = max(14, max(len(t) for t in titles) + 2)
    line = '=' * (18 + width * len(titles))
    mode = 'all pairs' if args.pair_mode == 'all' else f'hard: {args.hard_pos} pos vs nearest neg'
    print(f'\n{line}\nSEPARABILITY SUMMARY [{mode}] (range = P{args.range_pct[0]:g}-P{args.range_pct[1]:g} '
          f'of intra-class distances; positive policy: {args.positive_policy})\n{line}')
    print(f'{"metric":<18}' + ''.join(f'{t:>{width}}' for t in titles))
    for key in keys:
        tag = f' ({better[key][0]})' if key in better else ''
        cell = (lambda v: f'{v:>{width},}') if isinstance(rows[0][key], int) else \
            (lambda v: f'{v:>{width}.4f}')
        print(f'{key + tag:<18}' + ''.join(cell(row[key]) for row in rows))
    print(f'{line}\n(h) higher is better, (l) lower is better\n')

    output = f'{args.save_path}_stats.csv'
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'mat_file'] + keys)
        for title, path, row in zip(titles, args.mat_files, rows):
            writer.writerow([title, path] + [row[k] if isinstance(row[k], int) else f'{row[k]:.6f}'
                                             for k in keys])
    print(f'==> Saved: {output}')


def configure_font(args):
    if args.font_path:
        if not os.path.isfile(args.font_path):
            raise FileNotFoundError(args.font_path)
        font_manager.fontManager.addfont(args.font_path)
        prop = font_manager.FontProperties(fname=args.font_path)
        plt.rcParams['font.family'] = prop.get_name()
    else:
        plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams.update({
        'mathtext.fontset': 'stix',              # Times-like math text for d' and bold numbers
        'hatch.linewidth': 0.8,
        'font.size': 12, 'axes.linewidth': 1.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType fonts (no Type 3) for papers
    })


def plot(results, titles, bins, args):
    configure_font(args)
    centers = (bins[:-1] + bins[1:]) / 2.0
    pw, ph = args.panel_size
    fig, axes = plt.subplots(1, len(results), figsize=(pw * len(results), ph), squeeze=False)
    axes = axes[0]

    if args.xlim is None:
        lo = min(hist_percentile(r['pos'], bins, 0.2) for r in results)
        hi = max(hist_percentile(r['neg'], bins, 99.8) for r in results)
        pad = max(0.03, 0.06 * (hi - lo))
        xlim = (max(0.0, lo - pad), min(2.0, hi + pad))
    else:
        xlim = tuple(args.xlim)
    xspan = xlim[1] - xlim[0]
    num = f'{{:.{args.decimals}f}}'.format
    box = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.8)
    xlabel = args.xlabel or ('Feature Distance' if args.metric == 'euclidean' else 'Cosine Distance')
    if args.pair_mode == 'hard':
        intra_label = {'nearest': 'Nearest intra-class', 'farthest': 'Farthest intra-class',
                       'mean': 'Mean intra-class'}[args.hard_pos]
        inter_label = 'Nearest inter-class'
    else:
        intra_label, inter_label = 'Intra-class', 'Inter-class'
    global_ymax = max(max(float(r['y_pos'].max()), float(r['y_neg'].max())) for r in results)

    for i, (ax, result, title) in enumerate(zip(axes, results, titles)):
        # Filled histograms, as in the reference figure.
        ax.hist(centers, bins=bins, weights=result['y_pos'], histtype='stepfilled',
                color=INTRA_FILL, alpha=args.fill_alpha, label=intra_label, zorder=1)
        ax.hist(centers, bins=bins, weights=result['y_neg'], histtype='stepfilled',
                color=INTER_FILL, alpha=args.fill_alpha, label=inter_label, zorder=1)
        if args.show_curve:
            ax.plot(centers, result['ys_pos'], color=INTRA_FILL, lw=1.5, zorder=2)
            ax.plot(centers, result['ys_neg'], color=INTER_FILL, lw=1.5, zorder=2)

        ymax = max(float(result['y_pos'].max()), float(result['y_neg'].max()))
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.08 * (global_ymax if args.sharey else ymax))
        ax.set_xlabel(xlabel)
        # Repeat the y-axis label on every panel so each exported/cropped subplot is self-contained.
        ax.set_ylabel(args.ylabel)
        ax.set_title(title)
        leg = ax.legend(loc=args.legend_loc, frameon=True, framealpha=1.0,
                        edgecolor='0.8', fancybox=False, handlelength=1.6)
        leg.set_zorder(10)

        # The only numerical annotations retained in the paper figure are the exact
        # intra-class and inter-class mean distances and their dashed mean lines.
        pp, npk = result['pos_mean'], result['neg_mean']
        pp_r, npk_r = round(pp, args.decimals), round(npk, args.decimals)
        h_pos = float(result['ys_pos'].max())
        h_neg = float(result['ys_neg'].max())
        dx = 0.012 * xspan
        ax.axvline(pp, color=INTRA_TEXT, ls='--', lw=1.0, zorder=3)
        ax.axvline(npk, color=INTER_TEXT, ls='--', lw=1.0, zorder=3)

        y_intra = min(max(0.78 * h_pos, 0.42 * ymax), 0.78 * ymax)
        y_inter = min(max(0.88 * h_neg, 0.72 * ymax), 0.92 * ymax)
        ax.text(pp - dx, y_intra, f'Intra Mean: {num(pp_r)}', ha='right', va='center',
                color=INTRA_TEXT, fontsize=11, bbox=box, zorder=5)

        # Put the inter-mean label on the side of its mean line that has enough room.
        # This is important for Night600/Triplet: the inter mean is near the centre
        # of the panel, so a left-aligned label would otherwise be hidden by the
        # upper-left legend.  High-distance means keep the label on the left so the
        # text cannot run beyond the right edge of the axes.
        estimated_text_width = 0.25 * xspan
        right_room = xlim[1] - npk
        legend_on_left = 'left' in args.legend_loc.lower()
        place_inter_on_right = legend_on_left and right_room >= estimated_text_width
        inter_x = npk + dx if place_inter_on_right else npk - dx
        inter_ha = 'left' if place_inter_on_right else 'right'
        ax.text(inter_x, y_inter, f'Inter Mean: {num(npk_r)}',
                ha=inter_ha, va='center',
                color=INTER_TEXT, fontsize=11, bbox=box, zorder=5)

        # Mean-distance gap retained from the reference figure: a black double arrow
        # joins the two mean lines, and the displayed value equals the difference of
        # the two displayed (rounded) means.
        mean_gap = npk_r - pp_r
        # Keep the gap arrow below the inter-mean label.  The previous 0.78*ymax
        # upper bound made the arrow value collide with the high Night600/Triplet
        # annotation when four decimal places were displayed.
        y_gap = min(max(h_pos + 0.05 * ymax, 0.52 * ymax), 0.72 * ymax)
        ax.annotate('', xy=(pp, y_gap), xytext=(npk, y_gap), zorder=4,
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.0,
                                    shrinkA=0, shrinkB=0))
        ax.text((pp + npk) / 2.0, y_gap + 0.020 * ymax, num(mean_gap),
                ha='center', va='bottom', color='black', fontsize=11,
                bbox=box, zorder=5)

    fig.tight_layout(pad=1.0)
    parent = os.path.dirname(os.path.abspath(args.save_path))
    os.makedirs(parent, exist_ok=True)
    for fmt in args.formats:
        output = f'{args.save_path}.{fmt}'
        fig.savefig(output, dpi=args.dpi if fmt == 'png' else None,
                    bbox_inches='tight', format=fmt)
        print(f'==> Saved: {output}')
    plt.close(fig)


def main():
    args = parse_args()
    print(f'==> Plot version: {PLOT_VERSION}')
    titles = args.titles
    if args.num_bins is None:
        args.num_bins = 400 if args.pair_mode == 'all' else 100
    if args.num_bins < 10:
        raise ValueError('--num_bins must be at least 10')
    if not (0 <= args.range_pct[0] < args.range_pct[1] <= 100):
        raise ValueError('--range_pct must satisfy 0 <= LOW < HIGH <= 100')
    device = choose_device(args.device)
    print(f'==> Pair distance device: {device}')
    bins = np.linspace(0.0, 2.0, args.num_bins + 1)
    print(f'==> Pair mode: {args.pair_mode}' +
          (f' ({args.hard_pos} positive vs. nearest negative)' if args.pair_mode == 'hard' else ''))
    print(f'==> Pair protocol: {args.query_domain} x {args.gallery_domain}; '
          f'positive_policy={args.positive_policy}')
    results = [analyse(path, args, bins, device) for path in args.mat_files]
    plot(results, titles, bins, args)
    report_stats(results, titles, args)


if __name__ == '__main__':
    main()
