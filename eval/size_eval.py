"""S/M/L evaluation with MS COCO semantics (cocoeval.evaluateImg), applied to
VisDrone results: GT outside the size range is marked as ignore (not removed),
unmatched detections outside the range are ignored, ranges are lo <= area <= hi,
AP uses 101-point interpolation and is macro-averaged over classes.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# areaRng profiles
#   每个 profile: (labels, ranges)  ranges[i] = [lo, hi]，第 0 个恒为 'all'。
#   单位均为 pixel²（即便概念上用 sqrt(area)，也换算成平方区间，与 cocoapi 一致）。
# ----------------------------------------------------------------------------
INF = 1e10  # 与 cocoapi 的 1e5**2 量级一致，足够覆盖任意图像中的目标

PROFILES = {
    'coco': (
        ['all', 'small', 'medium', 'large'],
        [[0, INF], [0, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, INF]],
    ),
}

DEFAULT_IOU_THRS = np.round(np.arange(0.5, 1.0, 0.05), 2)  # 0.50:0.05:0.95 共 10 档


# ----------------------------------------------------------------------------
# IoU
# ----------------------------------------------------------------------------
def bbox_iou_matrix(dt_xywh: np.ndarray, gt_xywh: np.ndarray) -> np.ndarray:
    """计算 D×G 的 IoU 矩阵。bbox 格式 [x, y, w, h]，采用连续坐标（无 +1 像素约定，
    与 COCO / 原 comp_oas 一致）。"""
    D = dt_xywh.shape[0]
    G = gt_xywh.shape[0]
    if D == 0 or G == 0:
        return np.zeros((D, G), dtype=np.float64)

    dx1, dy1 = dt_xywh[:, 0], dt_xywh[:, 1]
    dx2, dy2 = dx1 + dt_xywh[:, 2], dy1 + dt_xywh[:, 3]
    da = dt_xywh[:, 2] * dt_xywh[:, 3]

    gx1, gy1 = gt_xywh[:, 0], gt_xywh[:, 1]
    gx2, gy2 = gx1 + gt_xywh[:, 2], gy1 + gt_xywh[:, 3]
    ga = gt_xywh[:, 2] * gt_xywh[:, 3]

    iou = np.zeros((D, G), dtype=np.float64)
    for i in range(D):
        iw = np.minimum(dx2[i], gx2) - np.maximum(dx1[i], gx1)
        ih = np.minimum(dy2[i], gy2) - np.maximum(dy1[i], gy1)
        valid = (iw > 0) & (ih > 0)
        if not np.any(valid):
            continue
        inter = np.zeros(G)
        inter[valid] = iw[valid] * ih[valid]
        union = da[i] + ga - inter
        nz = union > 0
        iou[i, nz] = inter[nz] / union[nz]
    return iou


def _in_range(area: np.ndarray, lo: float, hi: float, boundary: str) -> np.ndarray:
    if boundary == 'left_closed':       # COCO: lo <= area <= hi
        return (area >= lo) & (area <= hi)
    elif boundary == 'left_open':       # lo < area <= hi
        return (area > lo) & (area <= hi)
    raise ValueError(f'unknown boundary mode: {boundary}')


# ----------------------------------------------------------------------------
# COCO 风格单图匹配（evaluateImg 的精确语义）
# ----------------------------------------------------------------------------
def _evaluate_img(gt, dt, ious, iou_thr, lo, hi, boundary):
    """对单张图、单类别、单 area 档、单 IoU 阈值做匹配。

    输入
      gt:  (G,6) [x,y,w,h,cat,ignore]，ignore=1 表示 VisDrone "ignore 区域/忽略标注"
      dt:  (D,6) [x,y,w,h,cat,score]，**须已按 score 降序并截断到 maxDets**
      ious:(D,G) 预计算 IoU
    返回 (dt_scores(D,), dt_tp(D,bool), dt_fp(D,bool), n_pos(int))
      n_pos = 该图该类该档位内"非 ignore" GT 的数量（用作召回分母）
    """
    G = gt.shape[0]
    D = dt.shape[0]

    if G > 0:
        gt_area = gt[:, 2] * gt[:, 3]
        # GT ignore = 原 ignore 标注  OR  面积不在当前档位内
        gt_ig = (gt[:, 5] > 0) | (~_in_range(gt_area, lo, hi, boundary))
        # 把非 ignore 的 GT 排前面（COCO: gtind = argsort(gtIg)），保证 break 逻辑成立
        order = np.argsort(gt_ig, kind='stable')
        gt_ig = gt_ig[order]
        if D > 0:
            ious = ious[:, order]
    else:
        gt_ig = np.zeros(0, dtype=bool)

    n_pos = int(np.sum(~gt_ig)) if G > 0 else 0

    dt_scores = dt[:, 5] if D > 0 else np.zeros(0)
    dt_tp = np.zeros(D, dtype=bool)
    dt_ig = np.zeros(D, dtype=bool)

    if D > 0 and G > 0:
        gt_matched = np.zeros(G, dtype=bool)
        for d in range(D):
            m = -1
            best = min(iou_thr, 1 - 1e-10)
            for g in range(G):
                # 已被非 ignore GT 占用 → 跳过
                if gt_matched[g] and not gt_ig[g]:
                    continue
                # 已匹配到非 ignore GT，后面都是 ignore（已排序）→ 提前结束
                if m > -1 and (not gt_ig[m]) and gt_ig[g]:
                    break
                if ious[d, g] < best:
                    continue
                best = ious[d, g]
                m = g
            if m == -1:
                continue
            dt_ig[d] = gt_ig[m]
            gt_matched[m] = True
            if not gt_ig[m]:
                dt_tp[d] = True

    if D > 0:
        dt_area = dt[:, 2] * dt[:, 3]
        dt_in = _in_range(dt_area, lo, hi, boundary)
        # 未匹配且面积不在档位内 → 忽略（既非 TP 也非 FP），与 COCO 一致
        dt_ig = dt_ig | ((~dt_tp) & (~dt_in) & (dt_area >= 0))
        # 注意：已被判为 TP 的不受影响；匹配到 ignore-GT 的 dt_ig 已为 True

    dt_fp = (~dt_tp) & (~dt_ig)
    return dt_scores, dt_tp, dt_fp, n_pos


def _ap_from_pr(rec, prec, method):
    """由（已按 score 降序累计得到的）recall/precision 数组求 AP。"""
    if rec.size == 0:
        return 0.0
    if method == 'continuous':
        # all-point AUC (same as voc_ap)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
        return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))
    elif method == 'coco101':
        # 101 点插值（COCO 标准）
        rec_thrs = np.linspace(0.0, 1.0, 101)
        mpre = prec.copy()
        for i in range(len(mpre) - 1, 0, -1):
            if mpre[i] > mpre[i - 1]:
                mpre[i - 1] = mpre[i]
        inds = np.searchsorted(rec, rec_thrs, side='left')
        q = np.zeros(101)
        for ri, pi in enumerate(inds):
            if pi < len(mpre):
                q[ri] = mpre[pi]
        return float(np.mean(q))
    raise ValueError(f'unknown ap_method: {method}')


class SizeEvaluator:
    """按 COCO 语义、可配置口径的尺度评估器。"""

    def __init__(self, profile='coco', area_rng=None, area_rng_lbl=None,
                 iou_thrs=None, max_dets=500, cat_ids=tuple(range(1, 11)),
                 boundary='left_closed', class_average='macro', ap_method='coco101'):
        if area_rng is not None:
            assert area_rng_lbl is not None and len(area_rng) == len(area_rng_lbl)
            self.area_rng = [list(r) for r in area_rng]
            self.area_rng_lbl = list(area_rng_lbl)
        else:
            lbls, rngs = PROFILES[profile]
            self.area_rng_lbl = list(lbls)
            self.area_rng = [list(r) for r in rngs]
        self.iou_thrs = np.asarray(iou_thrs if iou_thrs is not None else DEFAULT_IOU_THRS, dtype=np.float64)
        self.max_dets = int(max_dets)
        self.cat_ids = list(cat_ids)
        assert boundary in ('left_closed', 'left_open')
        assert class_average in ('macro', 'weighted')
        assert ap_method in ('coco101', 'continuous')
        self.boundary = boundary
        self.class_average = class_average
        self.ap_method = ap_method

    # -- 主入口 ----------------------------------------------------------------
    def evaluate(self, gts, dets):
        """
        参数
          gts:  list（按图）。每个元素 ndarray (N,6) = [x, y, w, h, category, ignore]
                ignore=1 表示该 GT 为 VisDrone "ignore 区域/被忽略标注"
          dets: list（按图）。每个元素 ndarray (M,6) = [x, y, w, h, category, score]
        返回 dict:
          {
            'AP':  ndarray (n_cat, n_iou, n_area)  每类/每IoU/每档的 AP(%)
            'AR':  ndarray (n_cat, n_iou, n_area)  每类/每IoU/每档的 AR(%)
            'present': ndarray(bool, n_cat) 该类是否在 GT 中出现
            'present_count': ndarray(int, n_cat) 该类出现于多少张图（weighted 平均用）
            'summary': dict  按 area 档位聚合后的 AP@[.5:.95] / AP@.5 / AR@[.5:.95]
            'meta': dict 记录全部口径
          }
        """
        num_imgs = len(gts)
        assert len(dets) == num_imgs
        n_cat = len(self.cat_ids)
        n_iou = len(self.iou_thrs)
        n_area = len(self.area_rng)

        AP = np.zeros((n_cat, n_iou, n_area))
        AR = np.zeros((n_cat, n_iou, n_area))
        valid = np.zeros((n_cat, n_iou, n_area), dtype=bool)  # 该(类,IoU,档)是否有正样本
        present = np.zeros(n_cat, dtype=bool)
        present_count = np.zeros(n_cat, dtype=int)

        # 预排序检测（按类、按 score 降序、截断 maxDets 在每类内部进行）
        for ci, cat in enumerate(self.cat_ids):
            # 该类出现统计
            for im in range(num_imgs):
                g = gts[im]
                if g.shape[0] > 0 and np.any(g[:, 4] == cat):
                    present[ci] = True
                    present_count[ci] += 1

            # 预计算每图该类的 dt（按 score 降序 + 截断）、gt、iou
            per_img = []
            for im in range(num_imgs):
                g = gts[im]
                d = dets[im]
                g_c = g[g[:, 4] == cat] if g.shape[0] > 0 else np.zeros((0, 6))
                d_c = d[d[:, 4] == cat] if d.shape[0] > 0 else np.zeros((0, 6))
                if d_c.shape[0] > 0:
                    d_c = d_c[np.argsort(-d_c[:, 5], kind='stable')]
                    if d_c.shape[0] > self.max_dets:
                        d_c = d_c[:self.max_dets]
                iou = bbox_iou_matrix(d_c[:, :4], g_c[:, :4]) if (d_c.shape[0] and g_c.shape[0]) \
                    else np.zeros((d_c.shape[0], g_c.shape[0]))
                per_img.append((g_c, d_c, iou))

            for ai, (lo, hi) in enumerate(self.area_rng):
                for ti, thr in enumerate(self.iou_thrs):
                    all_scores = []
                    all_tp = []
                    all_fp = []
                    n_pos_total = 0
                    for (g_c, d_c, iou) in per_img:
                        s, tp, fp, npos = _evaluate_img(g_c, d_c, iou, thr, lo, hi, self.boundary)
                        n_pos_total += npos
                        if d_c.shape[0] > 0:
                            all_scores.append(s)
                            all_tp.append(tp)
                            all_fp.append(fp)
                    if n_pos_total == 0:
                        # 该档位无正样本 → 不参与平均（与 COCO 返回 -1 同义）
                        continue
                    valid[ci, ti, ai] = True
                    if all_scores:
                        scores = np.concatenate(all_scores)
                        tps = np.concatenate(all_tp).astype(np.float64)
                        fps = np.concatenate(all_fp).astype(np.float64)
                        order = np.argsort(-scores, kind='stable')
                        tps = np.cumsum(tps[order])
                        fps = np.cumsum(fps[order])
                        rec = tps / n_pos_total
                        prec = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
                        AR[ci, ti, ai] = (rec[-1] if rec.size else 0.0) * 100.0
                        AP[ci, ti, ai] = _ap_from_pr(rec, prec, self.ap_method) * 100.0
                    else:
                        AR[ci, ti, ai] = 0.0
                        AP[ci, ti, ai] = 0.0

        summary = self._summarize(AP, AR, valid, present, present_count)
        meta = {
            'profile_labels': self.area_rng_lbl,
            'area_rng': self.area_rng,
            'iou_thrs': self.iou_thrs.tolist(),
            'max_dets': self.max_dets,
            'cat_ids': self.cat_ids,
            'boundary': self.boundary,
            'class_average': self.class_average,
            'ap_method': self.ap_method,
            'area_source': 'bbox (w*h)',
        }
        return {'AP': AP, 'AR': AR, 'valid': valid, 'present': present,
                'present_count': present_count, 'summary': summary, 'meta': meta}

    # -- 类别 / IoU 聚合 -------------------------------------------------------
    def _summarize(self, AP, AR, valid, present, present_count):
        n_cat, n_iou, n_area = AP.shape
        out = {}
        for ai, lbl in enumerate(self.area_rng_lbl):
            # 每类先在"有效 IoU"上求均值（与 COCO 一致：对该类该档所有 IoU 求均值；
            # 无效的 IoU 档位 AP=0 且 valid=False → 这里按 COCO 把无效记 0 计入 10 档平均）
            # AP@[.5:.95]：对 10 个 IoU 求平均
            ap_per_cat = AP[:, :, ai].mean(axis=1)      # (n_cat,)
            ar_per_cat = AR[:, :, ai].mean(axis=1)      # (n_cat,)
            # AP@.5：取 IoU=0.5（索引 0）
            ap50_per_cat = AP[:, 0, ai]

            # 只在"该类在 GT 中出现过"的类别上做类别平均
            cat_mask = present.copy()
            # 进一步：该类在该档位至少有一个 IoU 有正样本，才纳入（避免某类无该尺寸目标时拉低）
            has_pos_in_area = valid[:, :, ai].any(axis=1)
            cat_mask = cat_mask & has_pos_in_area

            if not np.any(cat_mask):
                out[lbl] = {'AP': 0.0, 'AP50': 0.0, 'AR': 0.0, 'n_classes': 0}
                continue

            if self.class_average == 'macro':
                w = np.ones(n_cat)
            else:  # 'weighted': weighted by the number of images in which a class appears
                w = present_count.astype(np.float64)
            w = w * cat_mask
            wsum = w.sum()
            ap = float((ap_per_cat * w).sum() / wsum) / 100.0
            ar = float((ar_per_cat * w).sum() / wsum) / 100.0
            ap50 = float((ap50_per_cat * w).sum() / wsum) / 100.0
            out[lbl] = {'AP': ap, 'AP50': ap50, 'AR': ar, 'n_classes': int(cat_mask.sum())}
        return out


# ----------------------------------------------------------------------------
# 适配器：把现有 visdrone_eval_utils.save_anno_res 产出的 (allgt, alldet) 转成本模块格式
# ----------------------------------------------------------------------------
def from_visdrone_arrays(allgt, alldet):
    """把现有管线（save_anno_res 之后）的逐图数组转换为 SizeEvaluator 需要的格式。

    现有 save_anno_res 产出的逐图数组列约定（沿用原始 VisDrone 端口）：
      gt 行:  [x, y, w, h, flag, category, ...]   其中 flag: 1=ignore(原 score==0), 0=keep
      det 行: [x, y, w, h, score, category, ...]
    转换为：
      gt 行:  [x, y, w, h, category, ignore]
      det 行: [x, y, w, h, category, score]
    """
    gts, dets = [], []
    for g in allgt:
        if g is None or g.shape[0] == 0:
            gts.append(np.zeros((0, 6)))
            continue
        out = np.zeros((g.shape[0], 6))
        out[:, :4] = g[:, :4]
        out[:, 4] = g[:, 5]          # category
        out[:, 5] = (g[:, 4] == 1)   # ignore flag（save_anno_res 把原 score0 翻成 1）
        gts.append(out)
    for d in alldet:
        if d is None or d.shape[0] == 0:
            dets.append(np.zeros((0, 6)))
            continue
        out = np.zeros((d.shape[0], 6))
        out[:, :4] = d[:, :4]
        out[:, 4] = d[:, 5]          # category
        out[:, 5] = d[:, 4]          # score
        dets.append(out)
    return gts, dets
