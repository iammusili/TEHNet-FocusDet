"""Python port of the VisDrone-DET toolkit evaluation (ignored regions, matching, AP/AR)."""
import numpy as np
import os
from PIL import Image


def find_image_list(gt_path):
    files = sorted(os.listdir(gt_path))
    return [f for f in files if f not in ('.', '..')]


def create_int_img(img):
    return np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)


def load_gt_file(filepath):
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float64)
    if data.ndim == 1 and data.size > 0:
        data = data.reshape(1, -1)
    return data


def load_det_file(filepath):
    if not os.path.exists(filepath):
        return np.empty((0, 8))
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    delimiter = ',' if first_line and ',' in first_line else None
    data = np.loadtxt(filepath, dtype=np.float64, delimiter=delimiter)
    if data.ndim == 1 and data.size > 0:
        data = data.reshape(1, -1)
    return data


def drop_objects_in_igr(gt, det, img_height, img_width):
    idx_fr = gt[:, 5] != 0
    curgt = gt[idx_fr].copy()
    idx_igr = gt[:, 5] == 0
    igr_region = np.maximum(1, gt[idx_igr, :4]) if np.any(idx_igr) else np.empty((0, 4))

    if igr_region.shape[0] > 0:
        igr_map = np.zeros((img_height, img_width), dtype=np.float64)
        for j in range(igr_region.shape[0]):
            r1 = int(igr_region[j, 1]) - 1
            r2 = min(img_height, int(igr_region[j, 1] + igr_region[j, 3]))
            c1 = int(igr_region[j, 0]) - 1
            c2 = min(img_width, int(igr_region[j, 0] + igr_region[j, 2]))
            igr_map[r1:r2, c1:c2] = 1.0
        int_igr_map = create_int_img(igr_map)

        def _filter_rows(data):
            if data.shape[0] == 0:
                return np.empty((0, data.shape[1]))
            pos = np.maximum(1, np.round(data[:, :4])).astype(int)
            x = np.maximum(1, np.minimum(img_width, pos[:, 0]))
            y = np.maximum(1, np.minimum(img_height, pos[:, 1]))
            w = pos[:, 2]
            h = pos[:, 3]
            tl = int_igr_map[y - 1, x - 1]
            tr = int_igr_map[y - 1, np.minimum(img_width, x + w) - 1]
            bl_idx = np.maximum(1, np.minimum(img_height, y + h)) - 1
            br_x = np.minimum(img_width, x + w) - 1
            bl = int_igr_map[bl_idx, x - 1]
            br = int_igr_map[bl_idx, br_x]
            igr_val = tl + br - tr - bl
            keep = igr_val / (h * w) < 0.5
            return data[keep]

        curgt = _filter_rows(curgt)
        det = _filter_rows(det)

    newgt = curgt
    newdet = det
    return newgt, newdet


def _get_img_path(dataset_path, name_img):
    img_dir = os.path.join(dataset_path, 'images')
    if os.path.isdir(img_dir):
        return os.path.join(img_dir, name_img[:-4] + '.jpg')
    return os.path.join(dataset_path, name_img[:-4] + '.jpg')


def _get_img_size(dataset_path, name_img):
    p = _get_img_path(dataset_path, name_img)
    img_pil = Image.open(p)
    img_w, img_h = img_pil.size
    return img_h, img_w


def save_anno_res(gt_path, res_path, dataset_path, num_imgs, name_imgs):
    allgt = [None] * num_imgs
    alldet = [None] * num_imgs
    for id_img in range(num_imgs):
        oldgt = load_gt_file(os.path.join(gt_path, name_imgs[id_img]))
        olddet = load_det_file(os.path.join(res_path, name_imgs[id_img]))
        if oldgt.size == 0:
            oldgt = np.empty((0, 8))
        if olddet.size == 0:
            olddet = np.empty((0, 8))

        img_h, img_w = _get_img_size(dataset_path, name_imgs[id_img])

        newgt, det = drop_objects_in_igr(oldgt, olddet, img_h, img_w)
        if det.shape[0] > 0:
            det = det[np.argsort(-det[:, 4])]
        gt = newgt.copy()
        if gt.shape[0] > 0:
            col5_orig = newgt[:, 4].copy()
            gt[col5_orig == 0, 4] = 1
            gt[col5_orig == 1, 4] = 0
        allgt[id_img] = gt
        alldet[id_img] = det
    return allgt, alldet


def comp_oas(dt, gt, ig=None):
    m = dt.shape[0]
    n = gt.shape[0]
    if m == 0 or n == 0:
        return np.zeros((m, n))
    oa = np.zeros((m, n))
    if ig is None:
        ig = np.zeros(n, dtype=bool)

    de_x2 = dt[:, 0] + dt[:, 2]
    de_y2 = dt[:, 1] + dt[:, 3]
    da = dt[:, 2] * dt[:, 3]

    ge_x2 = gt[:, 0] + gt[:, 2]
    ge_y2 = gt[:, 1] + gt[:, 3]
    ga = gt[:, 2] * gt[:, 3]

    for i in range(m):
        w = np.minimum(de_x2[i], ge_x2) - np.maximum(dt[i, 0], gt[:, 0])
        h = np.minimum(de_y2[i], ge_y2) - np.maximum(dt[i, 1], gt[:, 1])
        valid = (w > 0) & (h > 0)
        if not np.any(valid):
            continue
        t = np.zeros(n)
        t[valid] = w[valid] * h[valid]
        u = np.zeros(n)
        ig_pos = ig & valid
        not_ig_pos = ~ig & valid
        u[ig_pos] = da[i]
        u[not_ig_pos] = da[i] + ga[not_ig_pos] - t[not_ig_pos]
        nonzero = u > 0
        oa[i, nonzero] = t[nonzero] / u[nonzero]
    return oa


def eval_res(gt0, dt0, thr):
    if gt0.size == 0:
        gt0 = np.zeros((0, 5))
    if dt0.size == 0:
        dt0 = np.zeros((0, 5))

    if gt0.ndim == 1 and gt0.size > 0:
        gt0 = gt0.reshape(1, -1)
    if dt0.ndim == 1 and dt0.size > 0:
        dt0 = dt0.reshape(1, -1)

    if gt0.shape[0] == 0 and gt0.size == 0:
        gt0 = np.zeros((0, 5))
    if dt0.shape[0] == 0 and dt0.size == 0:
        dt0 = np.zeros((0, 5))

    nd = dt0.shape[0]
    ng = gt0.shape[0]

    if nd > 0 and dt0.shape[1] >= 5:
        ord_idx = np.argsort(-dt0[:, 4])
        dt0 = dt0[ord_idx]
    if ng > 0 and gt0.shape[1] >= 5:
        ord_idx = np.argsort(gt0[:, 4])
        gt0 = gt0[ord_idx]

    gt = gt0.copy()
    dt = dt0.copy()
    dt = np.column_stack([dt, np.zeros(nd)])

    if ng > 0:
        gt[:, 4] = -gt[:, 4]

    if nd > 0 and ng > 0:
        oa = comp_oas(dt[:, :4], gt[:, :4], gt[:, 4] == -1)
        for d in range(nd):
            bst_oa = thr
            bstg = -1
            bstm = 0
            for g in range(ng):
                m = gt[g, 4]
                if m == 1:
                    continue
                if bstm != 0 and m == -1:
                    break
                if oa[d, g] < bst_oa:
                    continue
                bst_oa = oa[d, g]
                bstg = g
                if m == 0:
                    bstm = 1
                else:
                    bstm = -1
            g = bstg
            m = bstm
            if m == -1:
                dt[d, 5] = m
            elif m == 1:
                gt[g, 4] = m
                dt[d, 5] = m

    return gt, dt


def voc_ap(rec, prec):
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_arr = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[i_arr] - mrec[i_arr - 1]) * mpre[i_arr])
    return ap


def _calc_accuracy_core(num_imgs, allgt, alldet, id_class, thr, max_dets, sml_mode=False):
    gt_match = []
    det_match = []
    for id_img in range(num_imgs):
        gt = allgt[id_img]
        det = alldet[id_img]
        idx_gt_cur_class = gt[:, 5] == id_class
        n_det = min(det.shape[0], max_dets)
        det_sub = det[:n_det]
        idx_det_cur_class = det_sub[:, 5] == id_class
        gt0 = gt[idx_gt_cur_class, :5]
        dt0 = det_sub[idx_det_cur_class, :5]
        gt1, dt1 = eval_res(gt0, dt0, thr)
        if gt1.shape[0] > 0:
            gt_match.append(gt1[:, 4])
        if sml_mode:
            dt0_area_mark = det_sub[idx_det_cur_class, 8]
            if dt1.shape[0] > 0:
                for temp_i in range(dt1.shape[0]):
                    area_mark = dt0_area_mark[temp_i]
                    if dt1[temp_i, 5] or area_mark:
                        det_match.append(dt1[temp_i, 4:6].reshape(1, 2))
        else:
            if dt1.shape[0] > 0:
                det_match.append(dt1[:, 4:6])
    if gt_match:
        gt_match = np.concatenate(gt_match)
    else:
        gt_match = np.array([])
    if det_match:
        det_match = np.concatenate(det_match)
    else:
        det_match = np.empty((0, 2)) if not sml_mode else np.empty((0, 2))
    return gt_match, det_match


def calc_accuracy_full(num_imgs, allgt, alldet, max_dets_list, sml_mode=False, verbose=True):
    n_classes = 10
    n_thresholds = 10
    n_maxdets = len(max_dets_list)
    AP = np.zeros((n_classes, n_thresholds, n_maxdets))
    AR = np.zeros((n_classes, n_thresholds, n_maxdets))
    eval_class = []
    thresholds = np.arange(0.5, 1.0, 0.05)

    for id_class in range(1, n_classes + 1):
        if verbose:
            print(f'evaluating object category {id_class}/10...')
        for id_img in range(num_imgs):
            gt = allgt[id_img]
            if gt.shape[0] > 0 and np.any(gt[:, 5] == id_class):
                eval_class.append(id_class)

        for y_idx, max_dets in enumerate(max_dets_list):
            precomputed = []
            for id_img in range(num_imgs):
                gt = allgt[id_img]
                det = alldet[id_img]
                idx_gt_cur_class = gt[:, 5] == id_class
                n_det = min(det.shape[0], max_dets)
                det_sub = det[:n_det]
                idx_det_cur_class = det_sub[:, 5] == id_class
                gt0 = gt[idx_gt_cur_class, :5]
                dt0 = det_sub[idx_det_cur_class, :5]
                dt0_area_mark = det_sub[idx_det_cur_class, 8] if sml_mode else None

                if gt0.size == 0:
                    gt0 = np.zeros((0, 5))
                if dt0.size == 0:
                    dt0 = np.zeros((0, 5))
                if gt0.ndim == 1 and gt0.size > 0:
                    gt0 = gt0.reshape(1, -1)
                if dt0.ndim == 1 and dt0.size > 0:
                    dt0 = dt0.reshape(1, -1)
                nd = dt0.shape[0]
                ng = gt0.shape[0]
                if nd > 0:
                    ord_dt = np.argsort(-dt0[:, 4])
                    dt0 = dt0[ord_dt]
                if ng > 0:
                    ord_gt = np.argsort(gt0[:, 4])
                    gt0 = gt0[ord_gt]
                gt_s = gt0.copy()
                dt_s = np.column_stack([dt0.copy(), np.zeros(nd)]) if nd > 0 else np.zeros((0, 6))
                if ng > 0:
                    gt_s[:, 4] = -gt_s[:, 4]
                ig = (gt_s[:, 4] == -1) if ng > 0 else np.array([], dtype=bool)
                oa = comp_oas(dt_s[:, :4], gt_s[:, :4], ig) if (nd > 0 and ng > 0) else np.zeros((nd, ng))
                precomputed.append((gt_s, dt_s, oa, ng, nd, dt0_area_mark))

            for x, thr in enumerate(thresholds):
                gt_match_all = []
                det_match_all = []
                for id_img in range(num_imgs):
                    gt_s, dt_s, oa, ng, nd, dt0_area_mark = precomputed[id_img]
                    gt_r = gt_s.copy()
                    dt_r = dt_s.copy()
                    for d in range(nd):
                        bst_oa = thr
                        bstg = -1
                        bstm = 0
                        for g in range(ng):
                            m = gt_r[g, 4]
                            if m == 1:
                                continue
                            if bstm != 0 and m == -1:
                                break
                            if oa[d, g] < bst_oa:
                                continue
                            bst_oa = oa[d, g]
                            bstg = g
                            if m == 0:
                                bstm = 1
                            else:
                                bstm = -1
                        if bstm == -1:
                            dt_r[d, 5] = -1
                        elif bstm == 1:
                            gt_r[bstg, 4] = 1
                            dt_r[d, 5] = 1
                    if gt_r.shape[0] > 0:
                        gt_match_all.append(gt_r[:, 4])
                    if sml_mode:
                        if dt_r.shape[0] > 0:
                            for temp_i in range(dt_r.shape[0]):
                                area_mark = dt0_area_mark[temp_i]
                                if dt_r[temp_i, 5] or area_mark:
                                    det_match_all.append(dt_r[temp_i, 4:6].reshape(1, 2))
                    else:
                        if dt_r.shape[0] > 0:
                            det_match_all.append(dt_r[:, 4:6])

                if gt_match_all:
                    gt_match = np.concatenate(gt_match_all)
                else:
                    gt_match = np.array([])
                if det_match_all:
                    det_match = np.concatenate(det_match_all)
                else:
                    det_match = np.empty((0, 2))
                if det_match.shape[0] > 0:
                    idrank = np.argsort(-det_match[:, 0])
                    tp = np.cumsum(det_match[idrank, 1] == 1)
                    fp = np.cumsum(det_match[idrank, 1] == 0)
                    rec = tp / max(1, gt_match.size)
                    prec = tp / np.maximum(1, fp + tp)
                    AR[id_class - 1, x, y_idx] = np.max(rec) * 100 if rec.size > 0 else 0
                    AP[id_class - 1, x, y_idx] = voc_ap(rec, prec) * 100 if rec.size > 0 else 0

    return AP, AR, eval_class
