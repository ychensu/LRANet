import torch
import numpy as np
import torch.nn.functional as F
from mmocr.core.evaluation.utils import boundary_iou

def poly_nms(polygons, threshold, with_index=False):
    assert isinstance(polygons, list)
    keep_poly = []
    keep_index = []
    if len(polygons) != 0:
        polygons = np.array(polygons)
        scores = polygons[:, -1]
        sorted_index = np.argsort(scores)
        polygons = polygons[sorted_index]

        index = [i for i in range(polygons.shape[0])]
        vaild_index = np.ones(len(index))
        invalid_index = np.where(vaild_index==0)
        index = np.delete(index, invalid_index)

        while len(index) > 0:
            keep_poly.append(polygons[index[-1]].tolist())
            keep_index.append(sorted_index[index[-1]])
            A = polygons[index[-1]][:-1]
            index = np.delete(index, -1)

            iou_list = np.zeros((len(index), ))
            for i in range(len(index)):
                B = polygons[index[i]][:-1]
                iou_list[i] = boundary_iou(A, B)
            remove_index = np.where(iou_list > threshold)
            index = np.delete(index, remove_index)

    if with_index:
        return keep_poly, keep_index
    else:
        return keep_poly


def lra_decode( preds, U_t,
                        scale,
                        score_thr=0.1,
                        shift=0.1,
                        ):
    assert isinstance(preds, list)

    tr_pred = preds[0][0][0].sigmoid()
    ssr_pred = preds[2][0][0].sigmoid()
    score_pred = tr_pred * ssr_pred   
    tr_pred_mask = score_pred > score_thr 

    boundaries = []
    reg_pred = preds[3][0].permute(1, 2, 0)
    lra_pred = reg_pred.flatten(0,1)
    lra_c = lra_pred[tr_pred_mask.reshape(-1)]
    rows, cols = tr_pred_mask.nonzero(as_tuple=True)
    xy_text = torch.stack((rows, cols), dim=1)
    polygons = torch.matmul(lra_c, U_t)
    polygons = polygons.reshape(-1,polygons.shape[-1]//2,2)
    score = score_pred[tr_pred_mask].reshape(-1, 1)
    boundaries = []
    if polygons.shape[0] > 0:
        polygons[:, :, 0] += (xy_text[:, 1, None])
        polygons[:, :, 1] += (xy_text[:, 0, None])
        polygons += shift
        polygons = polygons.reshape(polygons.shape[0], -1) * scale
        polygons2 = torch.cat((polygons, score), dim=1)
        polygons2 = polygons2.data.cpu().numpy().tolist() 
        boundaries = boundaries + polygons2

    return boundaries
