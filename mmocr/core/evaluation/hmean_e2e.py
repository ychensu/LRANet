import sys
sys.path.insert(0, 'mmocr/core/evaluation/evaluation_e2e')
from .evaluation_e2e.text_evaluation import TextEvaluator
from mmcv.utils import print_log

import pickle
import numpy as np
import json
from mmdet.datasets.coco import COCO

def eval_hmean_e2e(eval_dataset, results, coco_annos, logger=None):
    # coco_annos = COCO('../../data/totaltext/instances_test.json')
    img_ids = coco_annos.get_img_ids()
    output = []
    for i, img_id in enumerate(img_ids):
        img_name = coco_annos.load_imgs([img_id])[0]['file_name']
        # idx = str(img_id).zfill(7)
        res = results[i]
        ins = res['boundary_result']
        if 'strs' in res.keys():
            recs = res['strs']
        else:
            recs = [""] * len(ins)
        for j in range(len(ins)):
            pts = np.array(ins[j][:-1]).reshape(-1,2).tolist()
            score = ins[j][-1]
            rec = recs[j]
            out = {
                "image_id": int(img_name[5:-4]) if eval_dataset == 'ctw1500' else i,
                # "image_id": i,
                "category_id": 1,
                "polys": pts,
                "rec": rec,
                "score": score
            }
            output.append(out)

    # if eval_dataset == 'ctw1500':
    #     with open('ctw1500_res/my_text_results.json', 'w') as f:
    #         json.dump(output, f)
    # eval_dataset = 'ctw1500'
    # eval_dataset = 'totaltext'
    if eval_dataset == 'ctw1500':
        with open('mmocr/core/evaluation/evaluation_e2e/ctw1500_res/my_text_results.json', 'w') as f:
            json.dump(output, f)
        dataset_name = ['ctw1500']
        outdir = 'mmocr/core/evaluation/evaluation_e2e/ctw1500_res'
    elif eval_dataset == 'totaltext':
        with open('mmocr/core/evaluation/evaluation_e2e/totaltext_res/my_text_results.json', 'w') as f:
            json.dump(output, f)
        dataset_name = ['totaltext']
        outdir = 'mmocr/core/evaluation/evaluation_e2e/totaltext_res'
    elif eval_dataset == 'custom':
        dataset_name = ['custom']
        outdir = 'custom_res'
    cfg = {}
    best_det_hmean = -1
    best_e2e_hmean = -1
    dets = []
    e2es = []

    for t in [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.23, 0.24, 0.26, 0.28, 0.3, 0.35, 0.4, 0.5]:
        print("---------------Score Thr: {}---------------".format(t))
        cfg['INFERENCE_TH_TEST'] = t # tune this parameter to achieve best result
        e = TextEvaluator(dataset_name, cfg, False, output_dir= outdir)
        res = e.evaluate()
        dets.append(res['DETECTION_ONLY_RESULTS'])
        e2es.append(res['E2E_RESULTS'])
        if res['DETECTION_ONLY_RESULTS']['hmean'] > best_det_hmean:
            best_det = res['DETECTION_ONLY_RESULTS']
            best_det_hmean = best_det['hmean']
        if res['E2E_RESULTS']['hmean'] > best_e2e_hmean:
            best_e2e = res['E2E_RESULTS']
            best_e2e_hmean = best_e2e['hmean']
        # if logger is not None:
        # print_log(', '.join(' : '.join([a, str(res[a])]) for a in res), logger)
        # print(res)

    # print("Detection results: {}" .format(best_det))
    # print("End-to-End results: {}".format(best_e2e))

    # if logger is not None:
    print("---------------Final Results---------------")
    print_log("Detection results:   " + ', '.join(' : '.join([a, str(best_det[a])]) for a in best_det), logger)
    print_log("E2E results:   " + ', '.join(' : '.join([a, str(best_e2e[a])]) for a in best_e2e), logger)

    return {'e2e-hmean':best_e2e['hmean'], 'det-hmean':best_det['hmean']}
#
# if __name__ == '__main__':
#     res = sys.argv[1]
#     main(res)