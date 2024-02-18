import numpy as np
import json

CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

def decode(rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < 95:
            s += CTLABELS[c]
        elif c == 95:
            s += u'口'

    return s

def bezier_to_polygon(bezier):
    u = np.linspace(0, 1, 10)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
             + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
             + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
             + np.outer(u ** 3, bezier[:, 3])

    # convert points to polygon
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    points = np.around(points, 2)
    return points.reshape(-1).tolist()


def main():
    abc = json.load(open('data/synthtext-150k/syntext1/annotations/train.json','r'))
    anns = abc['annotations']
    # coco_anns = []
    for i, ann in enumerate(anns):
        transcription = decode(ann['rec'])
        # if len(transcription) <= 2:
            # print(transcription)
        bezier_pts = ann['bezier_pts']
        segmentation = bezier_to_polygon(np.array(bezier_pts))
        anns[i] = {
            "iscrowd": ann['iscrowd'],
            "category_id": 1,
            "bbox": ann['bbox'],
            "area": ann['area'],
            "segmentation": [segmentation],
            "transcription": transcription,
            "image_id": ann['image_id'],
            "id": ann['id']
        }

    abc['annotations'] = anns
    with open('data/synthtext-150k/syntext1/train_polygon.json','w') as f:
        json.dump(abc, f)


main()