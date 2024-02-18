import json


'''
Step one: prepare your own detection result .json file.
    The resuls can be produced by abcnet. (the detection results should be clockwise)
    Let's create an abcnet-like result file. 
'''
data = {}
res = []
res.append({
    'image_id': '0000001',
    'category_id': 1,
    'polys': [[10,20], [20, 20], [20, 40], [10,40]], 
    'rec': "请你",
    "score": 0.996
})
res.append({
    'image_id': '0000001',
    'category_id': 1,
    'polys': [[100,200], [200, 200], [200, 400], [100,400]], 
    'rec': "happy",
    "score": 0.996
})
res.append({
    'image_id': '0000002',
    'category_id': 1,
    'polys': [[10,20], [20, 20], [20, 40], [10,40]], 
    'rec': "everyday",
    "score": 0.996
})
import os
if not os.path.isdir("custom_res"):
    os.mkdir("custom_res")
with open('custom_res/text_results.json', 'w') as outfile:
    json.dump(res, outfile)




'''
Step two: prepare your custom gt file for evaluation.
    Let's create the gt files. 
'''
os.mkdir("gt_custom_res")
f1 = open("gt_custom_res/0000001.txt", 'w') # corresponding to image_id
f1.writelines("10,20,20,20,20,40,10,40,####请你\n")
f1.writelines("100,200,200,200,200,400,100,400,####enjoy\n")
f1.close()
f2 = open("gt_custom_res/0000002.txt", 'w') # corresponding to image_id
f2.writelines("10,20,20,20,20,40,10,40,####everyday\n")
f2.close()




'''
Step three: make sure gt results are anti-clockwised
'''
import glob
from shapely.geometry import *

files = glob.glob("gt_custom_res/*.txt")
files.sort()
os.mkdir("sorted_gt_custom_res")

for i in files:
    out = i.replace("gt_custom_res", "sorted_gt_custom_res")
    fin = open(i, 'r').readlines()
    fout = open(out, 'w')
    for iline, line in enumerate(fin):
        ptr = line.strip().split(',####')
        rec  = ptr[1]
        cors = ptr[0].split(',')
        assert(len(cors) %2 == 0), 'cors invalid.'
        pts = [(int(cors[j]), int(cors[j+1])) for j in range(0,len(cors),2)]
        try:
            pgt = Polygon(pts)
        except Exception as e:
            print(e)
            print('An invalid detection in {} line {} is removed ... '.format(i, iline))
            continue
        
        if not pgt.is_valid:
            print('An invalid detection in {} line {} is removed ... '.format(i, iline))
            continue
            
        pRing = LinearRing(pts)
        if pRing.is_ccw:
            pts.reverse()
        outstr = ''
        for ipt in pts[:-1]:
            outstr += (str(int(ipt[0]))+','+ str(int(ipt[1]))+',')
        outstr += (str(int(pts[-1][0]))+','+ str(int(pts[-1][1])))
        outstr = outstr+',####' + rec
        fout.writelines(outstr+'\n')
    fout.close()




'''
Step four: save gt to zip file
'''
import zipfile
os.chdir("sorted_gt_custom_res")
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

zipf = zipfile.ZipFile('../datasets/evaluation/gt_custom.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('./', zipf)
zipf.close()
os.chdir("../")
# clean temp files
import shutil
shutil.rmtree("gt_custom_res")
shutil.rmtree("sorted_gt_custom_res")




'''
Step five: eval_dataset to 'custom' in main.py and run.
    Example results:
        "E2E_RESULTS: precision: 0.6666666666666666, recall: 0.6666666666666666, hmean: 0.6666666666666666"
        "DETECTION_ONLY_RESULTS: precision: 1.0, recall: 1.0, hmean: 1.0"

    E2E is less than Det result because 'happy' is not matched to 'enjoy'.
'''
