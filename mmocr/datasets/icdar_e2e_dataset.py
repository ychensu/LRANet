from mmocr.datasets.icdar_dataset import IcdarDataset
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation import eval_hmean_e2e
import mmocr.utils as utils


@DATASETS.register_module()
class IcdarE2EDataset(IcdarDataset):

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        if not self.test_mode:
            results['texts'] = results['ann_info']['texts']
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, masks_ignore, seg_map. "masks"  and
                "masks_ignore" are represented by polygon boundary
                point sequences.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ignore = []
        gt_masks_ann = []

        gt_texts_ignore = []
        gt_texts_ann = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            trans = ann.get('transcription', "")
            if len(ann.get('segmentation', None)[0]) < 8:
                ann['segmentation'] = [[bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]]
            if ann.get('iscrowd', False) or trans == '###':
                gt_bboxes_ignore.append(bbox)
                gt_masks_ignore.append(ann.get(
                    'segmentation', None))  # to float32 for latter processing
                gt_texts_ignore.append(ann.get(
                    'transcription', None
                ))
            #TODO Ignore according to transcription
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                trans = ann.get('transcription', None)
                if isinstance(trans, list)  > 1:
                    trans = ''.join(trans)
                gt_texts_ann.append(trans)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        for i in range(len(gt_masks_ann)):
            gt_masks_ann[i][0][::2] = np.clip(gt_masks_ann[i][0][::2], 0, img_info['width'])
            gt_masks_ann[i][0][1::2] = np.clip(gt_masks_ann[i][0][1::2], 0, img_info['height'])


        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks_ignore=gt_masks_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            texts=gt_texts_ann,
            texts_ignore=gt_texts_ignore
        )

        return ann

    def prepare_test_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results['texts'] = results['ann_info']['texts']
        return self.pipeline(results)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            try:
                if self.get_ann_info(idx)['bboxes'].shape[0] == 0:
                    idx = self._rand_another(idx)
                    continue
                data = self.prepare_train_img(idx)
                assert len(data['gt_texts'].data) != 0, "no texts in the image"
                assert data is not None
                return data
            except Exception as e:
                print(e)
                print('data error')
                idx = self._rand_another(idx)


    def evaluate(self,
                 results,
                 metric='hmean-iou',
                 logger=None,
                 score_thr=0.1,
                 rank_list=None,
                 **kwargs):
        """Evaluate the hmean metric.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            rank_list (str): json file used to save eval result
                of each image after ranking.
        Returns:
            dict[dict[str: float]]: The evaluation results.
        """
        # results = [{'boundary_result': r[0].tolist()} for r in results]
        # for r in results:
        #     bb = r['boundary_result']
        #     boundaries = []
        #     for b in bb:
        #         boundaries.append([b[0],b[1],b[2],b[1],b[2],b[3],b[0],b[3],b[4]])
        #     r['boundary_result'] = boundaries
        assert utils.is_type_list(results, dict)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['hmean-e2e']
        metrics = set(metrics) & set(allowed_metrics)

        img_infos = []
        ann_infos = []
        for i in range(len(self)):
            img_info = {'filename': self.data_infos[i]['file_name']}
            img_infos.append(img_info)
            ann_infos.append(self.get_ann_info(i))
        if 'totaltext' in self.ann_file:
            dataset_name = 'totaltext'
        if 'ctw' in self.ann_file:
            dataset_name = 'ctw1500'
        eval_results = eval_hmean_e2e(
            dataset_name,
            results,
            self.coco,
            # img_infos,
            # ann_infos,
            # metrics=metrics,
            # score_thr=score_thr,
            logger=logger,
            # rank_list=rank_list
        )

        return eval_results
