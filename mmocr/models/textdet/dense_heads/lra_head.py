import torch.nn as nn
import torch
from mmcv.cnn import normal_init, ConvModule
from mmcv.runner import BaseModule
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmocr.models.textdet.postprocess import lra_decode
from mmocr.models.textdet.dense_heads.head_mixin import HeadMixin
from ..postprocess.lra_decoder import  poly_nms
import math
import numpy as np

@HEADS.register_module()
class LRAHead(HeadMixin, BaseModule):

    def __init__(self,
                 in_channels,
                 scales,
                 num_coefficients,
                 path_lra,
                 loss=dict(type='LRALoss'),
                 score_thr=0.1,
                 nms_thr=0.1,
                 num_convs=0,
                 box_iou=False,
                 train_cfg=None,
                 test_cfg=None):

        super().__init__()
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        loss['steps'] = scales
        self.loss_module = build_loss(loss)
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_convs = num_convs
        self.out_channels_reg = num_coefficients
        self.box_iou = box_iou
        U_t = np.load(path_lra)['components_c']
        U_t = torch.from_numpy(U_t)
        self.U_t = U_t

        if self.num_convs > 0:
            cls_convs = []
            reg_convs = []
            conv_cfg = None
            norm = None
            for i in range(self.num_convs):

                cls_convs.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1,
                                            conv_cfg=conv_cfg if i < 3 else None, norm_cfg=norm, act_cfg=dict(type='ReLU')))
                reg_convs.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1,
                                            conv_cfg=conv_cfg if i < 3 else None, norm_cfg=norm, act_cfg=dict(type='ReLU')))
            self.cls_convs = nn.Sequential(*cls_convs)
            self.reg_convs = nn.Sequential(*reg_convs)

        self.out_conv_cls_dense = nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1)
        self.out_conv_reg_dense = nn.Conv2d(
            self.in_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1)

        self.out_conv_cls_sparse = nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1)

        self.out_conv_reg_sparse = nn.Conv2d(
            self.in_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.out_conv_cls_sparse.bias, bias_value)
        self.init_weights()

    def init_weights(self):
        normal_init(self.out_conv_cls_dense, mean=0, std=0.01)
        normal_init(self.out_conv_reg_dense, mean=0, std=0.01)
        normal_init(self.out_conv_reg_sparse, mean=0, std=0.01)

    def forward(self, feats):
        cls_dense, reg_dense, cls_sparse, reg_sparse = multi_apply(self.forward_single, feats)
        level_num = len(cls_dense)
        preds = [[cls_dense[i], reg_dense[i], cls_sparse[i], reg_sparse[i]] for i in range(level_num)]
        return preds

    def forward_single(self, x):
        if self.num_convs > 0:
            x_cls = self.cls_convs(x)
            x_reg = self.reg_convs(x)
        else:
            x_cls = x
            x_reg = x
        cls_predict_dense = self.out_conv_cls_dense(x_cls)
        reg_predict_dense = self.out_conv_reg_dense(x_reg)
        cls_predict_sparse = self.out_conv_cls_sparse(x_cls)
        reg_predict_sparse = self.out_conv_reg_sparse(x_reg)

        return cls_predict_dense, reg_predict_dense, cls_predict_sparse, reg_predict_sparse


    def get_boundary(self, score_maps, img_metas, rescale):

        assert len(score_maps) == len(self.scales)

        boundaries = []

        for idx, score_map in enumerate(score_maps):

            scale = self.scales[idx]
            boundary = self._get_boundary_single(self.U_t.cuda(), score_map, scale)
            boundaries = boundaries + boundary

        boundaries, _ = poly_nms(boundaries, self.nms_thr, with_index=True)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries, scales=self.scales)
        return results

    def _get_boundary_single(self, U_t, score_map, scale):

        return lra_decode(
            U_t = U_t, 
            preds=score_map,
            scale=scale,
            score_thr=self.score_thr,
        )
