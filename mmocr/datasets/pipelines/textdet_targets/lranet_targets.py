import Polygon
import cv2
import numpy as np
from numpy.linalg import norm
import mmocr.utils.check_argument as check_argument
from .textsnake_targets import TextSnakeTargets
from scipy.interpolate import splprep, splev
PI = 3.1415926
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LRATargets(TextSnakeTargets):

    def __init__(self,
                 path_lra,
                 num_coefficients=14,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3,
                 level_size_divisors=(8, 16, 32),
                 level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0)),
                 num_samples = 3,
                 with_area=True,
                 ):

        super().__init__()
        assert isinstance(level_size_divisors, tuple)
        assert isinstance(level_proportion_range, tuple)
        assert len(level_size_divisors) == len(level_proportion_range)
        self.with_area = with_area
        self.num_samples = num_samples
        self.num_coefficients = num_coefficients
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.level_size_divisors = level_size_divisors
        self.level_proportion_range = level_proportion_range
        U_t = np.load(path_lra)['components_c']
        self.U_t = U_t

    def generate_center_region_mask(self, img_size, text_polys):
        """Generate text center region mask.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)

        center_region_boxes = []
        for poly in text_polys:
            assert len(poly) == 1
            polygon_points = poly[0].reshape(-1, 2)
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            # resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                    head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                    head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            for i in range(0, len(center_line) - 1):
                tl = center_line[i] + (resampled_top_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                tr = center_line[i + 1] + (
                    resampled_top_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                br = center_line[i + 1] + (
                    resampled_bot_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                bl = center_line[i] + (resampled_bot_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                current_center_box = np.vstack([tl, tr, br,
                                                bl]).astype(np.int32)
                center_region_boxes.append(current_center_box)

        cv2.fillPoly(center_region_mask, center_region_boxes, 1)
        return center_region_mask

    def resample_polygon(self, top_line,bot_line, n=None):
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[float]): The input polygon.
            n (int): The number of resampled points.
        Returns:
            resampled_polygon (list[float]): The resampled polygon.
        """
        if n is None:
            n = self.num_coefficients // 2
        resample_line = []
        for polygon in [top_line, bot_line]:
            if polygon.shape[0] >= 3:
                x,y = polygon[:,0], polygon[:,1]
                tck, u = splprep([x, y], k=3 if polygon.shape[0] >=5 else 2, s=0)
                u = np.linspace(0, 1, num=n, endpoint=True)
                out = splev(u, tck)
                new_polygon = np.stack(out, axis=1).astype('float32')
            else:
                new_polygon = self.resample_line(polygon, n-1)
            resample_line.append(np.array(new_polygon))

        return resample_line # top line, bot line

    def clockwise(self, head_edge, tail_edge, top_sideline, bot_sideline):
        hc = head_edge.mean(axis=0)
        tc = tail_edge.mean(axis=0)
        d = (((hc - tc) ** 2).sum()) ** 0.5 + 0.1
        dx = np.abs(hc[0] - tc[0])
        if not dx / d <= 1:
            print(dx / d)
        angle = np.arccos(dx / d)
        direction = 0 if angle <= PI / 4 else 1  # 0 horizontal, 1 vertical
        if top_sideline[0, direction] > top_sideline[-1, direction]:
            top_indx = np.arange(top_sideline.shape[0] - 1, -1, -1)
        else:
            top_indx = np.arange(0, top_sideline.shape[0])
        top_sideline = top_sideline[top_indx]
        if direction == 1 and top_sideline[0, direction] < top_sideline[-1, direction]:
            top_indx = np.arange(top_sideline.shape[0] - 1, -1, -1)
            top_sideline = top_sideline[top_indx]

        if bot_sideline[0, direction] > bot_sideline[-1, direction]:
            bot_indx = np.arange(bot_sideline.shape[0] - 1, -1, -1)
        else:
            bot_indx = np.arange(0, bot_sideline.shape[0])
        bot_sideline = bot_sideline[bot_indx]
        if direction == 1 and bot_sideline[0, direction] < bot_sideline[-1, direction]:
            bot_indx = np.arange(bot_sideline.shape[0] - 1, -1, -1)
            bot_sideline = bot_sideline[bot_indx]
        if top_sideline[:, 1 - direction].mean() > bot_sideline[:, 1 - direction].mean():
            top_sideline, bot_sideline = bot_sideline, top_sideline

        return top_sideline, bot_sideline, direction


    def reorder_poly_edge(self, points):
        """Get the respective points composing head edge, tail edge, top
        sideline and bottom sideline.

        Args:
            points (ndarray): The points composing a text polygon.

        Returns:
            head_edge (ndarray): The two points composing the head edge of text
                polygon.
            tail_edge (ndarray): The two points composing the tail edge of text
                polygon.
            top_sideline (ndarray): The points composing top curved sideline of
                text polygon.
            bot_sideline (ndarray): The points composing bottom curved sideline
                of text polygon.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2

        head_edge, tail_edge, top_sideline, bot_sideline = super(LRATargets, self).reorder_poly_edge(points)
        top_sideline, bot_sideline,_ = self.clockwise(head_edge, tail_edge, top_sideline, bot_sideline)

        head_edge = np.stack((top_sideline[0], bot_sideline[0]), 0)
        tail_edge = np.stack((top_sideline[-1], bot_sideline[-1]), 0)
        return head_edge, tail_edge, top_sideline, bot_sideline

    def generate_lra_maps(self, img_size, text_polys,text_polys_idx=None, img=None, level_size=None):

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        coeff_maps = np.zeros((self.num_coefficients, h, w), dtype=np.float32)
        for poly,poly_idx in zip(text_polys, text_polys_idx):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.array(text_instance).reshape((1, -1, 2))
            _, _, top_sideline, bot_sideline = self.reorder_poly_edge(polygon[0])
            cv2.fillPoly(mask, np.round(polygon).astype(np.int32), 1)
            resample_top_line,resample_bot_line = self.resample_polygon(top_sideline,bot_sideline)
            resample_line = np.concatenate([resample_top_line, resample_bot_line[::-1]]).flatten()
            resample_line = np.expand_dims(resample_line, axis=1)
            lra_coeff = np.matmul(self.U_t, resample_line)

            yx = np.argwhere(mask > 0.5)
            y, x = yx[:, 0], yx[:, 1]
            batch_T = np.zeros((h, w, self.num_coefficients // 2, 2))
            batch_T[y,x,:,:] = lra_coeff.reshape(-1,2)
            batch_T = batch_T.reshape(h, w, -1).transpose(2, 0, 1)
            coeff_maps[:, y,x] = batch_T[:,y,x]

        return coeff_maps

    def generate_text_region_mask(self, img_size, text_polys, text_polys_idx):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly, poly_idx in zip(text_polys, text_polys_idx):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(
                np.round(text_instance), dtype=np.int).reshape((1, -1, 2))
            if self.with_area:
                cv2.fillPoly(text_region_mask, polygon, poly_idx)
            else:
                cv2.fillPoly(text_region_mask, polygon, 1)
        return text_region_mask

    def generate_level_targets(self, img_size, text_polys, ignore_polys,img=None):
        """Generate ground truth target on each level.

        Args:
            img_size (list[int]): Shape of input image.
            text_polys (list[list[ndarray]]): A list of ground truth polygons.
            ignore_polys (list[list[ndarray]]): A list of ignored polygons.
        Returns:
            level_maps (list(ndarray)): A list of ground target on each level.
            :param img:
        """
        h, w = img_size
        lv_size_divs = self.level_size_divisors
        lv_proportion_range = self.level_proportion_range
        lv_text_polys = [[] for i in range(len(lv_size_divs))]
        lv_text_polys_idx = [[] for i in range(len(lv_size_divs))]
        lv_ignore_polys = [[] for i in range(len(lv_size_divs))]
        polygons_area = []
        level_maps = []
        for poly_idx, poly in enumerate(text_polys):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            tl_x, tl_y, box_w, box_h = cv2.boundingRect(polygon)

            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_text_polys[ind].append([poly[0] / lv_size_divs[ind]])
                    lv_text_polys_idx[ind].append(poly_idx+1)

            if self.with_area:
                polygon_area = Polygon.Polygon(poly[0].reshape(-1,2)).area()
                polygons_area.append(polygon_area)


        for ignore_poly in ignore_polys:
            assert len(ignore_poly) == 1
            text_instance = [[ignore_poly[0][i], ignore_poly[0][i + 1]]
                             for i in range(0, len(ignore_poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_ignore_polys[ind].append(
                        [ignore_poly[0] / lv_size_divs[ind]])


        for ind, size_divisor in enumerate(lv_size_divs):
            current_level_maps = []
            level_img_size = (h // size_divisor, w // size_divisor)
            text_region = self.generate_text_region_mask(
                level_img_size, lv_text_polys[ind], lv_text_polys_idx[ind])[None]
            current_level_maps.append(text_region)

            center_region = self.generate_center_region_mask(
                    level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(center_region)

            effective_mask = self.generate_effective_mask(
                level_img_size, lv_ignore_polys[ind])[None]
            current_level_maps.append(effective_mask)

            lra_coeff_maps = self.generate_lra_maps(
                level_img_size, lv_text_polys[ind],lv_text_polys_idx[ind])

            current_level_maps.append(lra_coeff_maps)
            level_maps.append(np.concatenate(current_level_maps))

        transformed_polys = []
        for j in range(len(text_polys)):
            polygon = text_polys[j][0].reshape(-1,2)
            _, _, top_sideline, bot_sideline = self.reorder_poly_edge(polygon)
            resample_top_line,resample_bot_line = self.resample_polygon(top_sideline,bot_sideline)
            resample_line = np.concatenate([resample_top_line, resample_bot_line[::-1]]).flatten()
            resample_line = np.expand_dims(resample_line, axis=1)
            lra_coeff = np.matmul(self.U_t, resample_line)
            transformed_poly = np.matmul(self.U_t.transpose(), lra_coeff).flatten()
            transformed_polys.append(transformed_poly.flatten())
        transformed_polys = np.array(transformed_polys)

        if transformed_polys.shape[0] > 0:
            transformed_polys = np.concatenate([transformed_polys] * self.num_samples, axis=0)
            
        if self.with_area and len(polygons_area) > 0:
            polygons_area = np.array(polygons_area)
        else:
            polygons_area = np.array([])

        return level_maps, polygons_area, transformed_polys

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks
        gt_texts = results['texts']
        h, w, _ = results['img_shape']

        level_maps, polygons_area, transformed_polys = self.generate_level_targets((h, w), polygon_masks,
                                                 polygon_masks_ignore, results['img'])

        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        mapping = {
            'p3_maps': level_maps[0],
            'p4_maps': level_maps[1],
            'p5_maps': level_maps[2],
            'polygons_area': polygons_area,
            'gt_texts': DC(gt_texts, cpu_only=True),
            'lra_polys': transformed_polys
        }
        
        for key, value in mapping.items():
            results[key] = value

        return results
