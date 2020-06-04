import threading
from typing import Tuple, List

import cv2 as cv
import numpy as np

import utils.rectification as rct
import utils.uimg as uimg
from data import SingleCharData
from ocr_processor import OcrProcessor
from utils.utext import TextPage


class OcrManager(object):
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, charmap_path, aliasmap_path, ckpt_dir, h, w, num_class, batch_size):
        self.main = OcrProcessor().load(w, h, num_class=num_class, ckpt_dir=ckpt_dir)
        self.data = SingleCharData(h, w).load_char_map(charmap_path).load_alias_map(aliasmap_path)
        self.batch_size = batch_size

    def __new__(cls, charmap_path, aliasmap_path, ckpt_dir, h, w, num_class, batch_size):
        if OcrManager._instance is None:
            with OcrManager._instance_lock:
                if OcrManager._instance is None:
                    OcrManager._instance = object.__new__(cls)
        return OcrManager._instance

    def _process(self, page_path: str, p_thresh: float, auxiliary_img: str,
                 box: List[int] = None, remove_lines: bool = False) -> TextPage:
        src = uimg.read(page_path, 1)
        _page = TextPage(src, 0)
        # 1.
        # _page.auto_bin()
        if box is not None:
            print('temp_box: '+str(box))
            x1, y1, x2, y2 = box
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            region_img = _page.img[y1:y2, x1:x2]
        else:
            region_img = _page.img

        # if box == [643, 364, 861, 390]:
        #     uimg.save('static/aaa.jpg', region_img)

        region = TextPage(region_img, 0, drawing_copy=None)
        uimg.save('static/1.jpg', region_img)
        region.auto_bin()

        if np.mean(region.img) < 128:
            region.img = uimg.reverse(region.img)

        if remove_lines:
            region.remove_lines()

        if auxiliary_img is not None and auxiliary_img != '':
            region.drawing_copy = cv.cvtColor(region.img.copy(), cv.COLOR_GRAY2BGR)

            uimg.save('static/2.jpg', region.drawing_copy)

            # 2.
            # fixme 低分辨率旋转校正报错
            # page.fix_orientation()

            # 3. & 4.
            region.split()

        # uimg.save(auxiliary_img[:-4]+'line.jpg', region.get_lines()[0].drawing_copy)
        # 5. & 6.
        self.data.set_images(region.make_input_1(auxiliary_img[-8:-4])).init_indices()
        results = self.main.recognize(infer_data=self.data, batch_size=self.batch_size)
        # 7.
        region.set_result_1(results)
        # p_thresh 预测分数的阈值，过滤掉小于阈值的char,此处为懒删除标记为invalid
        region.filter_by_p(p_thresh=p_thresh)

        for line in region.get_lines(ignore_empty=True):
            print("标准宽度： "+str(line.get_relative_standard_width()))
            line.mark_half()
            # line.calculate_meanline_regression()
            line.merge_components()

        # 8.
        self.data.set_images(region.make_input_2()).init_indices()
        results2 = self.main.recognize(infer_data=self.data, batch_size=self.batch_size)
        region.set_result_2(results2)
        if auxiliary_img is not None:
            uimg.save(auxiliary_img, region.drawing_copy)
        region.mark_char_location()

        rct.rectify_by_location(region.iterate(1))
        rct.rectify_5(region.iterate(5))

        # if auxiliary_img is not None:
        #     uimg.save(auxiliary_img, region.drawing_copy)

        # if auxiliary_html is not None:
        #     with open(auxiliary_html, 'w', encoding='utf-8') as f:
        #         f.write(page.format_html(tplt))

        return region

    def get_json_result(self, page_path: str, p_thresh: float, auxiliary_img: str,
                        box: List[int] = None, remove_lines=False):
        page = self._process(page_path, p_thresh, auxiliary_img, box=box, remove_lines=remove_lines)
        return page.format_json(p_thresh=p_thresh)

    def get_text_result(self, page_path: str, p_thresh: float, auxiliary_img: str,
                        box: List[int] = None, remove_lines=False):
        page = self._process(page_path, p_thresh, auxiliary_img, box=box, remove_lines=remove_lines)
        return page.format_result(p_thresh=p_thresh)

    def get_verbose_result(self, page_path: str, p_thresh: float, auxiliary_img: str,
                           box: List[int] = None, remove_lines=False):
        page = self._process(page_path, p_thresh, auxiliary_img, box=box, remove_lines=remove_lines)
        return page.format_verbose(p_thresh=p_thresh)


if __name__ == '__main__':
    path = "doc_imgs/2015南立刑初字第0001号_枉法裁判罪84页.pdf/img-0228.jpg"

    proc = OcrManager("/usr/local/src/data/stage2/all_4190/all_4190.json",
                     "/usr/local/src/data/stage2/all_4190/aliasmap.json",
                     '/usr/local/src/data/stage2/all_4190/ckpts',
                      64, 64, 4190, 64)
    res = proc.get_text_result(path, 0.9, '/usr/local/src/data/results/auxiliary.png')
    print(res)
