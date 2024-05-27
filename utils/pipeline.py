import os
import sys
import json
import torch
import logging
import warnings
import cv2 as cv
import numpy as np
from typing import Optional
from fuzzywuzzy import fuzz
from ultralytics import YOLO
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from dataclasses import dataclass

sys.path.append(os.getcwd())
from utils.Models.charnet.charnet.modeling.model import CharNet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    yolo_path: str
    charnet_path: str
    img_dir: Optional[str] = None
    results_save_path: Optional[str] = None

class ModelPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        try:
            self.segmodel = YOLO(self.config.yolo_path)
            logger.info("YOLO trained weights loaded successfully")
        except FileNotFoundError:
            logger.error("FileNotFoundError: YOLO weights not found at {}".format(self.config.yolo_path))
            raise FileNotFoundError("Invalid file path: {}".format(self.config.yolo_path))
        try:
            self.textmodel = CharNet()
            self.textmodel.load_state_dict(torch.load(self.config.charnet_path))
            self.textmodel.eval()
            logger.info("CharNet weights loaded successfully")
        except FileNotFoundError:
            logger.error("FileNotFoundError: CharNet weights not found at {}".format(self.config.charnet_path))
            raise FileNotFoundError("Invalid file path: {}".format(self.config.charnet_path))
        
        if self.config.img_dir is not None:
            self.img_dir = os.listdir(self.config.img_dir)
            self.img_dir = [os.path.join(self.config.img_dir, file) for file in self.img_dir]
            self.results = dict()
            if self.config.results_save_path is None:
                warnings.warn("Path to save results not provided, won't be able to save analysis results", UserWarning)

    def process_dir(self):
        if self.config.img_dir is not None:
            for img_path in self.img_dir:
                self.process_single_image(img_path, img_path.split('\\')[-1])
            if self.config.results_save_path is None:
                logger.warning("Path to save results is not given, can't save results")
            else:
                with open(self.config.results_save_path, 'w') as file:
                    json.dump(self.results, file, indent=4)
        else:
            logger.warning("Image directory not provided in config, cant perform this operation")

    def process_single_image(self, img: np.ndarray | str, key: Optional[str] = None, visualize: bool = False):
        if isinstance(img, str):
            logger.warning("Image path provided instead of image, trying to load image")
            img = cv.cvtColor(cv.imread(img), cv.COLOR_BGR2RGB)
            if img is not None:
                logger.info("Image loaded successfully")
            else:
                logger.error("FileNotFoundError: Invalid image path {}".format(img))
                raise FileNotFoundError("Invalid image path: {}".format(img))
        results = self.segmodel(img, save=False)
        logger.info("{} books detected".format(len(results[0].boxes.conf)))
        book_names = []
        for mask in results[0].masks.data:
            _, book = self.__single_pass(img, mask, visualize)
            logger.info("Detected book name: {}".format(book))
            book_names.append(book)

        books_count = self.count_books(book_names)                  #// TODO: Replace this with counter that uses string matching score
        if key is not None:
            self.results[key] = books_count
            logger.info("{} processed".format(key))
        else:
            print(books_count)

    def __single_pass(self, img, mask, visualize: bool = False):
        mask = mask.cpu().numpy()
        mask *= 255
        mask = mask.astype(np.uint8)
        mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_LANCZOS4)
        masked_img = cv.bitwise_and(img, img, mask=mask)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)
        masked_img = masked_img[y:y+h, x:x+w]
        logger.debug("Detected book cropped out")
        if masked_img.shape[0] > masked_img.shape[1]:
            masked_img = cv.rotate(masked_img, cv.ROTATE_90_COUNTERCLOCKWISE)
            logger.debug("Rotating upright image")
        _, _, word_instances = self.textmodel(**self.preprocess_img(masked_img))
        name = ''
        for word in word_instances:
            name+= str(word.text)+' '
        if visualize:
            self.visualize_boxes(masked_img, word_instances)
        return word_instances, name.strip()

    @staticmethod
    def preprocess_img(im: np.ndarray | str, size: int = 2280):
        if isinstance(im, str):
            im = cv.imread(im)
        
        h, w, _ = im.shape
        scale = max(h, w) / float(size)
        image_resize_height = int(round(h / scale / 128) * 128)
        image_resize_width = int(round(w / scale / 128) * 128)
        scale_h = float(h) / image_resize_height
        scale_w = float(w) / image_resize_width
        im = cv.resize(im, (image_resize_width, image_resize_height), interpolation=cv.INTER_LINEAR)
        
        return {
            'im': im,
            'im_scale_w': scale_w,
            'im_scale_h': scale_h,
            'original_im_w': w,
            'original_im_h': h
        }
    
    @staticmethod
    def visualize_boxes(img, word_instances):
        img_word_ins = img.copy()
        for word_ins in word_instances:
            word_bbox = word_ins.word_bbox
            cv.polylines(img_word_ins, [word_bbox[:8].reshape((-1, 2)).astype(np.int32)], True, (255, 0, 0), 4)
            cv.putText(
                img_word_ins,
                '{}'.format(word_ins.text),
                (int(word_bbox[0]), int(word_bbox[1])), cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5
            )
        plt.imshow(img_word_ins)
        plt.show()

    @staticmethod
    def count_books(book_names, threshold: int = 75):
        similar_counts = defaultdict(int)
        grouped_strings = []

        for string in book_names:
            found_group = False
            for key in grouped_strings:
                if fuzz.token_set_ratio(string, key) >= threshold:
                    similar_counts[key] += 1
                    found_group = True
                    break
            if not found_group:
                similar_counts[string] += 1
                grouped_strings.append(string)
        
        return dict(similar_counts)