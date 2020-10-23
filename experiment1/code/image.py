import cv2
from config import *

class Image:

    def __init__(self, path: Path):
        self.path = path
        self.filename = str(path)
        self.picname = self.filename.split("\\")[-1]
        self.label = str(self.path.parent).split("\\")[-1]
        self.keypoints, self.descriptors = self._extract_keypoints_and_descriptors()
        self.descriptors_size = len(self.descriptors)
        self.histogram = None
        self.tf_idf_weighted_histogram = None

    def __str__(self):
        return str(self.filename)

    #TODO: use my sift function.
    def _extract_keypoints_and_descriptors(self):
        sift = cv2.SIFT_create()

        # log.debug("The file :" +  str(self.filename.exists()))
        if not self.path.exists():
            log.debug(msg=f"The {self.filename} isn't exist.")


        img = cv2.imread(self.filename)
        # cv2.imshow("hello", img)
        # cv2.waitKey(0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('', img)
        # 默认无mask
        return sift.detectAndCompute(img, None) #TODO bug return keypoints and diescriptors



    def set_histogram(self, histogram):
        self.histogram = histogram

    def set_tf_idf_weighted_histogram(self, tf_idf_weighted_histogram):
        self.tf_idf_weighted_histogram = tf_idf_weighted_histogram