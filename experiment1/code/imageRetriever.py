import torch
import math


# 计算tf-idf化的各个图像histogram, 同时也负责计算 test image的histogram, 以及计算选取the test image 和 train images 距离最小的图像
class ImageRetriever:
    def __init__(self, bag_of_visual_words):
        self.kmeans = bag_of_visual_words.kmeans
        self.images = bag_of_visual_words.images
        self.inverted_file_table = bag_of_visual_words.inverted_file_table
        self.generate_image_histogram = bag_of_visual_words.generate_image_histogram
        self._generate_tf_idf_weighted_histogram()

        self.total_tf_idf_weighted_histogram = self._generate_total_tf_idf_weighted_histogram()

    def _generate_tf_idf_weighted_histogram(self):
        for image in self.images:
            image.set_tf_idf_weighted_histogram(self._tf_idf(image))

    # 用 tf-idf 将histogram of image修正
    def _tf_idf(self, image):
        k = self.kmeans.n_clusters
        tf_idf_weighted_histogram = torch.zeros([1, k])
        visual_words_num = torch.sum(image.histogram)
        # 计算 tf-idf
        for i in range(k):
            tf = image.histogram[:, i] / visual_words_num
            idf = math.log(len(self.images) / (len(self.inverted_file_table[i]) + 1))
            tf_idf_weighted_histogram[:, i] = tf * idf
        return tf_idf_weighted_histogram

    # 将tf-idf histogram cat 起来
    def _generate_total_tf_idf_weighted_histogram(self):
        total_tf_idf_weighted_histogram = None
        for image in self.images:
            if total_tf_idf_weighted_histogram is None:
                total_tf_idf_weighted_histogram = image.tf_idf_weighted_histogram
            else:
                total_tf_idf_weighted_histogram = torch.cat(
                    (total_tf_idf_weighted_histogram, image.tf_idf_weighted_histogram), dim=0)
        return total_tf_idf_weighted_histogram



    def retrieve(self, image):
        labels = self.kmeans.predict(torch.Tensor(image.descriptors))
        self.generate_image_histogram(image, labels)
        image.set_tf_idf_weighted_histogram(self._tf_idf(image))
        m = self.total_tf_idf_weighted_histogram.shape[0]
        l = torch.zeros([m, 1])

        # 取出距离最小的值， 如果用矩阵并行计算更好一点？
        for i in range(m):
            l[i, :] = torch.sum((image.tf_idf_weighted_histogram-self.total_tf_idf_weighted_histogram[i])**2, dim=1)
        min_loss_image_index = torch.argmin(l, dim=0)
        return self.images[min_loss_image_index]
