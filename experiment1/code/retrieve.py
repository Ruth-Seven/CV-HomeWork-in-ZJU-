import torch
from kmeans import KMeans
from BagOfVisualWords import BagOfVisualWords
from imageRetriever import ImageRetriever
from utils import load_images, load_image, imshow
from config import *
import argparse

parser = argparse.ArgumentParser(description='Image retrieve base on bag of visual words')
parser.add_argument('--name', '-n', type=str, help='choose a image from ./dataset/test/* to retrieve')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    kmeans = KMeans(n_clusters=70, device=device)
    # a test image
    image_path =  test_data_path / str(args.name)
    target_image = load_image(image_path)
    # train dataset
    images = load_images(train_data_path)
    image_retriever = ImageRetriever(BagOfVisualWords(images=images, kmeans=kmeans))
    result = image_retriever.retrieve(target_image)
    imshow(result)
