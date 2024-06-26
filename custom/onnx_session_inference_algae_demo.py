import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def get_image(path, show=False):
    with Image.open(path) as img:
        img = np.array(img.convert("RGB"))
    if show:
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        plt.close()
    return img


def preprocess(img):
    img = img / 255.0
    img = image_resize(img, width=None, height=800)
    h, w = img.shape[0], img.shape[1]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


provider = ["CUDAExecutionProvider"]

ort_session = ort.InferenceSession(
    "mmdeploy_models/onnx_algae/model.onnx",
    providers=provider,
)

img = get_image("20210315_pc_19_0m_D1.jpg", show=False)
img = preprocess(img)
ort_inputs = {ort_session.get_inputs()[0].name: img}
preds = ort_session.run(None, ort_inputs)
bbox_and_score = preds[0]
bboxes = bbox_and_score[0][:, 0:4]
scores = bbox_and_score[0][:, -1]
labels = preds[1][0]

filtered_bboxes, filtered_scores, filtered_labels = list(), list(), list()

for i in range(len(labels)):
    if scores[i] > 0.5:
        filtered_bboxes.append(bboxes[i])
        filtered_scores.append(scores[i])
        filtered_labels.append(labels[i])

filtered_bboxes = np.array(filtered_bboxes)
filtered_scores = np.array(filtered_scores)
filtered_labels = np.array(filtered_labels)
