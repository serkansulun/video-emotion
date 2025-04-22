# Source: https://github.com/ultralytics/yolov5
from pathlib import Path
import sys
import os
import numpy as np
import cv2
import torch
import utils as u

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from face_utils.general import check_img_size, non_max_suppression_face, scale_coords
from face_utils.datasets import letterbox

from models.experimental import attempt_load

def load_model(weights):
    model = attempt_load(weights, map_location='cpu')  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect(
    model,
    im,
    device,
    # project,
    # name,
    # exist_ok,
    save_img,
    view_img
):
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz=(640, 640)

    coords = []

    h0, w0 = im.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(im, new_shape=imgsz)[0]
    # Convert from w,h,c to c,w,h
    img = img.transpose(2, 0, 1).copy()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = model(img)[0]
    
    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im.shape).round()
            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).int().tolist()
                coords.append(xyxy)
    return coords

                    
class FaceDetector(torch.nn.Module):
    def __init__(self):
        super(FaceDetector, self).__init__()
        script_dir = os.path.dirname(__file__)
        script_dir = Path(script_dir)

        model_path = script_dir / '../../weights/yolov5m-face.pt'
        if not os.path.exists(model_path):
            print('Downloading face detector model weights...', end=' ', flush=True)
            id = '1Sx-KEGXSxvPMS35JhzQKeRBiqC98VDDI'
            u.download_gdrive(id, model_path)
            print('Done.', flush=True)
        self.model = load_model(model_path)

    def to_device(self, device):
        self.model.to(device)

    def get_device(self):
        return next(self.model.parameters()).device.type

    def __call__(self, image):
        coordinates = detect(self.model, image, self.get_device(), save_img=False, view_img=True)
        faces = []
        for coordinate in coordinates:
            cropped = image[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2], :]
            faces.append(cropped)
        return faces, coordinates
    
    def sanity_check(self):
        import requests
        from PIL import Image
        import matplotlib.pyplot as plt

        url = r"https://store-media.mpowerpromo.com/5c7edfd8208c866f6f7c7fd0/pages/610ad84611fa9a2728e13280/Group-Chat-1628101396619.jpg"
        image = np.array(Image.open(requests.get(url, stream=True).raw)).transpose(2, 0, 1)
        faces = self.__call__(image)
        for face in faces:
            plt.figure()
            plt.imshow(face)
        plt.show()

