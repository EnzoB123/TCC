# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import os
import sys
import serial
import argparse
import pandas as pd
from pathlib import Path
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.general import (check_img_size, check_imshow, check_requirements, cv2,
                           increment_path, non_max_suppression, scale_coords)
from utils.dataloaders import LoadStreams
from models.common import DetectMultiBackend

import torch
import torch.backends.cudnn as cudnn
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

if not os.path.exists('results/'):
    os.mkdir('results/')
    os.mkdir('results/data/')
    os.mkdir('results/frames/')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
objetivo = ''


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        weights_dist=ROOT / 'model@1535470106.h5',
        model_dist=ROOT / 'model@1535470106.json',
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
):
    source = str(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    (save_dir / save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device('')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Data loading
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    annotator = Annotator(dataset=dataset, example=str(ROOT / 'data/images/bus.jpg'))

    # Model loading
    json_file = open(model_dist, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_dist)

    # Scaler loading
    scaler = StandardScaler()
    df = pd.read_csv('data.csv')
    X_train = df[['xmin', 'ymin', 'xmax', 'ymax']]
    scaler.fit(X_train)

    # Loop
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if False else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # Process detections
        det = pred[0]
        s = ''
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors[int(cls)])
                # Objeto principal
                if cls == 0:
                    objetivo = 'esquerda'
                else:
                    objetivo = 'direita'

                # Distance prediction
                X = [[xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]]
                X = scaler.transform(X)
                distance_pred = loaded_model.predict(X)

                # Objeto central
                principal = 0

                for x in distance_pred:
                    if x[0] > principal:
                        principal = x[0]

                time = int(principal) // 50

                if time == 0:
                    with serial.Serial('/dev/ttyUSB0', 9600, timeout=1) as ser:
                        ser.write("back3".encode())
                else:
                    if objetivo == "direita":
                        with serial.Serial('/dev/ttyUSB0', 9600, timeout=1) as ser:
                            ser.write("righ{}".format(time).encode())
                    else:
                        with serial.Serial('/dev/ttyUSB0', 9600, timeout=1) as ser:
                            ser.write("left{}".format(time).encode())
        time = 0


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--model_dist', help='model json file path')
    parser.add_argument('--weights_dist', help='model weights file path')
    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand

    return args


def main(options):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(options))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
