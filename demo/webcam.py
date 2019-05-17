# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np

import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )
    """
    cam = cv2.VideoCapture("./test.mp4")
    for i in range(10):
        start_time = time.time()
        ret_val, img = cam.read()
        composite = coco_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        # cv2.imshow("COCO detections", composite)
        cv2.imwrite("./output_mask/photo_{}.jpg".format(i), composite)
        # if cv2.waitKey(1) == 27:
        #     break  # esc to quit
    # cv2.destroyAllWindows()
    """
    testdata_path="./test_data"
    test_images = [f for f in os.listdir(testdata_path) if
                   os.path.isfile(os.path.join(testdata_path, f)) and f.endswith('.jpg')]
    samples = sorted(test_images)
    print(samples)
    for i in range(len(samples)):
        filename=samples[i]
        print('Start processing image: {}'.format(filename))
        img=cv2.imread(os.path.join(testdata_path,filename))
        composite = coco_demo.run_on_opencv_image(img)
        print(img.shape)
        print(composite.shape)
        cv2.imwrite("./output_mask1/photo_{}.jpg".format(i), composite)


if __name__ == "__main__":
    main()
