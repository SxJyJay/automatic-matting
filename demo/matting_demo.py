# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2 as cv
import os
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np
import scipy.sparse
import time
import skimage.io
import time
# import visualize

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

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
# testdata_path="./test_data"
# test_images = [f for f in os.listdir(testdata_path) if
#                os.path.isfile(os.path.join(testdata_path, f)) and f.endswith('.jpg')]
# samples = sorted(test_images)
# print(samples)
# for i in range(len(samples)):
#     filename=samples[i]
#     print('Start processing image: {}'.format(filename))
#     img=cv2.imread(os.path.join(testdata_path,filename))
#     composite = coco_demo.run_on_opencv_image(img)
#     print(img.shape)
#     print(composite.shape)
#     cv2.imwrite("./output_mask1/photo_{}.jpg".format(i), composite)

def generate_trimap(alpha):
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    fg = cv.erode(fg, kernel, iterations=np.random.randint(30, 40))
    # fg = cv.erode(fg, kernel, iterations=1)
    # fg = cv.erode(fg, kernel, iterations=np.random.randint(60, 70))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(20, 30))
    # unknown = cv.dilate(unknown, kernel, iterations=1)
    # unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(60, 70))
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)

# def generate_trimap(msk):
#     msk = msk / 255
#     dilated = cv.dilate(msk, kernel, iterations=np.random.randint(20, 30)) * 255
#     eroded = cv.erode(msk, kernel, iterations=np.random.randint(30, 40)) * 255
#     res = dilated.copy()
#     res[((dilated == 255) & (eroded == 0))] = 128
#     return res

# from closed_form_matting_solve_foreground_background import solve_foreground_background
import closed_form_matting

# Load a random image from the images folder
IMAGE_DIR="./images"
file_names = next(os.walk(IMAGE_DIR))[2]
file_names = sorted(file_names)
for name in file_names:
    t0 = time.time()
    # name = '8_0.jpg'
    image = skimage.io.imread(os.path.join(IMAGE_DIR, name))
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]

    # Run detection
    results = coco_demo.run_on_opencv_image(image)
    # cv.imwrite('./test_data/mask.jpg',results)

    # Visualize results
    # r = results[0]
    # alpha_pre = visualize.display_image_matting(image, r['rois'], r['masks'], r['class_ids'],
    #                                             class_names, r['scores'],
    #                                             image_path='./images/pred' + name[:-4] + '.png')

    # scipy.misc.imsave('./images_pred/' + name[:-4] +'trimap.png', alpha_pre)
    t1 = time.time()

    scribble = image.copy()
    # results = np.tile(results[:, :, np.newaxis], (1, 1, 3))
    # print(results.shape, alpha_pre.shape)
    # print(alpha_pre[-30:, -40:])
    trimap = generate_trimap(results)
    # cv.imwrite('./test_data/trimap.jpg', trimap)
    scipy.misc.imsave('./images_pred/' + name[:-4] +'trimap.png', trimap)
    scribble[trimap == 255] = 255
    scribble[trimap == 0] = 0
    # img = np.multiply(img, trimap)
    scipy.misc.imsave('./images_pred/' + name[:-4] + 'scribble.png', scribble)

    alpha = closed_form_matting.closed_form_matting_with_scribbles(image / 255.0, scribble / 255.0)
    scipy.misc.imsave('./images_pred/' + name[:-4] + 'alpha.png', alpha)
    t2 = time.time()

    # foreground, _ = solve_foreground_background(image/255.0, alpha)
    # output = np.concatenate((foreground, alpha[:, :, np.newaxis]), axis=2)
    # or
    image = np.multiply(image, alpha[:, :, np.newaxis])
    output = image

    # plt.imshow(output)
    # plt.show()
    scipy.misc.imsave('./images_pred/' + name[:-4] + 'fg.png', output)
    # print(alpha.max(), alpha.mean())

    # plt.imshow(alpha)
    # plt.show()
    t3 = time.time()
    print(name, 'time spent to calc mask, closed_alpha, fg', t1 - t0, t2 - t1, t3 - t2)
    break

