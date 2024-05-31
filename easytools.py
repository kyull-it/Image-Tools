import cv2
import matplotlib.pyplot as plt
from itertools import chain
import numpy as np
from PIL import Image, JpegImagePlugin
import math
import glob
import os

from difflib import SequenceMatcher
# https://docs.python.org/ko/3/library/difflib.html#sequencematcher-examples
from colorama import Fore, Back, Style


def display(img, color='rgb', original=False):
    if not original:

        dpi = 100
        height, width = img.shape[:2]

        fig_size = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=fig_size)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        if color == 'rgb':
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray')

        plt.show()

    else:
        plt.axis('off')
        if color == 'rgb':
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.show()


def easy_resize(image):
    height, width = image.shape[:2]
    if height > 1000 or width > 1000:
        rw = width / max(height, width)
        rh = height / max(height, width)
        re_width = int(1000 * rw)
        re_height = int(1000 * rh)
        resized_image = cv2.resize(image, (re_width, re_height), cv2.INTER_CUBIC)
        return resized_image
    elif height < 100 or width < 100:
        rw = width / max(height, width)
        rh = height / max(height, width)
        re_width = int(140 * rw)
        re_height = int(140 * rh)
        resized_image = cv2.resize(image, (re_width, re_height), cv2.INTER_CUBIC)
        return resized_image
    else:
        return image


def reformat_input(image):
    try:

        if type(image) == str:
            img = cv2.imread(image)
            img = easy_resize(img)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        elif type(image) == bytes:
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = easy_resize(img)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif type(image) == np.ndarray:
            if len(image.shape) == 2:  # grayscale
                image = easy_resize(image)
                img_cv_grey = image
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                img_cv_grey = np.squeeze(image)
                img_cv_grey = easy_resize(img_cv_grey)
                img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # BGRscale
                img = image
                img = easy_resize(img)
                img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
                img = image[:, :, :3]
                img = easy_resize(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif type(image) == JpegImagePlugin.JpegImageFile:
            image_array = np.array(image)
            image_array = easy_resize(image_array)
            img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img, img_cv_grey

    except:
        raise ValueError('Invalid input type. Supporting format = string(file path or url), bytes, numpy array')


def adjustment_boxes(image, boxes, padding=7):
    height, width = image.shape[:2]

    x1, y1 = boxes[0]
    x3, y3 = boxes[2]

    x1 = max(int(x1 - padding + 5), 0)
    x3 = min(int(x3 + padding - 5), width)
    y1 = max(int(y1 - padding), 0)
    y3 = min(int(y3 + padding), height)

    boxes[0] = [x1, y1]
    boxes[1] = [x3, y1]
    boxes[2] = [x3, y3]
    boxes[3] = [x1, y3]

    return boxes


def get_perspective_transform(boxes, image):
    point_1 = [[int(boxes[0][0]), int(boxes[0][1])]]
    point_2 = [[int(boxes[1][0]), int(boxes[1][1])]]
    point_3 = [[int(boxes[2][0]), int(boxes[2][1])]]
    point_4 = [[int(boxes[3][0]), int(boxes[3][1])]]

    cnt = np.array(list(chain(point_1, point_2, point_3, point_4)))
    # print("shape of cnt: {}".format(cnt.shape))  # shape of cnt: (4, 2)
    # print(cnt)

    rect = list(cv2.minAreaRect(cnt))
    # print("rect: {}".format(rect))

    # 검출된 bbox 사각형의 너비와 높이
    width = int(rect[1][0])
    height = int(rect[1][1])

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print("bounding box: {}".format(box))
    # box_img = image.copy()
    # display(cv2.drawContours(box_img, [box], 0, (0, 0, 255), 2))

    # 검출된 bbox 사각형의 좌표값
    src_pts = box.astype("float32")
    # print(f"src_pts :\n {src_pts}")

    # 검출된 bbox 사각형의 거리 벡터
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")
    # print(f"dst_pts :\n {dst_pts}")

    # 원근변환을 위한 행렬 구하기
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # print(f"M :\n {M}")

    # 원근변환 적용한 이미지 리턴받기
    img = image.copy()
    warped = cv2.warpPerspective(img, M, (width, height))
    if warped.shape[0] > warped.shape[1] * 2:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    # display(warped)
    # print('\n')

    return warped


def crop(image, bbox, pt=False):
    height, width = image.shape[:2]

    diff_x = abs(bbox[1][0] - bbox[2][0])
    diff_y = abs(bbox[0][1] - bbox[1][1])

    if diff_x >= 10 or diff_y >= 10:
        pt = True
        cropped_image = get_perspective_transform(bbox, image)
    else:
        start_x = max(math.floor(bbox[0][0]), 0)
        start_y = max(math.floor(bbox[0][1]), 0)
        end_x = min(math.ceil(bbox[2][0]), width)
        end_y = min(math.ceil(bbox[2][1]), height)

        cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image, pt


def invert_white_background(image):
    _, height = image.shape
    h1 = int(height / 4)
    h2 = int(height / 4) * 3

    # check if background color is black or not
    bc = [list(np.transpose(image)[h]) for h in range(h1, h2)]

    t, _ = cv2.threshold(image, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if np.average(bc) < t + 5:
        output = cv2.bitwise_not(image)
    else:
        output = image.copy()

    return output


def check_background(image):
    try:
        height, _ = image.shape[:2]
        h1 = int(height / 10)
        h2 = int(height / 10) * 9

        center = np.average([image[h] for h in range(h1, h2)])
        back = np.average([image[h] for h in range(0, h1)] + [image[h] for h in range(h2, height)])
        # print(center, back)

        if center >= back:
            output = cv2.bitwise_not(image)
        else:
            output = image.copy()

    except Exception as error:
        print(error)
        display(image, 'gray')
        output = invert_white_background(image)

    return output


def check_luminance(path, filter_size=20, k_size=3):
    image, _ = reformat_input(path)
    grad = get_sobel_grad(image, k_size=k_size)
    s = grad.shape
    l = filter_size*filter_size*0.95
    check_lum = False

    for i in range(filter_size, s[0]):
        for j in range(filter_size, s[1]):

            box = grad[i-filter_size:i, j-filter_size:j]

            check1 = np.count_nonzero(box == 0) > l
            if check1:
                pixel = np.average(image[i-(filter_size//2), j-(filter_size//2)])
                if pixel+20 >= 255:
                    check_lum = True
                    break
            else:
                continue
        if check_lum:
            break

    return check_lum


def get_sobel_grad(image, k_size=3):
    lum = image[:, :, 1]
    grad_x = cv2.Sobel(lum, -1, 0, 1, ksize=k_size)
    grad_y = cv2.Sobel(lum, -1, 1, 0, ksize=k_size)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return grad


def name_sort(text):
    return int(text.split('/')[-1].split('_')[-1].split('.')[0])


def get_path_list(rootpath):
    path_list = glob.glob(f'{rootpath}/*')
    path_list = sorted(path_list, key=name_sort)
    name_list = list(map(name_sort, path_list))
    return path_list, name_list


def get_labels(path):
    f = open(path, 'r').read()
    f = f.split()
    word_list = [f[i].replace('"', '') for i in range(4, len(f), 5)]
    chr_list = list(''.join(word_list))
    return word_list, chr_list


def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def save_bbox(save_path, idx, image):
    image_name = save_path.split('/')[-1].split('.')[0]
    folder_name = save_path.split('/')[-2:-1][0]
    save_path = '/'.join(save_path.split('/')[:-2]) + f'/{folder_name}_results/{image_name}'

    create_directory(save_path)
    cv2.imwrite(f'{save_path}/{idx}.jpg', image)


def save_results(result_list, path):
    with open(f'{path}/results.txt', 'w') as save_file:
        save_file.write(str(result_list))
        save_file.close()


def evaluate_chars(gt_list, prediction_list):
    similarity_list = []

    if len(gt_list) != len(prediction_list):
        print("The number of Ground Truth != The number of Predictions")

    else:
        for i in range(len(gt_list)):

            gt = ''.join(gt_list[i])
            prediction = ''.join(prediction_list[i])

            if len(prediction) == 0:
                sim = 0

            else:
                sim = round(SequenceMatcher(None, gt, prediction).ratio(), 3)

            similarity_list.append(sim)

    # 결과 데이터 분석
    print()
    print(Style.BRIGHT + "EDA : Exploratory Data Analysis" + Style.RESET_ALL)
    print("-" * 50)
    print(Style.DIM + 'The number of Image Dataset' + Style.RESET_ALL)
    print(':', (len(gt_list)))
    print("-" * 50)

    # 평균 정밀도(precision), 재현도(recall), f1 score
    avg_similarity = np.average(similarity_list)
    print(Style.DIM + "Average of similarity" + Style.RESET_ALL)
    print(": ", avg_similarity)
    print()

    return similarity_list
