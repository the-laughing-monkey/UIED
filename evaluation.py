import json
import numpy as np
import cv2

class_map = ['button', 'input', 'select', 'search', 'list', 'img', 'block', 'text', 'icon']


def resize_label(bboxes, d_height, gt_height, bias=10):
    bboxes_new = []
    scale = gt_height/d_height
    for bbox in bboxes:
        bbox = [int(b * scale + bias) for b in bbox]
        bboxes_new.append(bbox)
    return bboxes_new


def draw_bounding_box(org, corners, color=(0, 255, 0), line=2, show=False):
    """
    Draw bounding box of components on the original image
    :param org: original image
    :param corners: [(top_left, bottom_right)]
                    -> top_left: (column_min, row_min)
                    -> bottom_right: (column_max, row_max)
    :param color: line color
    :param line: line thickness
    :param show: show or not
    :return: labeled image
    """
    board = org.copy()
    for i in range(len(corners)):
        board = cv2.rectangle(board, (corners[i][0], corners[i][1]), (corners[i][2], corners[i][3]), color, line)
    if show:
        cv2.imshow('a', cv2.resize(board, (300, 600)))
        cv2.waitKey(0)
    return board


def read_detect_result(file_name):
    '''
    :return: {list of [[col_min, row_min, col_max, row_max]], list of [class id]
    '''

    def is_bottom_or_top(bbox):
        if bbox[1] < 100 and (bbox[1] + bbox[3]) < 100 or\
                bbox[1] > 500 and (bbox[1] + bbox[3]) > 500:
            return True
        return False

    file = open(file_name, 'r')
    bboxes = []
    categories = []
    for l in file.readlines():
        labels = l.split()[1:]
        for label in labels:
            label = label.split(',')
            bbox = [int(b) for b in label[:-1]]
            if not is_bottom_or_top(bbox):
                bboxes.append(bbox)
                categories.append(class_map[int(label[-1])])

    return {file_name.split('_')[0]: {'bboxes':bboxes, 'categories':categories}}


def read_ground_truth():
    def get_img_by_id(img_id):
        for image in images:
            if image['id'] == img_id:
                return image['file_name'].split('/')[-1][:-4], (image['height'], image['width'])

    def cvt_bbox(bbox):
        '''
        :param bbox: [x,y,width,height]
        :return: [col_min, row_min, col_max, row_max]
        '''
        bbox = [int(b) for b in bbox]
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    data = json.load(open('instances_test.json', 'r'))
    images = data['images']
    annots = data['annotations']

    compos = {}
    for annot in annots:
        img_name, size = get_img_by_id(annot['image_id'])
        if img_name not in compos:
            compos[img_name] = {'bboxes': [cvt_bbox(annot['bbox'])], 'categories': [annot['category_id']], 'size':size}
        else:
            compos[img_name]['bboxes'].append(cvt_bbox(annot['bbox']))
            compos[img_name]['categories'].append(annot['category_id'])
    return compos


def eval(detection, ground_truth):

    def match(org, d_bbox, gt_bboxes, matched):
        '''
        :param matched: mark if the ground truth component is matched
        :param d_bbox: [col_min, row_min, col_max, row_max]
        :param gt_bboxes: list of ground truth [[col_min, row_min, col_max, row_max]]
        :return: Boolean: if IOU large enough or detected box is contained by ground truth
        '''
        area_d = (d_bbox[2] - d_bbox[0]) * (d_bbox[3] - d_bbox[1])
        broad = draw_bounding_box(org, [d_bbox], color=(0, 0, 255), show=True)
        for i, gt_bbox in enumerate(gt_bboxes):
            if matched[i] == 0:
                continue
            area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[0])
            col_min = max(d_bbox[0], gt_bbox[0])
            row_min = max(d_bbox[1], gt_bbox[1])
            col_max = min(d_bbox[2], gt_bbox[2])
            row_max = min(d_bbox[3], gt_bbox[3])
            # if not intersected, area intersection should be 0
            w = max(0, col_max - col_min)
            h = max(0, row_max - row_min)
            area_inter = w * h

            iod = area_inter / area_d
            iou = area_inter / (area_d + area_gt - area_inter)

            print(iod, iou)
            draw_bounding_box(broad, [gt_bbox], color=(0, 255, 0), show=True)

            if area_inter == 0:
                continue
            # the interaction is d itself, so d is contained in gt, considered as correct detection
            if iod == area_d and area_d / area_gt > 0.1:
                matched[i] = 0
                return True
            if iou > 0.5:
                return True
        return False

    TP, FP, FN = 0, 0, 0
    for image in detection:
        d_compos = detection[image]
        gt_compos = ground_truth[image]
        d_compos['bboxes'] = resize_label(d_compos['bboxes'], 600, gt_compos['size'][0])

        matched = np.ones(len(gt_compos['bboxes']), dtype=int)
        img = cv2.imread(image + '.jpg')
        for d_bbox in d_compos['bboxes']:
            if match(img, d_bbox, gt_compos['bboxes'], matched):
                TP += 1
            else:
                FP += 1
        FN += sum(matched)

    print(TP, FP, FN)
    precesion = TP / (TP+FP)
    recall = TP / (TP+FN)
    print('Precesion:%.3f, Recall:%.3f' % (precesion, recall))


gt = read_ground_truth()
detect = read_detect_result('472_merged.txt')
eval(detect, gt)