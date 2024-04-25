import cv2
import os
import pytesseract
import time
import json


def ocr_detection_pytesseract(imgpath):
    """
    Performs OCR on the given image using pytesseract.

    Args:
        imgpath (str): Path to the input image file.

    Returns:
        dict: Dictionary containing the detected text annotations and their bounding boxes.
    """
    start = time.time()
    img = cv2.imread(imgpath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)
    print('*** Text Detection Time Taken: %.3fs ***' % (time.time() - start))

    if not results['text']:
        return None
    else:
        return results_to_annotations(results, img.shape)


def results_to_annotations(results, img_shape):
    """
    Converts the OCR results from pytesseract to the expected JSON format.

    Args:
        results (dict): Dictionary containing the OCR results from pytesseract.
        img_shape (tuple): Shape of the input image (height, width).

    Returns:
        dict: Dictionary containing the text annotations and their bounding boxes.
    """
    texts = []
    for i in range(len(results['level'])):
        if results['text'][i].strip():
            text = {
                'id': i,
                'content': results['text'][i],
                'column_min': results['left'][i],
                'row_min': results['top'][i],
                'column_max': results['left'][i] + results['width'][i],
                'row_max': results['top'][i] + results['height'][i],
                'width': results['width'][i],
                'height': results['height'][i]
            }
            texts.append(text)
    return {'img_shape': img_shape[:2], 'texts': texts}


def ocr_detection(imgpath):
    """
    Wrapper function for OCR detection using pytesseract.

    Args:
        imgpath (str): Path to the input image file.

    Returns:
        dict: Dictionary containing the detected text annotations and their bounding boxes.
    """
    return ocr_detection_pytesseract(imgpath)