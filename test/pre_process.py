import cv2

def pre_processing(img):
    if img.shape[0] < 640 or img.shape[1] < 640:
        if img.shape[0] > img.shape[1]:
            cal = img.shape[0]
        else:
            cal = img.shape[1]
        scale_percent = 640 * 100 / cal
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        upsized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    else:
        upsized = img
    gaussian = cv2.GaussianBlur(upsized, (3, 3), 0)
    return gaussian