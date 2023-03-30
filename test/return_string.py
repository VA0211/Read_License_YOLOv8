import cv2
import numpy as np

def sort_value_dict(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    sorted_value_index = np.argsort(values)
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
    return sorted_dict

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3,
                    txt_color, thickness=tf, lineType=cv2.LINE_AA)
        return p1
    
def print_plate(image, boxes, labels=[], colors=[], score=False, conf=0.4):
    if labels == []:
       labels = {0: u'background', 1: u'0', 2: u'1', 3: u'2', 4: u'3', 5: u'4', 6: u'5', 7: u'6', 8: u'7', 9: u'8', 
                 10: u'9', 11: u'A', 12: u'B', 13: u'C', 14: u'D', 15: u'E', 16: u'F', 17: u'G', 18: u'H', 19: u'K', 
                 20: u'L', 21: u'M', 22: u'N', 23: u'P', 24: u'S', 25: u'T', 26: u'U', 27: u'V', 28: u'X', 29: u'Y', 30: u'Z'}
    # Define colors
    if colors == []:
        colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),
                  (115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),
                  (25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),
                  (10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),
                  (251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1)]
    
    position = {}
    line1 = {}
    line2 = {}
    result = []
    for box in boxes:
        #add score in label if score=True
        if score :
            label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
        else:
            label = labels[int(box[-1])+1]

        if box[-2] > conf:
            p1 = box_label(image, box, label)
            position.update({label: p1})

    mean_y = np.mean(list(position.values()))
    for char, pos in position.items():
        if pos[1] < mean_y:
            line1.update({char: pos[0]})
        else:
            line2.update({char: pos[0]})
    line1 = sort_value_dict(line1)
    line2 = sort_value_dict(line2)
    result = str(list(line1.keys())) + str(list(line2.keys()))
    print(result)