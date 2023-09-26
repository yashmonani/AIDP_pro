import cv2
import skimage
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from img_util import imshow

#day 12

#blue colour detection and count 

vid = cv2.VideoCapture(0)
while True:
    ack, im=vid.read()#acquire video camera
    if ack:
        th,red_bw= cv2.threshold(
            cv2.subtract(
                im[:, :, -3], cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                ),50,255, cv2.THRESH_BINARY
                )
        strel =cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
        red_bw = cv2.morphologyEx(
            red_bw, cv2.MORPH_CLOSE, strel, iterations=1
        )
        red_bw = ski.morphology.remove_small_objects(
            red_bw.astype(bool), 300 #remoe pixels with low resolution 
        ).astype('uint8')*255
        rps = ski.measure.regionprops(
            ski.measure.label(red_bw.astype(bool))
        )
        count = len(rps)
        img_copy = im.copy()
        cv2.putText(img_copy, str(count),(150,150),
                    cv2.FONT_HERSHEY_PLAIN, 10 ,(0,0,255),10
                    )
        for rp in rps:
            y1,x1,y2,x2 = rp.bbox
            cv2.rectangle(img_copy, (x1,y1), (x2,y2), (0,0,255), thickness=5)
        cv2.imshow('Preview',img_copy)
        key = cv2.waitKey(1)
        if key == ord('x'): 
            break
cv2.destroyAllWindows()
vid.release()