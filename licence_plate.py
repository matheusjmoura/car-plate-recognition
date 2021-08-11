#! usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import numpy as np 
import cv2 as cv
import imutils
import pytesseract
import os

def auto_canny (image, sigma=0.33):
        v = np.median(image)
        lower = int(max(0, (1 - sigma) * v ))
        upper = int(min(255, (1 + sigma) * v ))
        ed = cv.Canny(image, lower, upper)
        return ed

def recog(car):
        image = cv.imread(car)
        image = imutils.resize(image, width=500)
        # cv.imshow("Img", image)
        
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_gray = cv.bilateralFilter(image_gray, 11, 17, 17)
        edged = auto_canny(image_gray)
        
        # cv.imshow("Edged", edged)
        
        cnts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv.contourArea, reverse=True)[:10]
        screenCnt = None
        
        for c in cnts:
                perimeter = cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, 0.018 * perimeter, True)
                if (len(approx) == 4):
                        screenCnt = approx
                        break

        if not (screenCnt is None):
                mask = np.zeros(image_gray.shape, np.uint8)
                new_img = cv.drawContours(mask, [screenCnt], 0, 255, -1)
                new_image = cv.bitwise_and(image,image, mask=mask)
                
                # cv.imshow("Contours", new_img)

                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                licence_plate = image[topx:bottomx+1, topy:bottomy+1]
                
                licence_plate = cv.cvtColor(licence_plate, cv.COLOR_BGR2GRAY)
                licence_plate = cv.GaussianBlur(licence_plate, (5, 5), 0)
                
                tmp, imgThs = cv.threshold(licence_plate,0,255,cv.THRESH_OTSU+cv.THRESH_BINARY)


                # cv.imshow("plate", imgThs)

                plate_array = []
                car_info = []

                text = pytesseract.image_to_string(imgThs)
                car_info.append(car)
                car_info.append(text)
                plate_array.append(car_info)
                
                k = cv.waitKey(0)
                if (k == 27):
                        cv.destroyAllWindows()
                
                return plate_array
        else:
                return -1


def dump_directory ():
        path = './car_img_repo/new_dataset/'
        files = os.listdir(path)
        licence_plates = []

        for index, file in enumerate(files): 
                string_img = path + file
                print(string_img)
                plate_text = recog(string_img)
                if (plate_text is not -1):
                        licence_plates.append(recog(string_img))
        print("\n\n Output => \n")

        for car in licence_plates:
                print car

def normalize_dir ():
        path = './car_img_repo/new_dataset/'
        files = os.listdir(path)
        for index, file in enumerate(files):
                os.rename(os.path.join(path, file), os.path.join(path, str(index)+'.jpg'))



if(len(sys.argv) > 1):
        if(sys.argv[1] == 'n'):
                normalize_dir()
        else:
                print(recog(sys.argv[1]))
else:
        dump_directory()
