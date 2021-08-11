import sys
import numpy as np 
import cv2 as cv
import imutils
import pytesseract
import os
import time

def auto_canny (image, sigma=0.33):
        v = np.median(image)
        lower = int(max(0, (1 - sigma) * v ))
        upper = int(min(255, (1 + sigma) * v ))
        ed = cv.Canny(image, lower, upper)
        return ed

def open_image (path):
    image = cv.imread(path)
    image = imutils.resize(image, width=500)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = cv.GaussianBlur(image_gray, (3, 3), 0)
    return image, image_gray

def find_contours (edged_image):
        cnts = cv.findContours(edged_image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv.contourArea, reverse=True)[:10]
        screenCnt = None
        
        for c in cnts:
                perimeter = cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, 0.018 * perimeter, True)
                if (len(approx) == 4):
                        screenCnt = approx
                        break
        return screenCnt

def licence_plate_find (image, image_gray, contours):
        if not (contours is None):
                mask = np.zeros(image_gray.shape, np.uint8)
                new_img = cv.drawContours(mask, [contours], 0, 255, -1)
                new_image = cv.bitwise_and(image,image, mask=mask)

                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                licence_plate = image[topx:bottomx+1, topy:bottomy+1]

                licence_plate = cv.cvtColor(licence_plate, cv.COLOR_BGR2GRAY)
                licence_plate = cv.GaussianBlur(licence_plate, (3, 3), 0)
                licence_plate = imutils.resize(licence_plate, width=800)
                return licence_plate
        else:
                return None

def plate_to_text (plate):
        return pytesseract.image_to_string(plate, config="-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --psm 11").encode('ascii', 'ignore')

def analyze_dir(path):
        files = os.listdir(path)
        licence_plates = []     
        sys.stdout.write("Analyzing "+ path + " ")
        for index, file in enumerate(files):
                string_img = path + file
                licence_plates.append([string_img,run(string_img)])
                sys.stdout.write('.')
                sys.stdout.flush() 

        print("\n\nOutput =>")
        for plate_obj in licence_plates:
                print plate_obj

def analyze_single(path):
        result = [path, run(path, silence=False)]
        print result

def run(image_path, silence=True):
        img, img_gray = open_image(image_path)
        ed = auto_canny(img_gray)
        cnts = find_contours(ed)
        plate = licence_plate_find(img, img_gray, cnts)
        if (silence is not True): 
                cv.imshow('plate_output', plate)
                cv.waitKey(0)

        if (plate is not None):
                return plate_to_text(plate)
        else:
                 return -1

if __name__ == '__main__':
        if (len(sys.argv) == 3):
                analyze_dir(sys.argv[1])
        else:
                analyze_single(sys.argv[1])
