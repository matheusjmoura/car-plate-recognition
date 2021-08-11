import sys
from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from skimage import measure
from skimage.measure import regionprops 

car_image = imread("car_img_repo/car_5_w.jpg", as_gray=True)

# 2-dimensional array of img shape

# convert img to grayscale and binary 
# show both imgs side by side with matplot
gray_image = car_image * 255
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(gray_image, cmap="gray")
threshold_value = threshold_otsu(gray_image)
binary_img = gray_image > threshold_value
# ax2.imshow(binary_img, cmap="gray")
# plt.show()

label_img = measure.label(binary_img)
#getting the maximum width and height that a standard car plate have 
plate_dim = (0.08*label_img.shape[0], 0.2*label_img.shape[0], 0.15*label_img.shape[1], 0.4*label_img.shape[1])
min_height, max_height, min_width, max_width = plate_dim
plate_obj_cordinates = []
plate_like_obj = []

fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_image, cmap="gray")


for region in regionprops(label_img):
    if region.area < 50:
        continue
     
    min_r, min_c, max_r, max_c = region.bbox
    region_h = max_r - min_r
    region_w = max_c - min_c
    if region_h >= min_height and region_h <= max_height and region_w >= min_width and region_w <= max_width:
        plate_like_obj.append(binary_img[min_r:max_r, min_c:max_c])
        rectBorder = patches.Rectangle((min_c, min_r), max_c - min_c, max_r - min_r, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)

plt.show()