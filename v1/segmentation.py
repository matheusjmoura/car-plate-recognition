import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cca2

'''
cca2.plate_like_obj its a N-dimensional vector that contains possible car plate areas
todo 
    :-> the licence_plate its passed by a fixed index of plate_like_obj, make this dinamic 
        and find a way to remove non plate_like objects from this vector to proced
'''
print (cca2.plate_like_obj[0])
license_plate = np.invert(cca2.plate_like_obj[0])
labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")

character_dim = (0.35*license_plate.shape[0], 0.65*license_plate.shape[0], 0.05*license_plate.shape[1], 0.35*license_plate.shape[1])

min_h, max_h, min_w, max_w = character_dim

characters = []
counter = 0
column_lits = []

for regions in regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    region_h = y1 - y0
    region_w = x1 - x0

    if region_h >= min_h and region_h <= max_h and region_w >= min_w and region_w <= max_w:
        roi = license_plate[y0:y1, x0:x1]
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)
        column_lits.append(x0)

plt.show()
