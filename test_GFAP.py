file = '/Users/frasersim/Library/CloudStorage/Box-Box/NewLabData/People/Greg/Noggin Related/20230201/rep 1/all images/Noggin_hOPC__ImageID-16433.tif'

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from singleCompositeImage import singleCompositeImage, showImages

dirname = os.path.dirname(file)
basename = os.path.basename(file)

sCI = singleCompositeImage(dirname, basename, dapi_ch=1)

sCI.processDAPI(threshold_method='th2')

# print(sCI.nucleiMask.shape)

# dilate nucleiMask
gfap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
gfap_mask = cv2.dilate(sCI.nucleiMask, gfap_kernel)

ret, markers = cv2.connectedComponents(gfap_mask)
# markers = markers+1

# print(markers)

# Now, mark the region of unknown with zero
# markers[unknown==255] = 0
#
# img = cv2.normalize(src=sCI.images[2], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# markers = cv2.watershed(img, markers)
# img[markers == -1] = [0,255,0]
#
#
# imageList = [sCI.colorImage(blue=sCI.images[3], red=sCI.images[1], green=sCI.images[2]), sCI.nucleiMask, gfap_mask, markers, img, sCI.nucleiMarkers]
# print(len(imageList))
#
# sCI.showImages(imageList, "Title")

img = cv2.normalize(src=sCI.images[2], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img[markers == -1] = [0,255,0]
#
# marker_intensity = []
# for i in range(1, sCI.nucleiCount):
#     # print(i)
#
#     try:
#         mean_intensity = np.mean(sCI.images[2][markers == i])
#         upper_intensity = np.percentile(sCI.images[2][markers == i], 90)
#         sum_intensity = np.sum(sCI.images[2][markers == i])
#         # print(upper_intensity)
#         marker_intensity.append(upper_intensity)
#         if upper_intensity > 3000:
#             img[markers == i] = [255, 0, 0]
#     except:
#         print(f"Error in {i}")
#         print(sCI.images[2][markers == i])
#
#
# # print(marker_intensity)
# plt.hist(marker_intensity, bins=50)
# plt.show()
#
# sCI.showImages([sCI.colorImage(blue=sCI.images[3], red=sCI.images[1], green=sCI.images[2]),
#                 img])


# trying a new tack

# go nuclei by nuclei, dilate each mask, then calculate the surrounding intensity

gfap_image = sCI.images[2]
gfap_image = sCI.rolling_ball_subtraction(gfap_image)
# sCI.showImages([sCI.colorImage(blue=sCI.images[3], red=sCI.images[1], green=sCI.images[2]), gfap_image])

gfap_intensities = []
for i in range(2, np.max(sCI.nucleiMarkers) ):
    # print(f"Nucleus: {i}")
    img[sCI.nucleiMarkers == i] = [0, 0, 255]
    gfap_mask = np.zeros(sCI.nucleiMarkers.shape, dtype="uint8")
    gfap_mask[sCI.nucleiMarkers == i] = 1

    gfap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
    gfap_mask = cv2.dilate(gfap_mask, gfap_kernel)

    gfap_intensity = np.mean(gfap_image[gfap_mask == 1])
    gfap_intensities.append(gfap_intensity)
    # print(gfap_intensity)

    if gfap_intensity > 1000:
        # positive cell
        img[sCI.nucleiMarkers == i] = [255,0,0]

showImages([sCI.colorImage(blue=sCI.images[3], red=sCI.images[1], green=sCI.images[2]),
                img])

plt.hist(gfap_intensities, bins=50)
plt.show()
