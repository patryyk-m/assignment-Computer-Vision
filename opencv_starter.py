import cv2 as cv
import numpy as np


def compute_histogram(image):
    hist = np.zeros(256, dtype=np.int64)
    for value in image.ravel():
        hist[int(value)] += 1
    return hist


image_path = "Orings/oring1.jpg"

img_color = cv.imread(image_path, cv.IMREAD_COLOR)
if img_color is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

b = img_color[..., 0].astype(np.float32)
g = img_color[..., 1].astype(np.float32)
r = img_color[..., 2].astype(np.float32)
img = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)

hist = compute_histogram(img)
print("Total pixels:", img.size)
peak = int(np.argmax(hist))
print("Peak intensity:", peak, "count:", int(hist[peak]))
nonzero_bins = np.count_nonzero(hist)
print("Non-zero bins:", int(nonzero_bins))
nonzero_indices = np.where(hist > 0)[0]
print(
    "Intensity range:",
    int(nonzero_indices.min()),
    "to",
    int(nonzero_indices.max()),
)

x = 100
y = 100
if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
    pix = img[x, y]
    print("The pixel value at image location [" + str(x) + "," + str(y) + "] is:" + str(pix))

cv.imshow("thresholded image 1", img)
cv.waitKey(0)
cv.destroyAllWindows()
