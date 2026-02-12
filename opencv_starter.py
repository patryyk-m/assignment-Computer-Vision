import cv2 as cv
import numpy as np
from collections import deque


def compute_histogram(image):
    hist = np.zeros(256, dtype=np.int64)
    for value in image.ravel():
        hist[int(value)] += 1
    return hist


def find_threshold_clustering(image, max_iters=100, tol=0.5):
    gray = image.astype(np.float32)
    T = gray.mean()

    for _ in range(max_iters):
        above = gray[gray > T]
        below = gray[gray <= T]

        if above.size == 0 or below.size == 0:
            break

        m1 = above.mean()
        m2 = below.mean()
        new_T = (m1 + m2) / 2.0

        if abs(new_T - T) < tol:
            T = new_T
            break

        T = new_T

    return int(round(T))


def dilate(binary_img, se):
    h, w = binary_img.shape
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2

    padded = np.pad(
        binary_img,
        ((ph, ph), (pw, pw)),
        mode="constant",
        constant_values=0,
    )
    out = np.zeros_like(binary_img, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            region = padded[i : i + sh, j : j + sw]
            if np.any((region == 255) & (se == 1)):
                out[i, j] = 255
    return out


def erode(binary_img, se):
    h, w = binary_img.shape
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2

    padded = np.pad(
        binary_img,
        ((ph, ph), (pw, pw)),
        mode="constant",
        constant_values=0,
    )
    out = np.zeros_like(binary_img, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            region = padded[i : i + sh, j : j + sw]
            if np.all(region[se == 1] == 255):
                out[i, j] = 255
    return out


def close(binary_img, se, iterations=1):
    result = binary_img.copy()
    for _ in range(iterations):
        result = dilate(result, se)
        result = erode(result, se)
    return result


def connected_components(binary_img):
    h, w = binary_img.shape
    labels = np.zeros((h, w), dtype=np.int32)
    areas = {}
    current_label = 0

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(h):
        for x in range(w):
            if binary_img[y, x] == 255 and labels[y, x] == 0:
                current_label += 1
                labels[y, x] = current_label
                area = 1

                queue = deque([(y, x)])
                while queue:
                    cy, cx = queue.popleft()
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and binary_img[ny, nx] == 255
                            and labels[ny, nx] == 0
                        ):
                            labels[ny, nx] = current_label
                            area += 1
                            queue.append((ny, nx))

                areas[current_label] = area

    return labels, areas


def compute_region_properties(mask):
    ys, xs = np.where(mask == 255)
    
    if len(ys) == 0:
        return {
            "area": 0,
            "bbox": (0, 0, 0, 0),
            "centroid": (0, 0),
            "perimeter": 0,
        }
    
    area = len(ys)
    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    bbox = (min_x, min_y, max_x, max_y)
    
    centroid = (float(xs.mean()), float(ys.mean()))
    
    h, w = mask.shape
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    perimeter = 0
    
    for y, x in zip(ys, xs):
        is_boundary = False
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or mask[ny, nx] == 0:
                is_boundary = True
                break
        if is_boundary:
            perimeter += 1
    
    return {
        "area": area,
        "bbox": bbox,
        "centroid": centroid,
        "perimeter": perimeter,
    }


image_path = "Orings/oring1.jpg"

img_color = cv.imread(image_path, cv.IMREAD_COLOR)
if img_color is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
print("Loaded image:", image_path)

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

T = find_threshold_clustering(img)
print("Chosen threshold (clustering):", T)

binary = np.zeros_like(img, dtype=np.uint8)
count_above = np.count_nonzero(img > T)
count_below = np.count_nonzero(img <= T)
if count_above < count_below:
    binary[img > T] = 255
else:
    binary[img <= T] = 255
se = np.ones((3, 3), dtype=np.uint8)
closed = close(binary, se)

labels, areas = connected_components(closed)
if areas:
    largest_label = max(areas, key=areas.get)
    print(f"Largest component: label {largest_label}, area {areas[largest_label]}")

    largest_mask = np.zeros_like(closed, dtype=np.uint8)
    largest_mask[labels == largest_label] = 255
else:
    largest_mask = np.zeros_like(closed, dtype=np.uint8)
    print("No components found")

print("Number of components:", len(areas))
print("White pixels in closed:", int(np.count_nonzero(closed == 255)))
print("White pixels in largest_mask:", int(np.count_nonzero(largest_mask == 255)))

props = compute_region_properties(largest_mask)
print("Region properties:", props)

annotated = img_color.copy()
min_x, min_y, max_x, max_y = props["bbox"]
cv.rectangle(annotated, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

cx, cy = props["centroid"]
cv.circle(annotated, (int(cx), int(cy)), 5, (255, 0, 0), -1)

text_area = f"Area: {props['area']}"
text_perim = f"Perimeter: {props['perimeter']}"
cv.putText(annotated, text_area, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv.putText(annotated, text_perim, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

x = 100
y = 100
if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
    pix = img[x, y]
    print("The pixel value at image location [" + str(x) + "," + str(y) + "] is:" + str(pix))

cv.imshow("Annotated O-ring", annotated)
cv.waitKey(0)
cv.destroyAllWindows()
