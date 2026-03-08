
import argparse
import time
from pathlib import Path
from collections import deque

import cv2 as cv
import numpy as np


# Pass/fail limits
BAND_RATIO_HIGH = 0.05        # if more than 5% of band is black then fail
BAND_RATIO_LOW = 0.003        # if more than 0.3% black then fail
BAND_RATIO_MID = 0.005        # if more than 0.5% black then fail
STD_R_MID = 6.0               # if ring not round (std_r > 6) then fail. looser: allows worse circle
STD_R_LOW = 4.9               # if ring very not round (std_r > 4.9) then fail. stricter: demands better circle
BBOX_PAD = 5                  # extra pixels around ring when we crop
BAND_MARGIN_FRAC = 0.15       # skip 15% from inner/outer edge


# 8 directions for connected components

NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def compute_histogram(image):
    hist_data = np.zeros(256, dtype=np.int64)  # count pixels for each value 0..255
    src_data = image.ravel()  # flatten image to 1D

    ptr = 0
    while ptr < len(src_data):
        h = int(src_data[ptr]) & 0xFF # get the grey value of the current pixel
        hist_data[h] += 1  # add 1 to the count for that grey value
        ptr += 1 # move to next pixel

    return hist_data  # how many pixels have each brightness value


def find_threshold_from_histogram(hist_data):
    total = hist_data.sum()
    if total == 0:
        return 128

    sum_val = 0.0
    for t in range(256):
        sum_val += t * hist_data[t]

    sum_b = 0.0
    w_b = 0
    w_f = 0
    var_max = 0.0
    threshold = 0

    for t in range(256):
        w_b += hist_data[t]
        if w_b == 0:
            continue

        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += t * hist_data[t]

        m_b = sum_b / w_b
        m_f = (sum_val - sum_b) / w_f

        var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f)

        if var_between > var_max:
            var_max = var_between
            threshold = t

    return threshold


def dilate(binary_img, structuring_element):
    height, width = binary_img.shape
    struct_height, struct_width = structuring_element.shape
    pad_height, pad_width = struct_height // 2, struct_width // 2

    padded = np.pad(
        binary_img,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode="constant",
        constant_values=0
    )
    out = np.zeros_like(binary_img, dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            region = padded[row:row + struct_height, col:col + struct_width]

            if np.any((region == 255) & (structuring_element == 1)):  # any white nearby then expand
                out[row, col] = 255

    return out


def erode(binary_img, structuring_element):
    height, width = binary_img.shape
    struct_height, struct_width = structuring_element.shape
    pad_height, pad_width = struct_height // 2, struct_width // 2

    padded = np.pad(
        binary_img,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode="constant",
        constant_values=0
    )
    out = np.zeros_like(binary_img, dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            region = padded[row:row + struct_height, col:col + struct_width]

            if np.all(region[structuring_element == 1] == 255):  # all white then keep
                out[row, col] = 255

    return out


def close(binary_img, structuring_element, iterations=1):
    result = binary_img.copy()
    for _ in range(iterations):
        result = dilate(result, structuring_element)
        result = erode(result, structuring_element)
    return result


def connected_components(binary_img):
    height, width = binary_img.shape
    labels = np.zeros((height, width), dtype=np.int32)
    areas = {}
    current_label = 0

    for row in range(height):
        for col in range(width):

            if binary_img[row, col] != 255 or labels[row, col] != 0:
                continue

            current_label += 1
            labels[row, col] = current_label
            area = 1
            queue = deque([(row, col)])

            while queue:
                current_row, current_col = queue.popleft()
                for delta_row, delta_col in NEIGHBORS_8:
                    neighbor_row = current_row + delta_row
                    neighbor_col = current_col + delta_col

                    if 0 <= neighbor_row < height and 0 <= neighbor_col < width:

                        if binary_img[neighbor_row, neighbor_col] == 255 and labels[neighbor_row, neighbor_col] == 0:
                            labels[neighbor_row, neighbor_col] = current_label
                            area += 1
                            queue.append((neighbor_row, neighbor_col))

            areas[current_label] = area

    return labels, areas


def compute_region_properties(mask):
    row_coords, col_coords = np.where(mask == 255)
    if row_coords.size == 0:
        return {"area": 0, "bbox": (0, 0, 0, 0), "centroid": (0.0, 0.0), "perimeter": 0}

    area = int(row_coords.size)

    min_row, max_row = int(row_coords.min()), int(row_coords.max())
    min_col, max_col = int(col_coords.min()), int(col_coords.max())
    bbox = (min_col, min_row, max_col, max_row)

    centroid_x = float(col_coords.mean())
    centroid_y = float(row_coords.mean())
    centroid = (centroid_x, centroid_y)

    height, width = mask.shape
    perimeter = 0
    for row, col in zip(row_coords, col_coords):
        for delta_row, delta_col in NEIGHBORS_8:
            neighbor_row = row + delta_row
            neighbor_col = col + delta_col
            if (neighbor_row < 0 or neighbor_row >= height or
                neighbor_col < 0 or neighbor_col >= width or
                mask[neighbor_row, neighbor_col] == 0):  # on edge = count as perimeter
                perimeter += 1
                break

    return {"area": area, "bbox": bbox, "centroid": centroid, "perimeter": perimeter}


def classify_oring(largest_mask, props):
    min_col, min_row, max_col, max_row = props["bbox"]
    centroid_x, centroid_y = props["centroid"]
    height, width = largest_mask.shape

    crop_row_start = max(0, min_row - BBOX_PAD)
    crop_row_end = min(height, max_row + BBOX_PAD + 1)
    crop_col_start = max(0, min_col - BBOX_PAD)
    crop_col_end = min(width, max_col + BBOX_PAD + 1)
    crop = largest_mask[crop_row_start:crop_row_end, crop_col_start:crop_col_end]  # zoom in on ring

    crop_height, crop_width = crop.shape
    centroid_x_in_crop = centroid_x - crop_col_start
    centroid_y_in_crop = centroid_y - crop_row_start

    row_coords, col_coords = np.where(crop == 255)
    if row_coords.size == 0:
        return 0.0, 0.0, 0.0, "FAIL"

    # Roundness: distance from centre to each ring pixel. Perfect circle = same radius everywhere
    radius = np.sqrt((col_coords - centroid_x_in_crop) ** 2 + (row_coords - centroid_y_in_crop) ** 2)
    radius_min = float(radius.min())  # inner edge
    radius_max = float(radius.max())  # outer edge
    thickness = radius_max - radius_min  # ring thickness in pixels
    std_r = float(radius.std())  # roundness: low = round, high = bumpy/oval. Std of radius

    # Skip 15% at each edge
    margin = BAND_MARGIN_FRAC * thickness

    row_range = np.arange(crop_height, dtype=np.float32)
    col_range = np.arange(crop_width, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(row_range, col_range, indexing="ij")
    radius_map = np.sqrt((col_grid - centroid_x_in_crop) ** 2 + (row_grid - centroid_y_in_crop) ** 2)

    # Band = middle 70% of ring (skip 15% at each edge)
    band = (radius_map >= radius_min + margin) & (radius_map <= radius_max - margin)
    band_count = int(band.sum())
    if band_count == 0:
        band_bg_ratio = 0.0
    else:
        # Cracks: fraction of band that is black. Good ring = 0, cracked = higher.
        band_bg_ratio = float(((crop == 0) & band).sum()) / band_count

    # Fail criteria:
    # 1. band_bg_ratio > 5%: too many cracks in band
    # 2. std_r > 6 and band_bg_ratio > 0.3%: not round + some cracks
    # 3. std_r > 4.9 and band_bg_ratio > 0.5%: very not round + more cracks
    fail = (
        band_bg_ratio > BAND_RATIO_HIGH
        or (std_r > STD_R_MID and band_bg_ratio > BAND_RATIO_LOW)
        or (std_r > STD_R_LOW and band_bg_ratio > BAND_RATIO_MID)
    )
    return thickness, std_r, band_bg_ratio, ("FAIL" if fail else "PASS")


def process_image(image_path):
    start_time = time.perf_counter()

    img_color = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")


    blue = img_color[..., 0].astype(np.float32)
    green = img_color[..., 1].astype(np.float32)
    red = img_color[..., 2].astype(np.float32)
    img = (0.114 * blue + 0.587 * green + 0.299 * red).astype(np.uint8)  # convert colour image to grayscale

    hist = compute_histogram(img)
    threshold = find_threshold_from_histogram(hist)

    binary = np.zeros_like(img, dtype=np.uint8)  # thresholding
    count_above = np.count_nonzero(img > threshold)
    count_below = img.size - count_above

    if count_above < count_below:  # ring is bright then make bright white
        binary[img > threshold] = 255
    else:  # background bright then invert so ring is white
        binary[img <= threshold] = 255

    structuring_element = np.ones((3, 3), dtype=np.uint8)
    closed = close(binary, structuring_element, iterations=1)  # morphology

    labels, areas = connected_components(closed)  # extract regions

    if areas:
        biggest = max(areas, key=areas.get)  # biggest blob = ring
        largest_mask = np.zeros_like(closed, dtype=np.uint8)
        largest_mask[labels == biggest] = 255
    else:
        largest_mask = np.zeros_like(closed, dtype=np.uint8)

    props = compute_region_properties(largest_mask)
    annotated = img_color.copy()

    if props["area"] > 0:  # analyse region
        min_col, min_row, max_col, max_row = props["bbox"]
        cv.rectangle(annotated, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)

        centroid_x, centroid_y = props["centroid"]
        cv.circle(annotated, (int(centroid_x), int(centroid_y)), 5, (255, 0, 0), -1)


        cv.putText(annotated, f"Area: {props['area']}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(annotated, f"Perimeter: {props['perimeter']}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        _, std_r, band_ratio, label = classify_oring(largest_mask, props)
    else:
        label = "FAIL"
        std_r, band_ratio = 0.0, 0.0
        cv.putText(annotated, "NO RING FOUND", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    time_ms = (time.perf_counter() - start_time) * 1000.0  # elapsed time

    cv.putText(
        annotated,
        label,
        (10, annotated.shape[0] - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0) if label == "PASS" else (0, 0, 255),
        3
    )

    cv.putText(
        annotated,
        f"Time: {time_ms:.1f} ms",
        (10, annotated.shape[0] - 50),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    return annotated, largest_mask, label, time_ms, std_r, band_ratio


def main():
    parser = argparse.ArgumentParser(description="Check O-ring images for defects")
    parser.add_argument("path", type=str, help="Path to image file or folder")
    parser.add_argument("--save", action="store_true", help="Save annotated outputs")
    parser.add_argument("--out", type=str, default="output", help="Output folder (default: output)")
    args = parser.parse_args()

    input_path = Path(args.path)
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return

    if input_path.is_dir():
        valid_exts = {".png", ".jpg"}
        paths = [p for p in input_path.iterdir() if p.suffix.lower() in valid_exts]

        def order_key(path):
            name = path.stem

            if len(name) > 5 and name[5:].isdigit():
                return int(name[5:])
            return 0

        image_paths = sorted(paths, key=order_key)
        if not image_paths:
            print(f"No image files found in: {input_path}")
            return
    else:
        image_paths = [input_path]

    out_dir = None
    if args.save:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        try:
            annotated, largest_mask, label, time_ms, std_r, band_ratio = process_image(img_path)
            print(
                f"{img_path.name}: {label}  ({time_ms:.1f} ms)  std_r={std_r:.2f}  band_ratio={band_ratio:.3f}"
            )

            if out_dir is not None:
                stem = img_path.stem
                cv.imwrite(str(out_dir / f"{stem}_annotated.png"), annotated)
                cv.imwrite(str(out_dir / f"{stem}_mask.png"), largest_mask)

            cv.imshow("Annotated O-ring", annotated)
            key = cv.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
