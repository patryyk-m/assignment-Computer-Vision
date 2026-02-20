import argparse
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from collections import deque

STD_THRESH = 8.0
HOLE_RATIO_THRESH = 0.065
BBOX_PAD = 5
BAND_MARGIN_FRAC = 0.15


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


def classify_oring(largest_mask, props):
    min_x, min_y, max_x, max_y = props["bbox"]
    cx, cy = props["centroid"]
    h, w = largest_mask.shape

    y0 = max(0, min_y - BBOX_PAD)
    y1 = min(h, max_y + BBOX_PAD + 1)
    x0 = max(0, min_x - BBOX_PAD)
    x1 = min(w, max_x + BBOX_PAD + 1)
    mask_crop = largest_mask[y0:y1, x0:x1]

    crop_h, crop_w = mask_crop.shape
    cx_crop = cx - x0
    cy_crop = cy - y0

    ys, xs = np.where(mask_crop == 255)
    if len(ys) == 0:
        return 0.0, 0.0, 0.0, "FAIL"

    r = np.sqrt((xs - cx_crop) ** 2 + (ys - cy_crop) ** 2)
    r_min = float(r.min())
    r_max = float(r.max())
    thickness = r_max - r_min
    std_r = float(r.std())

    margin = BAND_MARGIN_FRAC * thickness
    rr = np.arange(crop_h, dtype=np.float32)
    cc = np.arange(crop_w, dtype=np.float32)
    yy, xx = np.meshgrid(rr, cc, indexing="ij")
    r_map = np.sqrt((xx - cx_crop) ** 2 + (yy - cy_crop) ** 2)
    band_mask = (r_map >= r_min + margin) & (r_map <= r_max - margin)
    band_count = int(band_mask.sum())
    if band_count == 0:
        band_background_ratio = 0.0
    else:
        band_bg = (mask_crop == 0) & band_mask
        band_background_ratio = float(band_bg.sum()) / band_count

    fail = std_r > STD_THRESH or band_background_ratio > HOLE_RATIO_THRESH
    label = "FAIL" if fail else "PASS"

    return thickness, std_r, band_background_ratio, label


def process_image(image_path):
    t0 = time.perf_counter()

    img_color = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    b = img_color[..., 0].astype(np.float32)
    g = img_color[..., 1].astype(np.float32)
    r = img_color[..., 2].astype(np.float32)
    img = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)

    hist = compute_histogram(img)
    T = find_threshold_clustering(img)

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
        largest_mask = np.zeros_like(closed, dtype=np.uint8)
        largest_mask[labels == largest_label] = 255
    else:
        largest_mask = np.zeros_like(closed, dtype=np.uint8)

    props = compute_region_properties(largest_mask)
    annotated = img_color.copy()

    if props["area"] > 0:
        min_x, min_y, max_x, max_y = props["bbox"]
        cv.rectangle(annotated, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        cx, cy = props["centroid"]
        cv.circle(annotated, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        text_area = f"Area: {props['area']}"
        text_perim = f"Perimeter: {props['perimeter']}"
        cv.putText(annotated, text_area, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(annotated, text_perim, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        thickness, std_r, band_background_ratio, result_label = classify_oring(largest_mask, props)
    else:
        result_label = "FAIL"
        std_r, band_background_ratio = 0.0, 0.0
        cv.putText(annotated, "NO RING FOUND", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    t1 = time.perf_counter()
    time_ms = (t1 - t0) * 1000.0

    cv.putText(
        annotated, result_label, (10, annotated.shape[0] - 20),
        cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0) if result_label == "PASS" else (0, 0, 255), 3,
    )
    cv.putText(
        annotated, f"Time: {time_ms:.1f} ms", (10, annotated.shape[0] - 50),
        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    return annotated, largest_mask, result_label, time_ms, std_r, band_background_ratio


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
        image_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in valid_exts])
        if not image_paths:
            print(f"No image files found in: {input_path}")
            return
    else:
        image_paths = [input_path]

    if args.save:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        try:
            annotated, largest_mask, result_label, time_ms, std_r, band_ratio = process_image(img_path)
            print(f"{img_path.name}: {result_label}  ({time_ms:.1f} ms)  std_r={std_r:.2f}  band_ratio={band_ratio:.3f}")

            if args.save:
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
