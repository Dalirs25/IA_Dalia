from pathlib import Path
from collections import deque

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = 'C:/python_projects/IA_Dalia/figuras/figuras_formas.jpg'
TARGET_COLOR = "green"                   


# HSV color ranges (OpenCV scale: H [0..179], S [0..255], V [0..255])
COLOR_RANGES = {
    "blue":   [(90, 60, 40), (140, 255, 255)],
    "green":  [(35, 40, 40), (85, 255, 255)],
    "yellow": [(20, 60, 40), (35, 255, 255)],
    "red_lo": [(0, 60, 40), (10, 255, 255)],
    "red_hi": [(170, 60, 40), (179, 255, 255)],
}


def get_mask_for_color(hsv_img: np.ndarray, color_name: str) -> np.ndarray:
    name = color_name.lower()

    if name == "red":
        # Combine low and high parts of red hue range
        lo1, hi1 = COLOR_RANGES["red_lo"]
        lo2, hi2 = COLOR_RANGES["red_hi"]
        mask1 = cv.inRange(hsv_img, np.array(lo1, np.uint8), np.array(hi1, np.uint8))
        mask2 = cv.inRange(hsv_img, np.array(lo2, np.uint8), np.array(hi2, np.uint8))
        return cv.bitwise_or(mask1, mask2)

    # For regular colors (single range)
    if name not in COLOR_RANGES:
        raise ValueError(f"El color no existe")

    lo, hi = COLOR_RANGES[name]
    return cv.inRange(hsv_img, np.array(lo, np.uint8), np.array(hi, np.uint8))


def count_islands_bfs(binary_img: np.ndarray):
    h, w = binary_img.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0

    # 4-connected neighbors: up, down, left, right
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for y in range(h):
        for x in range(w):
            if binary_img[y, x] and not visited[y, x]:
                count += 1
                # Start BFS from (y, x)
                q = deque([(y, x)])
                visited[y, x] = True

                while q:
                    cy, cx = q.popleft()
                    # Explore 4 neighbors
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        # Check bounds and if not visited and is foreground
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and binary_img[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))

    return count


def main():
    # 1) Load image (BGR)
    bgr = cv.imread(str(IMAGE_PATH))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at: {IMAGE_PATH}")

    # 2) Convert to HSV (for color thresholding)
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)

    # 3) Build color mask (0 or 255)
    mask = get_mask_for_color(hsv, TARGET_COLOR)

    # 5) Convert mask to 0/1 for BFS counting (foreground=1)
    bw = (mask > 0).astype(np.uint8)

    # 6) Count connjected components using BFS
    count = count_islands_bfs(bw)

    # 7) Show the mask and print the result
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Selected Color: {TARGET_COLOR}")
    plt.axis("off")
    plt.show()

    print(f"Cantidad de figuras coloc ({TARGET_COLOR.capitalize()}): {count}")


if __name__ == "__main__":
    main()
