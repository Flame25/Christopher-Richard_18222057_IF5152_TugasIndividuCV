# Nama: Christopher Richard Chandra
# NIM: 18222057
# Fitur unik: Menambahkan fungsi penyimpanan otomatis hasil deteksi fitur ke dalam folder hasil sesuai metode, Membuat statisik deteksi fitur pada csv

import cv2
import numpy as np
from skimage import data, io
from pathlib import Path
import csv
from typing import List, Tuple, Dict

def prepare_image(img: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 grayscale.

    Parameters
    ----------
    img : np.ndarray
        Input image (float or uint8).

    Returns
    -------
    np.ndarray
        Image in uint8 format.
    """
    if img.max() <= 1:
        return np.array(img * 255, dtype=np.uint8)
    return np.array(img, dtype=np.uint8)


def draw_keypoints(img_gray: np.ndarray, keypoints: List[cv2.KeyPoint],
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw keypoints on a grayscale image.

    Parameters
    ----------
    img_gray : np.ndarray
        Grayscale image.
    keypoints : list of cv2.KeyPoint
        Keypoints to draw.
    color : tuple, optional
        BGR color for keypoints, by default (0, 255, 0)

    Returns
    -------
    np.ndarray
        Image with keypoints drawn.
    """
    img_color = cv2.cvtColor(np.array(img_gray, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img_color, (x, y), 3, color, -1)
    return img_color


def detect_harris(img_gray: np.ndarray, block_size: int = 2, ksize: int = 3,
                  k: float = 0.04, thresh: float = 0.01) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detect Harris corners.

    Parameters
    ----------
    img_gray : np.ndarray
        Input grayscale image (uint8).
    block_size : int
        Neighborhood size for Harris detection.
    ksize : int
        Aperture parameter for Sobel operator.
    k : float
        Harris detector free parameter.
    thresh : float
        Threshold factor for selecting corners.

    Returns
    -------
    tuple
        List of KeyPoint objects and Harris response image.
    """
    img_float = np.array(img_gray, dtype=np.float32)
    harris_resp = cv2.cornerHarris(img_float, block_size, ksize, k)
    harris_resp = cv2.dilate(harris_resp, None)

    keypoints = [
        cv2.KeyPoint(x, y, 1, -1, harris_resp[y, x], 0, -1)
        for y in range(harris_resp.shape[0])
        for x in range(harris_resp.shape[1])
        if harris_resp[y, x] > thresh * harris_resp.max()
    ]
    return keypoints, harris_resp


def detect_sift(img_gray: np.ndarray, sift_detector: cv2.SIFT) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detect SIFT keypoints and descriptors.

    Parameters
    ----------
    img_gray : np.ndarray
        Grayscale image (uint8).
    sift_detector : cv2.SIFT
        Initialized SIFT detector.

    Returns
    -------
    tuple
        List of KeyPoint objects and descriptors.
    """
    keypoints, descriptors = sift_detector.detectAndCompute(img_gray, None)
    return keypoints, descriptors


def detect_fast(img_gray: np.ndarray, fast_detector: cv2.FastFeatureDetector) -> List[cv2.KeyPoint]:
    """
    Detect FAST keypoints.

    Parameters
    ----------
    img_gray : np.ndarray
        Grayscale image (uint8).
    fast_detector : cv2.FastFeatureDetector
        Initialized FAST detector.

    Returns
    -------
    list
        List of KeyPoint objects.
    """
    keypoints = fast_detector.detect(img_gray, None)
    return keypoints


def process_images(images: Dict[str, np.ndarray], output_dir: str) -> List[List]:
    """
    Process images with Harris, SIFT, and FAST detectors.

    Parameters
    ----------
    images : dict
        Dictionary of {name: image_array}.
    output_dir : str
        Directory to save results.

    Returns
    -------
    list
        Summary table with each row: [Image, Method, Num_Features, Mean_Response].
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize detectors
    sift_detector = cv2.SIFT_create()

    fast_detector = cv2.FastFeatureDetector_create(
    threshold=20,
    nonmaxSuppression=True,
    type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    summary_table = []

    for name, img in images.items():
        img_gray = prepare_image(img)
        print(f"Processing: {name}")

        # Harris
        kp_harris, _ = detect_harris(img_gray)
        io.imsave(f"{output_dir}/{name.lower()}_harris.png", draw_keypoints(img_gray, kp_harris))
        summary_table.append([
            name, "Harris", len(kp_harris),
            np.mean([kp.response for kp in kp_harris]) if kp_harris else 0
        ])

        # SIFT
        kp_sift, _ = detect_sift(img_gray, sift_detector)
        img_sift = cv2.drawKeypoints(img_gray, kp_sift, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        io.imsave(f"{output_dir}/{name.lower()}_sift.png", img_sift)
        summary_table.append([
            name, "SIFT", len(kp_sift),
            np.mean([kp.response for kp in kp_sift]) if kp_sift else 0
        ])

        # FAST
        kp_fast = detect_fast(img_gray, fast_detector)
        io.imsave(f"{output_dir}/{name.lower()}_fast.png", draw_keypoints(img_gray, kp_fast))
        summary_table.append([
            name, "FAST", len(kp_fast),
            np.mean([kp.response for kp in kp_fast]) if kp_fast else 0
        ])

    # Save summary CSV
    csv_file = f"{output_dir}/feature_summary.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Method", "Num_Features", "Mean_Response"])
        writer.writerows(summary_table)

    return summary_table


def main():
    custom_dataset = "custom_dataset"
    images = {
        "cameraman": data.camera(),
        "room": io.imread(f"{custom_dataset}/room.jpg", as_gray=True),
        "rubik": io.imread(f"{custom_dataset}/rubik.jpg", as_gray=True)
    }

    output_dir = "feature_results"
    summary = process_images(images, output_dir)
    print("Feature detection completed. Results saved in:", output_dir)
    print("Summary table:")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
