# Nama: Christopher Richard Chandra
# NIM: 18222057
# Fitur unik: Menambahkan fungsi penyimpanan otomatis hasil deteksi fitur ke dalam folder hasil sesuai metode, Pencatatan parameter di file csv

import numpy as np
from skimage import data, io, img_as_ubyte
from skimage.filters import sobel
from skimage.feature import canny
from pathlib import Path
import csv


# --- Utility functions ---

def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize a floating-point image to uint8 [0, 255].

    Parameters
    ----------
    image : np.ndarray
        Input image (float or uint8).

    Returns
    -------
    np.ndarray
        Normalized uint8 image.
    """
    img_min, img_max = image.min(), image.max()
    if img_max - img_min == 0:
        return np.zeros_like(image, dtype=np.uint8)
    norm_img = (image - img_min) / (img_max - img_min)
    return (norm_img * 255).astype(np.uint8)


def save_image(image: np.ndarray, filename: str) -> None:
    """
    Save an image after converting to uint8 if necessary.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    filename : str
        Output filename.
    """
    if image.dtype != np.uint8:
        image = normalize_to_uint8(image)
    io.imsave(filename, image)


# --- Edge detection functions ---
def sobel_edge_detection(img_gray: np.ndarray) -> np.ndarray:
    """
    Compute Sobel edges for a grayscale image.

    Parameters
    ----------
    img_gray : np.ndarray
        Grayscale image (float in [0,1] or uint8).

    Returns
    -------
    np.ndarray
        Sobel edge image (float).
    """
    return sobel(img_gray)


def canny_edge_detection(
    img_gray: np.ndarray, sigma: float = 1.0, low_threshold: float = 0.1, high_threshold: float = 0.3
) -> np.ndarray:
    """
    Compute Canny edges for a grayscale image.

    Parameters
    ----------
    img_gray : np.ndarray
        Grayscale image (float in [0,1]).
    sigma : float, optional
        Gaussian smoothing parameter, by default 1.0
    low_threshold : float, optional
        Low hysteresis threshold [0,1], by default 0.1
    high_threshold : float, optional
        High hysteresis threshold [0,1], by default 0.3

    Returns
    -------
    np.ndarray
        Binary edge image (bool).
    """
    return canny(img_gray, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

def process_edge_detection(
    images: dict,
    output_dir: str,
    sigma_values: list = [1, 2],
    low_thresholds: list = [0.1, 0.2],
    high_thresholds: list = [0.3, 0.5],
) -> list:
    """
    Perform Sobel and Canny edge detection on multiple images and save results.

    Parameters
    ----------
    images : dict
        Dictionary of {image_name: image_array}.
    output_dir : str
        Directory to save results.
    sigma_values : list, optional
        List of Gaussian sigmas for Canny, by default [1,2]
    low_thresholds : list, optional
        List of low thresholds for Canny, by default [0.1,0.2]
    high_thresholds : list, optional
        List of high thresholds for Canny, by default [0.3,0.5]

    Returns
    -------
    list
        Summary table with each row: [Image, Method, Threshold/Sigma, Num_Edge_Pixels]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_table = []

    for name, img in images.items():
        print(f"Processing: {name}")

        img_gray = np.array(img, dtype=float)
        if img_gray.max() > 1:
            img_gray /= 255.0  # normalize to [0,1]

        # --- Save original image ---
        save_image(img_gray, f"{output_dir}/{name.lower()}_before.png")

        # --- Sobel edges ---
        edges_sobel = sobel_edge_detection(img_gray)
        save_image(edges_sobel, f"{output_dir}/{name.lower()}_sobel_after.png")
        summary_table.append([name, "Sobel", "N/A", int(np.sum(edges_sobel > 0.1))])

        # --- Canny edges ---
        for sigma, low, high in zip(sigma_values, low_thresholds, high_thresholds):
            edges_canny = canny_edge_detection(img_gray, sigma=sigma, low_threshold=low, high_threshold=high)
            filename_after = f"{output_dir}/{name.lower()}_canny_sigma{sigma}_th{low}-{high}_after.png"
            save_image(img_as_ubyte(edges_canny), filename_after)
            summary_table.append([name, f"Canny_sigma{sigma}", f"{low}-{high}", int(np.sum(edges_canny))])

    # --- Save summary CSV ---
    csv_file = f"{output_dir}/threshold_table.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Method", "Threshold/Sigma", "Num_Edge_Pixels"])
        writer.writerows(summary_table)

    return summary_table


def main():
    custom_dataset = "custom_dataset"
    images = {
        "cameraman": data.camera(),
        "coin": data.coins(),
        "rubik": io.imread(f"{custom_dataset}/rubik.jpg", as_gray=True),
        "dice": io.imread(f"{custom_dataset}/dice.jpg", as_gray=True),
    }

    output_dir = "edge_results"
    summary = process_edge_detection(images, output_dir)

    print("Edge detection completed. Results saved in:", output_dir)
    print("Summary table:")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
