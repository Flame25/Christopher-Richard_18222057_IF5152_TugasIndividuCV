# Nama: Christopher Richard Chandra
# NIM: 18222057
# Fitur unik: Menambahkan validasi hasil kalibrasi dengan menampilkan ulang gambar yang sudah dikoreksi distorsinya, Melakukan penyimpanan hasil kalibrasi ke sebuah file csv

import cv2
import numpy as np
from skimage import io
from pathlib import Path
import csv
from typing import List, Tuple


def generate_checkerboard(size: Tuple[int, int], square_size: int, margin: int = 0) -> np.ndarray:
    """
    Generate a synthetic checkerboard image.

    Parameters
    ----------
    size : tuple
        Checkerboard inner corners (cols, rows).
    square_size : int
        Pixels per square.
    margin : int
        Margin around the checkerboard.

    Returns
    -------
    np.ndarray
        Checkerboard image (uint8).
    """
    rows, cols = size[1] + 1, size[0] + 1
    height = rows * square_size + 2 * margin
    width = cols * square_size + 2 * margin
    checkerboard = np.zeros((height, width), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                y0 = i * square_size + margin
                x0 = j * square_size + margin
                checkerboard[y0:y0 + square_size, x0:x0 + square_size] = 255

    return checkerboard

def simulate_views(checkerboard: np.ndarray, num_views: int = 5) -> List[np.ndarray]:
    """
    Generate transformed views of the checkerboard with small rotation and translation.

    Parameters
    ----------
    checkerboard : np.ndarray
        Base checkerboard image.
    num_views : int
        Number of simulated views.

    Returns
    -------
    list
        List of transformed checkerboard images.
    """
    views = []
    for _ in range(num_views):
        angle = np.random.uniform(-5, 5)      # degrees
        scale = np.random.uniform(1, 1.05)   # slight scaling
        tx = np.random.randint(-10, 10)
        ty = np.random.randint(-10, 10)

        center = (checkerboard.shape[1] // 2, checkerboard.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[:, 2] += [tx, ty]

        transformed = cv2.warpAffine(checkerboard, M,
                                     (checkerboard.shape[1], checkerboard.shape[0]))
        views.append(transformed)

    return views

def detect_corners(image: np.ndarray, checkerboard_size: Tuple[int, int]) -> Tuple[bool, np.ndarray]:
    """
    Detect chessboard corners in an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    checkerboard_size : tuple
        Checkerboard inner corners (cols, rows).

    Returns
    -------
    tuple
        (ret, corners) where ret is True if corners detected, corners are the detected points.
    """
    ret, corners = cv2.findChessboardCorners(image, checkerboard_size, None)
    if ret:
        corners = cv2.cornerSubPix(
            image, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
    return ret, corners

def calibrate_camera(objpoints: List[np.ndarray], imgpoints: List[np.ndarray],
                     image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Calibrate the camera given object points and image points.

    Parameters
    ----------
    objpoints : list
        3D object points.
    imgpoints : list
        2D image points.
    image_shape : tuple
        Image shape (height, width).

    Returns
    -------
    tuple
        camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors
    """
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape[::-1], None, None
    )
    return camera_matrix, dist_coeffs, rvecs, tvecs

def main():
    output_dir = "geometry_results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    checkerboard_size = (6, 6)
    square_size = 100
    num_views = 5

    checkerboard = generate_checkerboard(checkerboard_size, square_size)
    io.imsave(f"{output_dir}/checkerboard.png", checkerboard)

    objp = np.zeros((checkerboard_size[1] * checkerboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    views = simulate_views(checkerboard, num_views=num_views)

    for i, view in enumerate(views):
        ret, corners = detect_corners(view, checkerboard_size)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            img_corners = cv2.drawChessboardCorners(view.copy(), checkerboard_size, corners, ret)
            io.imsave(f"{output_dir}/checkerboard_corners_view_{i}.png", img_corners)
            print(f"View {i}: corners detected.")
        else:
            print(f"View {i}: corners NOT detected!")

    if objpoints:
        camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, checkerboard.shape)
        print("\nCamera matrix (intrinsic):\n", camera_matrix)
        print("\nDistortion coefficients:\n", dist_coeffs)
        print("\nRotation vectors (extrinsic):\n", rvecs)
        print("\nTranslation vectors (extrinsic):\n", tvecs)

        # Save parameters to CSV
        with open(f"{output_dir}/camera_params.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Values"])
            writer.writerow(["Camera matrix (intrinsic)", camera_matrix.flatten()])
            writer.writerow(["Distortion coefficients", dist_coeffs.flatten()])
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                writer.writerow([f"Rotation vector {i}", rvec.flatten()])
                writer.writerow([f"Translation vector {i}", tvec.flatten()])

        print("\nCalibration complete. Results saved in:", output_dir)
    else:
        print("No corners detected in any view. Calibration failed.")


if __name__ == "__main__":
    main()
