# Nama: Christoher Richard Chandra
# NIM: 18222057
# Fitur unik: Mengimplementasikan Gaussian dan Sobel filter dari nol menggunakan NumPy, bukan fungsi bawaan OpenCV.

import numpy as np
from skimage import data, io, img_as_ubyte
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from pathlib import Path

def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian kernel.
    
    Parameters
    ----------
    size : int
        The width and height of the kernel (e.g., 3 means a 3x3 kernel).
    sigma : float
        Standard deviation (spread) of the Gaussian curve.
        Larger sigma = smoother / wider blur.
    
    Returns
    -------
    kernel : np.ndarray
        The normalized 2D Gaussian kernel.
    """
    
    # Create a 1D range of coordinates centered at 0
    # Example (size=5): [-2, -1, 0, 1, 2]
    ax = np.linspace(-(size // 2), size // 2, size)
    
    # Create 2D coordinate grids (xx for x positions, yy for y positions)
    # meshgrid expands the 1D coordinates into a full 2D grid
    # so we can calculate distances for every (x, y) point in the kernel.
    xx, yy = np.meshgrid(ax, ax)
    
    # Compute the Gaussian function for each (x, y) pair.
    # (x^2 + y^2) gives the squared distance from the center (0,0).
    # Pixels farther from the center have smaller weights,
    # and sigma controls how fast this weight decreases with distance.
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel * 1/ np.sqrt(2* 3.14* sigma**2)
    
    # Normalize the kernel so all weights sum to 1
    # (this keeps the overall image brightness unchanged when used for blurring)
    return kernel

def apply_gaussian_blur(image: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to an image using a convolution operation.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image array.
    size : int
        Size of the Gaussian kernel.
    sigma : float
        Spread of the Gaussian function.
    
    Returns
    -------
    blurred : np.ndarray
        Blurred version of the input image.
    """
    kernel = gaussian_kernel(size, sigma)
    blurred = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    return blurred


def binom(k: int, n: int) -> int:
    """
    Compute binomial coefficient C(n, k) safely.

    Parameters
    ----------
    k : int
        The subset size.
    n : int
        The total number of elements.

    Returns
    -------
    int
        Binomial coefficient.
    """
    if k < 0 or k > n:
        return 0
    result = 1
    for i in range(1, k + 1):
        result = result * (n - i + 1) // i
    return result


def diffx_win_array(winsize: int) -> np.ndarray:
    """
    Compute derivative coefficients (diffx_win) for Sobel kernel.

    Parameters
    ----------
    winsize : int
        Window size.

    Returns
    -------
    np.ndarray
        Derivative coefficients array of length winsize.
    """
    diff = np.zeros(winsize, dtype=np.int32)
    for x_win in range(winsize):
        diff[x_win] = binom(x_win, winsize - 2) - binom(x_win - 1, winsize - 2)
    return diff


def smoothx_win_array(winsize: int) -> np.ndarray:
    """
    Compute smoothing coefficients (smoothx_win) for Sobel kernel.

    Parameters
    ----------
    winsize : int
        Window size.

    Returns
    -------
    np.ndarray
        Smoothing coefficients array of length winsize.
    """
    smooth = np.zeros(winsize, dtype=np.int32)
    for x_win in range(winsize):
        smooth[x_win] = binom(x_win, winsize - 1)
    return smooth


def generate_sobel_kernel(winsize: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D Sobel kernel (horizontal and vertical) using binomial coefficients.

    Reference:
    Nixon, M. S., & Aguado, A. S. (2010). Feature Extraction and Image Processing
    (2nd ed., reprinted), pages 123-125.

    Parameters
    ----------
    winsize : int
        Size of the Sobel kernel (nxn).

    Returns
    -------
    tuple of np.ndarray
        (horizontal kernel, vertical kernel)
    """
    smooth = smoothx_win_array(winsize)
    diff = diffx_win_array(winsize)

    print("Smooth coefficients (smoothx_win):", smooth)
    print("Derivative coefficients (diffx_win):", diff)

    sobel_h = np.outer(smooth, diff)  # horizontal derivative
    sobel_v = np.outer(diff, smooth)  # vertical derivative

    print("2D Sobel kernel (Horizontal for X):\n", sobel_h)
    print("2D Sobel kernel (Vertical for Y):\n", sobel_v)

    return sobel_h, sobel_v


def save_kernel_image(kernel: np.ndarray, filename: str):
    """
    Save a visualization of a kernel as an image file.

    Parameters
    ----------
    kernel : np.ndarray
        Kernel to visualize.
    filename : str
        Output filename.
    """
    side = kernel.shape[0]
    fig_size = max(2, side / 2)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(kernel, cmap='viridis')
    plt.title(f'Kernel ({side}x{side})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize a float image (can be negative) to uint8 [0,255].

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Normalized uint8 image.
    """
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min == 0:
        return np.zeros_like(image, dtype=np.uint8)
    norm_img = (image - img_min) / (img_max - img_min)
    return (norm_img * 255).astype(np.uint8)


def main(): 

    # --- Dataset ---
    custom_dataset = "custom_dataset"
    images = {
        "cameraman": data.camera(),
        "coin": data.coins(),
        "rubik": io.imread(f"{custom_dataset}/rubik.jpg", as_gray=True),
        "dice": io.imread(f"{custom_dataset}/dice.jpg", as_gray=True)
    }

    output_dir = Path("numpy_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gaussian_filter").mkdir(parents=True, exist_ok=True)
    (output_dir / "sobel_filter").mkdir(parents=True, exist_ok=True)

    # --- Gaussian Blur (placeholder, implement gaussian_kernel/apply_gaussian_blur) ---
    kernel_size = 11
    sigma = 2
    kernel = gaussian_kernel(kernel_size, sigma)
    save_kernel_image(kernel, output_dir / "gaussian_filter" / "gaussian_kernel.png")
    for name, img in images.items():
        blurred_img = apply_gaussian_blur(img, kernel_size, sigma)
        io.imsave(output_dir / "gaussian_filter" / f"{name.lower()}_original.png", img_as_ubyte(img))
        io.imsave(output_dir / "gaussian_filter" / f"{name.lower()}_blurred.png", img_as_ubyte(blurred_img / blurred_img.max()))

    # --- Sobel Filters ---
    sobel_size = 3
    sobel_h, sobel_v = generate_sobel_kernel(sobel_size)

    for name, img in images.items():
        print(f"Processing Sobel: {name}")
        img_float = np.array(img, dtype=float)

        grad_x = ndimage.convolve(img_float, sobel_h, mode='reflect')
        grad_y = ndimage.convolve(img_float, sobel_v, mode='reflect')

        grad_mag = np.hypot(grad_x, grad_y)

        # Normalize and save
        io.imsave(output_dir / "sobel_filter" / f"{name.lower()}_sobel_horizontal.png", normalize_to_uint8(grad_x))
        io.imsave(output_dir / "sobel_filter" / f"{name.lower()}_sobel_vertical.png", normalize_to_uint8(grad_y))
        io.imsave(output_dir / "sobel_filter" / f"{name.lower()}_sobel_magnitude.png", normalize_to_uint8(grad_mag))


if __name__ == "__main__":
    main()
