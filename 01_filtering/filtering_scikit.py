# Nama: Christopher Richard Chandra
# NIM: 18222057
# Fitur unik: Menambahkan fungsi penyimpanan otomatis hasil deteksi fitur ke dalam folder hasil sesuai metode.

from skimage import data, filters, io, img_as_ubyte
import matplotlib.pyplot as plt
from pathlib import Path


def process_and_save(image, name_prefix, output_dir="."):
    """Apply Gaussian, Sobel, and Median filters to an image and save results."""
    # Apply filters
    gaussian_img = filters.gaussian(image, sigma=2, mode="mirror")
    sobel_img = filters.sobel(image)
    median_img = filters.median(image)
    
    # Convert to uint8 for saving
    image_uint8 = img_as_ubyte(image)
    gaussian_uint8 = img_as_ubyte(gaussian_img)
    sobel_uint8 = img_as_ubyte(sobel_img)
    median_uint8 = img_as_ubyte(median_img)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save results
    io.imsave(f"{output_dir}/{name_prefix}_original.png", image_uint8)
    io.imsave(f"{output_dir}/{name_prefix}_gaussian.png", gaussian_uint8)
    io.imsave(f"{output_dir}/{name_prefix}_sobel.png", sobel_uint8)
    io.imsave(f"{output_dir}/{name_prefix}_median.png", median_uint8)


def main():
    """Main function to process sample images."""

    output_folder = "scikit_results"
    custom_dataset = "custom_dataset"

    # Load test images
    images = {
        "cameraman": data.camera(),
        "coin": data.coins(),
        "rubik": io.imread(f"{custom_dataset}/rubik.jpg", as_gray=True),
        "dice": io.imread(f"{custom_dataset}/dice.jpg", as_gray=True)
    }

    for name, img in images.items():
        process_and_save(img, name_prefix=name, output_dir=output_folder)
        print(f"✅ Processed and saved filters for '{name}'")


if __name__ == "__main__":
    main()
