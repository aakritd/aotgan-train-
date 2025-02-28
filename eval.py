import argparse
from glob import glob
from multiprocessing import Pool

import numpy as np
import metric as module_metric
from PIL import Image
from tqdm import tqdm

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Image Inpainting Metrics Evaluation")
parser.add_argument("--real_dir", required=True, type=str, help="Directory containing real images")
parser.add_argument("--fake_dir", required=True, type=str, help="Directory containing fake images")
parser.add_argument("--metric", type=str, nargs="+", choices=["mae", "psnr", "ssim", "fid"],
                    required=True, help="Metrics to compute (space-separated). Options: mae, psnr, ssim, fid.")
args = parser.parse_args()


def read_img(name_pair):
    """Read image pairs (real and fake) and return them as numpy arrays."""
    rname, fname = name_pair
    rimg = Image.open(rname)
    fimg = Image.open(fname)

    # Resize images to the same size (if needed)
    if rimg.size != fimg.size:
        fimg = fimg.resize(rimg.size)

    # Convert images to RGB (if they have alpha channel or different color mode)
    rimg = rimg.convert("RGB")
    fimg = fimg.convert("RGB")

    return np.array(rimg), np.array(fimg)


def main(num_worker=8):
    # Get sorted lists of image files
    real_names = sorted(glob(f"{args.real_dir}/*.png"))
    fake_names = sorted(glob(f"{args.fake_dir}/*.png"))
    print(f"Real images: {len(real_names)}, Fake images: {len(fake_names)}")

    if len(real_names) != len(fake_names):
        raise ValueError("Number of real and fake images must be equal.")

    # Load images in parallel
    real_images, fake_images = [], []
    pool = Pool(num_worker)
    for rimg, fimg in tqdm(
        pool.imap_unordered(read_img, zip(real_names, fake_names)), total=len(real_names), desc="Loading images"
    ):
        real_images.append(rimg)
        fake_images.append(fimg)

    
    
    # Prepare metrics
    metrics = {met: getattr(module_metric, met) for met in args.metric}
    evaluation_scores = {key: 0 for key in metrics}

    # Compute metrics
    for key, metric_function in metrics.items():
        evaluation_scores[key] = metric_function(real_images, fake_images, num_worker=num_worker)

    # Print results
    print("\nEvaluation Results:")
    for key, value in evaluation_scores.items():
        print(f"{key.upper()}: {value:.6f}")


if __name__ == "__main__":
    main()
