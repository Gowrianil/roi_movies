"""
roicoalign_time.py

Batch co-alignment and movie creation for solar sunspot FITS images using cross-correlation.

Features:
- Filters FITS files by filter band and optional time range.
- Interactive ROI selection on a reference image.
- Batch alignment using cross-correlation (via sunkit_image).
- Crops aligned images and saves as JPG.
- Creates an MP4 movie from aligned images.
- Command-line interface for flexible batch processing.

Usage:
    python roicoalign_time.py <in_folder> <out_folder> <jpg_folder> <movie_name>
        [--num_images N] [--start_time YYYY-MM-DDTHH:MM:SS] [--end_time YYYY-MM-DDTHH:MM:SS]
        [--filter NB03] [--ref_idx 0] [--save_fits] [--method cross-correlation]

Arguments:
    in_folder:      Directory containing FITS files.
    out_folder:     Output directory for results and movie.
    jpg_folder:     Directory to save aligned JPG images.
    movie_name:     Name of the output MP4 movie.
    --num_images:   Number of images to process (default: 100).
    --start_time:   Optional start time for filtering files.
    --end_time:     Optional end time for filtering files.
    --filter:       Filter band to select (default: NB03).
    --ref_idx:      Index of reference image for ROI selection.
    --save_fits:    Save aligned FITS files (not implemented).
    --method:       Alignment method (only 'cross-correlation' supported).

Dependencies:
    sunpy, sunkit_image, astropy, numpy, matplotlib, cv2, customLogging, colormap

"""

import os, sys
import argparse
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy import units as u
from multiprocessing import Pool
import sunpy.map
from sunkit_image.coalignment import calculate_match_template_shift, apply_shifts
from pathlib import Path
from colormap import filterColor
from customLogging import *
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
from datetime import datetime, timedelta
from matplotlib.widgets import RectangleSelector

def select_roi_with_mouse(sunpy_map):
    """
    Allows user to interactively select a rectangular ROI on a SunPy map.
    Returns a SunPy submap of the selected region.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=sunpy_map)
    ax.set_title("Select ROI (click and drag) then close the window")
    sunpy_map.plot(axes=ax)
    coords = []

    def onselect(eclick, erelease):
        coords.append((eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata))

    toggle_selector = RectangleSelector(ax, onselect, useblit=True,
                                        button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                        interactive=True)
    plt.show()

    if not coords:
        raise RuntimeError("ROI selection cancelled or failed.")

    x1, y1, x2, y2 = coords[0]
    bottom_left = (min(x1, x2), min(y1, y2)) * u.pixel
    top_right = (max(x1, x2), max(y1, y2)) * u.pixel
    submap = sunpy_map.submap(bottom_left=bottom_left, top_right=top_right)
    return submap

def roi_co_align(in_folder, out_folder, jpg_folder, mp4_name, num_images=200, filter='NB03',
                  ref_idx=0, save_fits=False, method='cross-correlation',
                    start_time = None, end_time = None):
    """
    Main function for batch co-alignment and movie creation.
    - Filters files by filter and time range.
    - Selects reference ROI interactively.
    - Processes images in batches, aligns, crops, and saves.
    - Creates a movie from aligned images.
    """
    # Clean jpg_folder before processing
    if os.path.isdir(jpg_folder) and os.listdir(jpg_folder):
        for f in os.listdir(jpg_folder):
            path = os.path.join(jpg_folder, f)
            if os.path.isfile(path):
                os.remove(path)
                logger.debug(f"Deleted {path}")

    start_dt = None
    end_dt = None

    # Parse optional start and end times
    if start_time:
        try:
            start_dt = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
            logger.info(f"Filtering files starting from: {start_dt}")
        except ValueError:
            logger.error(f"Invalid start time format: {start_time}. Use YYYY-MM-DDTHH:MM:SS.")
            sys.exit(1)
    if end_time:
        try:
            end_dt = datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
            logger.info(f"Filtering files ending at: {end_dt}")
        except ValueError:
            logger.error(f"Invalid end time format: {end_time}. Use YYYY-MM-DDTHH:MM:SS.")
            sys.exit(1)

    # Collect FITS files matching filter
    collected_files = [os.path.join(in_folder, fname)
                       for fname in os.listdir(in_folder)
                       if fname.endswith('.fits') and fname.split('_')[-1][4:8] in filter]

    # Filter files based on time range
    time_filtered_files = []
    for f in collected_files:
        try:
            file_dt = datetime.strptime(os.path.basename(f).split('_')[5], "%Y-%m-%dT%H.%M.%S.%f")
            if (start_dt is None or file_dt >= start_dt) and \
               (end_dt is None or file_dt <= end_dt):
                time_filtered_files.append(f)
        except (ValueError, IndexError):
            logger.warning(f"Could not parse timestamp from filename: {f}. Skipping.")

    # Sort the time-filtered files
    sorted_files = sorted(time_filtered_files, key=lambda file_name: datetime.strptime(os.path.basename(file_name).split('_')[5], "%Y-%m-%dT%H.%M.%S.%f"))

    if not sorted_files:
        logger.error(f"No FITS files found matching pattern: {in_folder}")
        if start_dt or end_dt:
            logger.error(f"Or no files found within the specified time range.")
        sys.exit(1)

    try:
        if ref_idx < 0 or ref_idx >= len(sorted_files):
            logger.error(f"Reference index {ref_idx} is out of bounds for {len(sorted_files)} filtered files.")
            sys.exit(1)
        ref_img = sunpy.map.Map(sorted_files[ref_idx])
        logger.info(f"Using reference image: {sorted_files[ref_idx]}")
        ref_submap = select_roi_with_mouse(ref_img)
    except Exception as e:
        logger.error(f"Failed to select ROI: {e}")
        sys.exit(1)

    # Apply num_images limit *after* filtering and sorting
    if len(sorted_files) > num_images:
        sorted_files = sorted_files[:num_images]
        logger.info(f"Limited processing to the first {num_images} files within the time range.")

    batch_size = 20
    batches = [sorted_files[i:i + batch_size] for i in range(0, len(sorted_files), batch_size)]

    # Process each batch (reference image is always included)
    for batch in batches:
        process_batch([sorted_files[ref_idx]] + batch, ref_submap, method, save_fits, jpg_folder, out_folder)
    
    movie_name = os.path.join(out_folder, mp4_name)
    create_movie(jpg_folder, movie_name)

def process_batch(batch_files, ref_submap, method, save_fits, jpg_fold, out_folder):
    """
    Aligns a batch of FITS files to the reference ROI using cross-correlation.
    Crops and saves aligned images as JPG.
    """
    ref_head = ref_submap.fits_header
    ref_scale_x = ref_submap.scale.axis1
    ref_scale_y = ref_submap.scale.axis2

    filtered_files = [f for f in batch_files if sunpy.map.Map(f).meta.get('QVAL', None) == 100]

    if not filtered_files:
        logger.error("No files with QVAL = 100 found.")
        return

    try:
        Sequence = sunpy.map.Map(filtered_files, sequence=True)
    except Exception as e:
        logger.error(f"Failed to create SunPy map sequence: {e}")
        return

    if method == 'cross-correlation':
        if len(Sequence) <= 1:
            logger.info("Zero or one image in sequence. Assuming zero shift.")
            shifts_x_pix = np.zeros(len(Sequence))
            shifts_y_pix = np.zeros(len(Sequence))
        else:
            try:
                align_shift = calculate_match_template_shift(Sequence, template=ref_submap)
                shifts_x_pix = -(align_shift['x'] / ref_scale_x).to(u.pix).value
                shifts_y_pix = -(align_shift['y'] / ref_scale_y).to(u.pix).value
            except Exception as e:
                logger.error(f"Alignment failed: {e}")
                return

    try:
        aligned_maps = apply_shifts(Sequence, yshift=shifts_y_pix * u.pixel, xshift=shifts_x_pix * u.pixel, clip=False)
    except Exception as e:
        logger.error(f"Applying shifts failed: {e}")
        return
    
    for j in range(1, len(aligned_maps)):
        if abs(shifts_x_pix[j]) < 200 and abs(shifts_y_pix[j]) < 200:
            img = aligned_maps[j]
            img_head = img.fits_header.copy()
            img_head['CRPIX1'] += shifts_x_pix[j]
            img_head['CRPIX2'] += shifts_y_pix[j]
            try:
                aligned_img_full = sunpy.map.Map(img.data, img_head)
                crop_pixels = 50
                ny, nx = aligned_img_full.data.shape
                if nx > 2 * crop_pixels and ny > 2 * crop_pixels:
                    bottom_left_crop = (crop_pixels, crop_pixels) * u.pixel
                    top_right_crop = (nx - crop_pixels, ny - crop_pixels) * u.pixel
                    aligned_img_cropped = aligned_img_full.submap(bottom_left_crop, top_right=top_right_crop)
                    fl_nm = os.path.join(jpg_fold, f"{Path(filtered_files[j]).stem}.jpg")
                    save_image(aligned_img_cropped, fl_nm)
                else:
                    logger.warning(f"Image {Path(filtered_files[j]).stem} is too small ({nx}x{ny}) to crop by {crop_pixels} pixels. Saving uncropped.")
                    fl_nm = os.path.join(jpg_fold, f"{Path(filtered_files[j]).stem}.jpg")
                    save_image(aligned_img_full, fl_nm)
            except Exception as e:
                logger.error(f"Failed to save aligned image: {e}")
        else:
            logger.warning(f"Image shift too large: {filtered_files[j]}")

def save_image(aligned_img, file_name):
    """
    Saves a SunPy map as a JPG image with annotation.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=aligned_img)
    cmap = filterColor.get(aligned_img.meta.get('FILTER', 'NB03'), 'gray')
    aligned_img.plot(title=False, #clip_interval = (1,99.0)* u.percent,
                     vmin= 200,
                     vmax= 40000
                     )
    plot_str = f"{aligned_img.date}  {aligned_img.meta.get('FTR_NAME')}"
    ax.text(0.2, 0.95, plot_str, color='white', weight='bold', fontsize=16, transform=ax.transAxes)
    plt.axis('off')
    plt.tight_layout()
    try:
        plt.savefig(file_name, bbox_inches='tight', transparent=True, pad_inches=0)
    except Exception as e:
        logger.error(f"Failed to save image {file_name}: {e}")
    finally:
        plt.close()

def create_movie(jpg_fold, movie_name):
    """
    Creates an MP4 movie from a sequence of JPG images.
    """
    rate = 20
    img_files = [os.path.join(jpg_fold, fname)
                    for fname in os.listdir(jpg_fold)
                    if fname.endswith('.jpg')]
    img_files = sorted(img_files, key=lambda file_name: datetime.strptime(os.path.basename(file_name).split('_')[5], "%Y-%m-%dT%H.%M.%S.%f"))
    if not img_files:
        logger.error(f"No images found in folder: {jpg_fold}")
        return

    # Ensure the output directory for the movie exists
    movie_dir = os.path.dirname(movie_name)
    os.makedirs(movie_dir, exist_ok=True)
    first_frame = cv2.imread(img_files[0])
    height, width, layers = first_frame.shape
    # Try using 'mp4v' codec as 'avc1' might not be available
    video = cv2.VideoWriter(movie_name, cv2.VideoWriter_fourcc(*'mp4v'), rate, (width, height))
    for img in img_files:
        frame = cv2.imread(img)
        if frame is not None:
            video.write(frame)
    video.release()
    logger.info(f"Movie created successfully: {movie_name}")

def main():
    """
    Parses command-line arguments and runs the co-alignment workflow.
    """
    parser = argparse.ArgumentParser(description="Co-align solar sunspot images using cross-correlation.")
    parser.add_argument('in_folder', type=str)
    parser.add_argument('out_folder', type=str)
    parser.add_argument('jpg_folder', type=str)
    parser.add_argument('movie_name', type=str)
    parser.add_argument('--num_images', type=int, default=100, help="Number of images to process (default: 100)")
    parser.add_argument('--start_time', type=str, default=None, help="Optional start time in YYYY-MM-DDTHH:MM:SS format")
    parser.add_argument('--end_time', type=str, default=None, help="Optional end time in YYYY-MM-DDTHH:MM:SS format")
    parser.add_argument('--filter', type=str, default='NB03')
    parser.add_argument('--ref_idx', type=int, default=0)
    parser.add_argument('--save_fits', action='store_true')
    parser.add_argument('--method', type=str, default='cross-correlation', choices=['cross-correlation'])
    args = parser.parse_args()

    for folder in [args.out_folder, args.jpg_folder]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    roi_co_align(
        in_folder=args.in_folder,
        out_folder=args.out_folder,
        jpg_folder=args.jpg_folder,
        mp4_name=args.movie_name,
        num_images=args.num_images,
        start_time=args.start_time,
        end_time=args.end_time,
        filter=args.filter,
        ref_idx=args.ref_idx,
        save_fits=args.save_fits,
        method=args.method
    )

if __name__ == "__main__":
    main()
