from PIL import Image
import os
import tifffile
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import cv2

import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tifffile
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_weekly_composite(images, method='weighted'):
    """
    Create a composite image from a list of weekly images.
    
    Args:
        images: List of numpy arrays representing images
        method: 'weighted' or 'mean' compositing method
    """
    if not images:
        return None
        
    if method == 'weighted':
        composite = images[0].copy()
        for i in range(1, len(images)):
            alpha = 1.0 / (i + 1)
            composite = cv2.addWeighted(composite, 1 - alpha, images[i], alpha, 0)
    else:  # mean method
        composite = np.mean(images, axis=0).astype(np.uint8)
        
    return composite

def process_weekly_composites(input_dir, output_dir, days_interval=7):
    """
    Process NDSI images into weekly composites.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get and sort all tif files
    tif_files = sorted(list(input_path.glob("*.tif")))
    
    current_week = []
    current_week_start = None
    
    for tif_file in tif_files:
        # Extract date from filename
        parts = tif_file.stem.split("_doy")[1].split("_")
        year = int(parts[0][:4])
        doy = int(parts[0][4:])
        date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        
        if current_week_start is None:
            current_week_start = date
            
        # Check if we should create a composite
        if (date - current_week_start).days >= days_interval:
            if current_week:
                # Create and save composite
                composite = create_weekly_composite(current_week)
                if composite is not None:
                    output_file = output_path / f"composite_{current_week_start.strftime('%Y%m%d')}.tif"
                    tifffile.imwrite(output_file, composite)
                    logging.info(f"Created composite for week starting {current_week_start.strftime('%Y%m%d')}")
                
            # Reset for next week
            current_week = []
            current_week_start = date
            
        # Add current image to week
        try:
            img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
            if img is not None:
                current_week.append(img)
            else:
                logging.warning(f"Could not read image: {tif_file}")
        except Exception as e:
            logging.error(f"Error processing {tif_file}: {str(e)}")
            
    # Process final week if any images remain
    if current_week:
        composite = create_weekly_composite(current_week)
        if composite is not None:
            output_file = output_path / f"composite_{current_week_start.strftime('%Y%m%d')}.tif"
            tifffile.imwrite(output_file, composite)

if __name__ == "__main__":
    setup_logging()
    
    INPUT_DIR = "/Volumes/WWWTfilms-1/MODIS_NH_IMGS/COOS_2"
    OUTPUT_DIR = "/Users/briannamarcosano/Documents/SF_Testing_code/CNN_LSTM_Thesis/CNN/Datasets/COOS_Composites_December03_2024"
    
    process_weekly_composites(INPUT_DIR, OUTPUT_DIR)