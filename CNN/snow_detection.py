import numpy as np
from PIL import Image
import os
import shutil
from pathlib import Path
import logging

def determine_snowiness(image_path, thresholds):
    """Maintain same classification structure but with improved accuracy"""
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img)
            
            # Keep original classification logic to maintain compatibility
            # but with improved processing
            if img_array.max() > 255:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
            
            num_dark_pixels = np.sum(img_array <= 120)
            proportion_dark_pixels = num_dark_pixels / img_array.size
            
            # Use same thresholds as before
            if proportion_dark_pixels >= thresholds['very_snowy_threshold']:
                return 'very_snowy'
            elif proportion_dark_pixels >= thresholds['snowy_threshold']:
                return 'snowy'
            elif proportion_dark_pixels >= thresholds['moderately_snowy_threshold']:
                return 'moderately_snowy'
            elif proportion_dark_pixels >= thresholds['some_snow_threshold']:
                return 'some_snow'
            else:
                return 'little_to_no_snow'
                
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None

def organize_images(source_directory, mostly_snowy_imgs, very_snowy_imgs, 
                   moderately_snowy_imgs, little_snow_imgs, little_to_no_snow_imgs):
    """Organize images using same folder structure as before"""
    
    # Your original thresholds
    snw_lvl_thresholds = {
        'very_snowy_threshold': .085,
        'snowy_threshold': .07,
        'moderately_snowy_threshold': .05,
        'some_snow_threshold': .025,
        'little_to_no_snow_threshold': .005
    }

    # Create directories if they don't exist
    for directory in [mostly_snowy_imgs, very_snowy_imgs, moderately_snowy_imgs, 
                     little_snow_imgs, little_to_no_snow_imgs]:
        os.makedirs(directory, exist_ok=True)

    processed = 0
    for filename in os.listdir(source_directory):
        if filename.endswith('.tif'):
            source_path = os.path.join(source_directory, filename)
            snow_cover_level = determine_snowiness(source_path, snw_lvl_thresholds)
            
            if snow_cover_level == 'very_snowy':
                target_path = os.path.join(mostly_snowy_imgs, filename)
            elif snow_cover_level == 'snowy':
                target_path = os.path.join(very_snowy_imgs, filename)
            elif snow_cover_level == 'moderately_snowy':
                target_path = os.path.join(moderately_snowy_imgs, filename)
            elif snow_cover_level == 'some_snow':
                target_path = os.path.join(little_snow_imgs, filename)
            else:
                target_path = os.path.join(little_to_no_snow_imgs, filename)
                
            shutil.copy(source_path, target_path)
            processed += 1
            
            if processed % 100 == 0:
                print(f"Processed {processed} images")

    return processed

# Use with your original paths
if __name__ == "__main__":
    source_directory = '/Volumes/WWWTfilms-1/MODIS_NH_IMGS/COOS_Composites'
    mostly_snowy_imgs = '/Volumes/WWWTfilms-1/MODIS_NH_IMGS/COOS_Composites/very_snowy_imgs'
    very_snowy_imgs = '/Volumes/WWWTfilms-1/MODIS_NH_IMGS/COOS_Composites/pretty_snowy_imgs'
    moderately_snowy_imgs = '/Volumes/WWWTfilms-1/MODIS_NH_IMGS/COOS_Composites/moderately_snowy_imgs'
    little_snow_imgs = '/Volumes/WWWTfilms-1/MODIS_NH_IMGS/COOS_Composites/some_snow_imgs'
    little_to_no_snow_imgs = '/Volumes/WWWTfilms-1/MODIS_NH_IMGS/COOS_Composites/little_to_no_snow_imgs'

    total_processed = organize_images(
        source_directory,
        mostly_snowy_imgs,
        very_snowy_imgs,
        moderately_snowy_imgs,
        little_snow_imgs,
        little_to_no_snow_imgs
    )
    
    print(f"Finished processing {total_processed} images")