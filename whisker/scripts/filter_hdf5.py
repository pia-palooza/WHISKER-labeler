import os
import pandas as pd
import logging

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"D:\workspace\whisker-workspace\datasets\PJATrain500frames\PJATrain500frames"

# Inputs
COMBINED_H5_PATH = os.path.join(BASE_DIR, "manifest.h5")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")

# Outputs
TRAIN_H5_OUT = os.path.join(BASE_DIR, "train_manifest.h5")
TEST_H5_OUT = os.path.join(BASE_DIR, "test_manifest.h5")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_whisker_h5(df, output_path, split_name):
    """Saves the DataFrame and recreates the necessary WHISKER metadata."""
    if df.empty:
        logging.warning(f"No data found for {split_name} split. Skipping HDF5 generation.")
        return

    logging.info(f"Saving {split_name} manifest to {output_path} (Frames: {len(df.index.get_level_values('frame_index').unique())})")
   
    # 1. Save the main table
    df.to_hdf(output_path, key='keypoints', format='table', mode='w')
   
    # 2. Extract and save metadata series
    body_parts = pd.Series(df.index.get_level_values('body_part').unique())
    individuals = pd.Series(df.index.get_level_values('individual_id').unique())
   
    body_parts.to_hdf(output_path, key='metadata/body_parts', mode='a')
    individuals.to_hdf(output_path, key='metadata/individuals', mode='a')
   
    logging.info(f"Successfully saved {split_name} HDF5.")

def split_hdf5_dataset():
    # 1. Load the combined master dataset
    logging.info(f"Loading master HDF5 file from: {COMBINED_H5_PATH}")
    if not os.path.exists(COMBINED_H5_PATH):
        logging.error("Combined HDF5 file not found. Please check the path.")
        return

    try:
        master_df = pd.read_hdf(COMBINED_H5_PATH, key='keypoints')
    except Exception as e:
        logging.error(f"Failed to read HDF5: {e}")
        return

    # 2. Read image filenames from the physical directories
    # Ignoring hidden/system files just in case
    train_files = [f for f in os.listdir(TRAIN_IMG_DIR) if os.path.isfile(os.path.join(TRAIN_IMG_DIR, f))]
    test_files = [f for f in os.listdir(TEST_IMG_DIR) if os.path.isfile(os.path.join(TEST_IMG_DIR, f))]
   
    logging.info(f"Found {len(train_files)} images in Train directory.")
    logging.info(f"Found {len(test_files)} images in Test directory.")

    # 3. Safely intersect directory files with the DataFrame index to avoid KeyErrors
    df_known_frames = master_df.index.get_level_values('frame_index').unique()
   
    valid_train_files = [f for f in train_files if f in df_known_frames]
    valid_test_files = [f for f in test_files if f in df_known_frames]

    # 4. Slice the master dataframe using .loc
    # Because 'frame_index' is the top level of the MultiIndex, we can pass the list of files directly
    logging.info("Slicing DataFrames...")
    train_df = master_df.loc[valid_train_files].copy()
    test_df = master_df.loc[valid_test_files].copy()

    # 5. Write the outputs
    save_whisker_h5(train_df, TRAIN_H5_OUT, "Train")
    save_whisker_h5(test_df, TEST_H5_OUT, "Test")

if __name__ == "__main__":
    split_hdf5_dataset()
    logging.info("Splitting process complete!")