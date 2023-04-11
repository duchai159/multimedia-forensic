import os
import pandas as pd
import exifread

# Define the directory path where the images are stored
directory_path = "/home/hai/Downloads/Image_Manipulation_Detection_System_Python/input/CFA-Artifact Detection/"

# Get the list of image files in the directory
files = [file for file in os.listdir(directory_path) if file.endswith(".jpg")]

# Initialize an empty list to store the metadata of each file
metadata_list = []

# Loop through each file and extract its metadata using exifread
for file in files:
    # Open the file in binary mode
    with open(os.path.join(directory_path, file), "rb") as f:
        # Read the metadata using exifread
        metadata = exifread.process_file(f)
        # Append the metadata to the list
        metadata_list.append(metadata)

# Convert the list of metadata dictionaries into a pandas dataframe
metadata_df = pd.DataFrame(metadata_list)

# Drop any columns with all missing values
metadata_df = metadata_df.dropna(axis=1, how="all")

# Display the first few rows of the metadata dataframe
print(metadata_df.head())

# Get the number of unique camera makes in the metadata
num_unique_camera_makes = metadata_df["Image Make"].nunique()
print("Number of unique camera makes:", num_unique_camera_makes)

# Group the metadata by camera make and get the count of each make
camera_make_counts = metadata_df.groupby("Image Make").size()
print("Camera make counts:\n", camera_make_counts)

# Get the summary statistics of the image height and width
image_size_summary = metadata_df[["EXIF ExifImageLength", "EXIF ExifImageWidth"]].describe()
print("Image size summary statistics:\n", image_size_summary)
