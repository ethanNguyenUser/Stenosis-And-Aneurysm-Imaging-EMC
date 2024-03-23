import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from dcimg import DCIMGFile
from tkinter import filedialog
from tkinter import Tk
import tifffile as tiff
# import cv2

def select_and_read_dcimg():
    # Set up a root window for the file dialog (not displayed)
    root = Tk()
    root.withdraw()

    # Open a file dialog to select the DCIMG file
    file_path = filedialog.askopenfilename(filetypes=[("DCIMG files", "*.dcimg")])

    if file_path:
        # Read the DCIMG file
        dcimg_obj = DCIMGFile(file_path)
        
        # Assuming dcimg_obj can be iterated to get individual frames
        frames = [frame for frame in dcimg_obj]

        # Convert list of frames to 3D NumPy array
        data = np.stack(frames, axis=0)

        return data
    else:
        print("No file selected.")
        return None

def select_folder_and_read_dcimg():
    # Set up a root window for the folder dialog (not displayed)
    root = Tk()
    root.withdraw()

    # Open a folder dialog to select the folder
    folder_path = filedialog.askdirectory()

    if folder_path:
        return folder_path, read_all_dcimg_in_folder(folder_path)
    else:
        print("No folder selected.")
        return []

def read_all_dcimg_in_folder(folder_path):
    # Search for all .dcimg files in the folder and its subfolders
    search_pattern = os.path.join(folder_path, '**/*.dcimg')
    dcimg_files = sorted(glob.glob(search_pattern, recursive=True))

    if not dcimg_files:
        print("No DCIMG files found in the specified folder.")
        return None

    # Read the first file to determine the size of each stack
    first_dcimg_obj = DCIMGFile(dcimg_files[0])
    first_stack = np.stack([frame for frame in first_dcimg_obj], axis=0)

    # Number of stacks is the number of files
    num_stacks = len(dcimg_files)

    # Preallocate the 4D array
    stack_4d = np.empty((first_stack.shape[0], first_stack.shape[1], first_stack.shape[2], num_stacks), dtype=first_stack.dtype)

    # Load the first stack
    stack_4d[..., 0] = first_stack

    # Load the remaining stacks
    for i, file_path in enumerate(dcimg_files[1:], start=1):
        print(f'Reading in file: {i} of {num_stacks}, ' + file_path)
        dcimg_obj = DCIMGFile(file_path)
        stack = np.stack([frame for frame in dcimg_obj], axis=0)
        stack_4d[..., i] = stack
    

    return stack_4d

def generate_stack_order(num_x_steps, num_y_steps):
    stack_order = []
    for y in range(num_y_steps):
        row = list(range(y * num_x_steps, (y + 1) * num_x_steps))
        if y % 2 == 1:
            row.reverse()
        stack_order.append(row)
    return stack_order

# Usage example
folder_path, stack_4d = select_folder_and_read_dcimg()
if stack_4d is not None:
    print("Shape of the 4D array:", stack_4d.shape)
    print("Folder path: " + folder_path)
else:
    print("No data to display.")

stack_dim = stack_4d.shape  # Use shape instead of np.size
print(stack_dim)

# z_start = 105
num_z_slices = 100

stage_distance = 1000
pixels_distance = int(stage_distance / (6.5 / 10))  # Corrected variable name and conversion to int
pixels_half_distance = pixels_distance // 2
pixels_distance = pixels_half_distance * 2

# Define the order of stack indices for XY plane
num_x_steps = 15  # Number of steps in the X direction defined as up down
num_y_steps = 5  # Number of steps in the Y direction defined as left right

# Calculate the number of sets in the 4D stack
stacks_per_set = num_x_steps * num_y_steps
num_sets = stack_4d.shape[3] // stacks_per_set

stack_order = np.transpose(generate_stack_order(num_x_steps, num_y_steps))

# Extract the name of the selected folder to use as the output folder name
output_folder = f"{folder_path}/rough_mosaic_stacks"
os.makedirs(output_folder, exist_ok=True)


for set_index in range(num_sets):
    # Create a reconstructed stack
    reconstructed_stack = np.zeros((num_z_slices, pixels_distance * num_x_steps, pixels_distance * num_y_steps), dtype=np.uint16)

    for jj in range(num_y_steps):
        for ii in range(num_x_steps):
            stack_idx = set_index * stacks_per_set + stack_order[jj][ii]
            print(stack_idx)
            sub_stack = stack_4d[:,
                                stack_dim[1] // 2 - pixels_half_distance:stack_dim[1] // 2 + pixels_half_distance,
                                stack_dim[2] // 2 - pixels_half_distance:stack_dim[2] // 2 + pixels_half_distance,
                                stack_idx]
            reconstructed_stack[:, 
                                ii * pixels_distance:(ii + 1) * pixels_distance,
                                jj * pixels_distance:(jj + 1) * pixels_distance] = sub_stack
            

    # Save the reconstructed stack as a multi-page TIFF in the same folder
    output_filename = os.path.join(output_folder, f"reconstructed_set_{set_index + 1}.tif")
    tiff.imwrite(output_filename, reconstructed_stack, photometric='minisblack')

    print(f"Multi-page TIFF for set {set_index + 1} saved in: {output_filename}")
