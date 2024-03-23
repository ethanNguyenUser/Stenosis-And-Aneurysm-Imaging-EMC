import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from dcimg import DCIMGFile
from tkinter import filedialog
import tkinter as tk
import tifffile as tiff
from skimage.measure import block_reduce
import subprocess
from threading import Thread



def read_all_dcimg_in_folder(folder_path, binning_factor):
    # Search for all .dcimg files in the folder and its subfolders
    search_pattern = os.path.join(folder_path, '**/*.dcimg')
    dcimg_files = sorted(glob.glob(search_pattern, recursive=True))

    if not dcimg_files:
        print("No DCIMG files found in the specified folder.")
        return None

    # Read the first file to determine the size of each stack
    first_dcimg_obj = DCIMGFile(dcimg_files[0])
    first_stack = np.stack([frame for frame in first_dcimg_obj], axis=0)

    # Apply binning to the first stack
    binned_first_stack = block_reduce(first_stack, block_size=binning_factor, func=np.mean)

    # Number of stacks is the number of files
    num_stacks = len(dcimg_files)

    # Preallocate the 4D array with binned dimensions
    stack_4d = np.empty((binned_first_stack.shape[0], binned_first_stack.shape[1], binned_first_stack.shape[2], num_stacks), dtype=np.uint16)
    #print(stack_4d)

    # Load the first binned stack
    stack_4d[..., 0] = binned_first_stack.astype(np.uint16)
    #print("shape of stack 4d before loop" , stack_4d.shape)
    

    # Load the remaining stacks with binning
    for i, file_path in enumerate(dcimg_files[1:], start=1):
        print(f'Reading in file: {i} of {num_stacks}, ' + file_path)
        dcimg_obj = DCIMGFile(file_path)
        stack = np.stack([frame for frame in dcimg_obj], axis=0)
        binned_stack = block_reduce(stack, block_size=binning_factor, func=np.mean).astype(np.uint16)
        stack_4d[..., i] = binned_stack
    
    
    #print("shape of stack 4d after loop", stack_4d.shape)

    return stack_4d

def select_folder_and_read_dcimg(binning_factor):
    # Set up a root window for the folder dialog (not displayed)
    root = tk.Tk()
    root.withdraw()

    # Open a folder dialog to select the folder
    folder_path = filedialog.askdirectory()

    if folder_path:
        return folder_path, read_all_dcimg_in_folder(folder_path, binning_factor)
    else:
        print("No folder selected.")
        return []
    


def generate_stack_order(num_x_steps, num_y_steps):
    stack_order = []
    for y in range(num_y_steps):
        row = list(range(y * num_x_steps, (y + 1) * num_x_steps))
        if y % 2 == 1:
            row.reverse()
        stack_order.append(row)
    return stack_order



def get_user_input():
    global num_x_steps, num_y_steps, num_z_slices
    root = tk.Tk()
    root.withdraw()

    input_window = tk.Toplevel(root)
    input_window.title("Input Parameters")
    input_window.geometry("400x200")

    tk.Label(input_window, text="num_x_step:").grid(row=0)
    tk.Label(input_window, text="num_y_step:").grid(row=1)
    tk.Label(input_window, text="num_z_slices:").grid(row=2)

    num_x_step_entry = tk.Entry(input_window)
    num_y_step_entry = tk.Entry(input_window)
    num_z_slices_entry = tk.Entry(input_window)

    num_x_step_entry.grid(row=0, column=1)
    num_y_step_entry.grid(row=1, column=1)
    num_z_slices_entry.grid(row=2, column=1)

    def submit():
        global num_x_steps, num_y_steps, num_z_slices
        try:
            num_x_steps = int(num_x_step_entry.get())
            num_y_steps = int(num_y_step_entry.get())
            num_z_slices = int(num_z_slices_entry.get())
            input_window.destroy()  # Close the window after submitting
        except ValueError:
            print("Please enter valid integers for all fields.")

    submit_button = tk.Button(input_window, text="Submit", command=submit)
    submit_button.grid(row=3, columnspan=2)

    input_window.wait_window()  # Wait here until the window is destroyed


    
def apply_ashlar_to_files(input_dir, output_dir):
    file_list = os.listdir(input_dir)
    num_files = len(file_list)
    for i in range(num_files):
        input_path = os.path.join(input_dir, f"reconstructed_set_{i}.ome.tif")
        output_path = os.path.join(output_dir, f"240207sheet_new_{i}_ashlar.ome.tif")
        # Call ashlar command using subprocess
        subprocess.run(["ashlar", "--output", output_path, input_path])

def process_files(files, input_dir, output_dir):
    # Process each file in the chunk
    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"240207sheet_new_{file_name}_ashlar.ome.tif")
        subprocess.run(["ashlar", "--output", output_path, input_path])

def apply_ashlar_with_threading(input_dir, output_dir):
    # Create a list to hold the threads
    threads = []
    num_threads = 5  # Set the maximum number of threads
    file_list= os.listdir(input_dir)
    chunk_size =len(file_list)//num_threads
    file_chunks = [file_list[i:i+chunk_size] for i in range(0, len(file_list), chunk_size)]

    for chunk in file_chunks:
        thread = Thread(target=process_files, args=(chunk, input_dir, output_dir))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


binning_factor=(1,2, 2)
get_user_input() 

folder_path, stack_4d = select_folder_and_read_dcimg(binning_factor)
 

if stack_4d is not None:
    print("Shape of the 4D array:", stack_4d.shape)
else:
    print("No data to display.")

stack_dim = stack_4d.shape 

# Calculate the number of sets in the 4D stack
stacks_per_set = num_x_steps * num_y_steps
num_sets = stack_4d.shape[3] // stacks_per_set

stack_order = generate_stack_order(num_x_steps, num_y_steps)
stack_order = np.array(stack_order)
stack_order= np.flipud(np.transpose(stack_order))

# Extract the name of the selected folder to use as the output folder name
output_folder = f"{folder_path}/mosaic_tiles"
os.makedirs(output_folder, exist_ok=True)

z = stack_4d.shape[0]
y = stack_4d.shape[1]
x = stack_4d.shape[2]
pixel_size = 0.65

flattened_stack_order = stack_order.flatten()
image_set = np.squeeze(stack_4d[0, :, :, flattened_stack_order])
print(stack_order, np.shape(stack_order), flattened_stack_order, np.shape(image_set))

counter = 0
#ii is "Big Z" or the total number of Z steps (stage movement)
#jj is "little z" the number images in volume
for ii in range(num_sets):
    for jj in range(z):
        # Extract the image set while removing any singleton dimensions
        indices_extracted = flattened_stack_order + stacks_per_set * ii

        # Initialize an empty array for the rearranged image set
        rearranged_image_set = np.empty_like(image_set)

        # Rearrange the image set according to indices_extracted
        for idx, value in enumerate(indices_extracted):
            rearranged_image_set[idx, ...] = np.squeeze(stack_4d[jj, :, :, value])

        #print(indices_extracted, rearranged_image_set.shape)

        # Save the rearranged stack as a multi-page TIFF in the same folder
        output_filename = os.path.join(output_folder, f"reconstructed_set_{counter}.ome.tif")

        #tiff.imwrite(output_filename, rearranged_image_set)

        positions = np.array([np.unravel_index(i, (num_x_steps, num_y_steps)) for i in range(image_set.shape[0])])

        with tiff.TiffWriter(output_filename, bigtiff=True) as tif:
            for idx, (img, p) in enumerate(zip(rearranged_image_set, positions)):
                metadata = {
                    'Pixels': {
                        'PhysicalSizeX': pixel_size,
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': pixel_size,
                        'PhysicalSizeYUnit': 'µm'
                        },
                     'Plane': {
                        'PositionX': p[1] * pixel_size * 1538 / binning_factor[1],
                        'PositionY': p[0] * pixel_size * 1538 / binning_factor[2]
                        }
                    }
                tif.write(img, metadata=metadata)
        print(f"File saved with meta data: {output_filename}")
        counter += 1



    

# Set input and output directories
output_folder = f"{folder_path}/mosaic_tiles"
input_directory = output_folder 
output_directory = f"{folder_path}/ashlar_stacks"

# Ensure output directory exists, create if it doesn't
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Iterate over files and apply ashlar function
apply_ashlar_with_threading(input_directory, output_directory)