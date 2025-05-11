import os

# Set the target directory
directory = "/Users/masha/Documents/diplom/Waste-or-Garbage-Classification-Using-Deep-Learning/DataSets/Train/trash"

# Set the base name for renaming
base_name = "trash"

# Get all image files in the directory
image_files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

# Rename files
for count, filename in enumerate(image_files, start=1):
    old_path = os.path.join(directory, filename)
    
    # Get the file extension
    file_ext = os.path.splitext(filename)[1]
    
    # Set the new file name
    new_filename = f"{base_name}{count}{file_ext}"
    new_path = os.path.join(directory, new_filename)

    # Rename the file
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} -> {new_filename}")

print("Renaming complete!")

