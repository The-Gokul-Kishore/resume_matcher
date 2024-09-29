

import os
import shutil

# Define the source folder (the folder containing the subfolders)
source_folder = 'D:\\mlprojt\\resume_dicson\\data\\data'# Change this to the actual path

if os.path.exists(source_folder):
    if os.path.isdir(source_folder):
        print(f"The source folder exists and is a directory: {source_folder}")
    else:
        print("not a directory")
else:
    print("source path is wrong")
# Define the destination folder (where all PDFs will be moved)
destination_folder = 'D:\\mlprojt\\resume_dicson\\static'  # Change this to the actual path

# Make sure the destination folder exists, if not, create it
if os.path.exists(destination_folder):
    if os.path.isdir(destination_folder):
        print("it is a valid path and a dir")
    else:
        print("not a directory")
else:
    print("path incorrect")

# Traverse the source folder and all its subfolders
for root, dirs, files in os.walk(source_folder):
    print(f"Checking directory: {root}")  # Debug: See which directory is being checked
    for file in files:
        if file.endswith('.pdf'):  # Check if the file is a PDF
            file_path = os.path.join(root, file)  # Get the full path of the file
            print(f"Found PDF: {file_path}")  # Debug: Print the PDF file found
            try:
                # Move the file to the destination folder
                shutil.move(file_path, os.path.join(destination_folder, file))
                print(f"Moved: {file_path} to {destination_folder}")  # Debug: Confirm the move
            except Exception as e:
                print(f"Failed to move {file}: {e}")  # Debug: If there's an issue, print the error
        else:
            print("not a pdf.")

print("All PDF files have been processed.")