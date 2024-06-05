import os

def rename_files(folder_path, old_char, new_char):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Construct the old and new filenames
        old_filename = os.path.join(folder_path, filename)
        new_filename = os.path.join(folder_path, filename.replace(old_char, new_char))
        
        # Rename the file
        os.rename(old_filename, new_filename)
        print(f"Renamed '{old_filename}' to '{new_filename}'")

# Replace 'old_char' with 'new_char' in all files in the folder
folder_path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split left fast S6\\'
old_char = ','
new_char = '.'
rename_files(folder_path, old_char, new_char)