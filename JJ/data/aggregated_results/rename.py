import os


root_dir = './'  # 或者 '/path/to/your/folders/'

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if ':' in filename:
            old_path = os.path.join(dirpath, filename)
            new_filename = filename.replace(':', '_')
            new_path = os.path.join(dirpath, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

print("All done!")
