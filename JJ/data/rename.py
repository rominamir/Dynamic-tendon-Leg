import os

for folder in os.listdir('.'):
    if os.path.isdir(folder):
        new_name = folder.replace(':', '')
        if new_name != folder:
            os.rename(folder, new_name)
