from PIL import Image
import os
import numpy as np

folder='./dataset/'
save_folder='./rotated_dataset/'
# List all files in the folder
files=os.listdir(folder)

# Loop through all files, if not shape (512, 768, 3) then rotate.
for file in files:
    img=Image.open(os.path.join(folder, file))
    if np.shape(img)!=(512, 768, 3):
        img=img.rotate(90,expand=True)
        img.save(save_folder+file)
        print('Rotated: ', file)
    else:
        img.save(save_folder+file)
        print('No need to rotate: ', file)