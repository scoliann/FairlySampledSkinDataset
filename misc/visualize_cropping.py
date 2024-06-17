

# Do imports
import os
import cv2
import sys
import numpy as np


# Do local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helper_functions as hf


def main():

    # Define key variables
    s_img_dir = 'images_to_crop'

    # Get image files
    ls_img_files = [os.path.join(s_img_dir, f) for f in os.listdir(s_img_dir)]

    # Test
    for s_img_file in ls_img_files:

        # Read in image
        na_img = cv2.imread(s_img_file)

        # Visualize
        na_img_w_crop = hf.vis_cropping(na_img, b_show=True)

        # Save
        cv2.imwrite(s_img_file.replace('.jpg', '_cropped.jpg'), na_img_w_crop)


if __name__ == '__main__':
    main()
