

# Do imports
import cv2
import numpy as np


def show(na_img, s_window_title='Image Window'):

    # Show
    cv2.imshow(s_window_title, na_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_crop_metrics(na_img):

    # Get details
    i_num_rows = na_img.shape[0]
    i_num_cols = na_img.shape[1]

    # Get section
    i_p38_row = int(round(i_num_rows * 0.38))
    i_p62_row = int(round(i_num_rows * 0.62))
    i_p25_col = int(round(i_num_cols * 0.25))
    i_p75_col = int(round(i_num_cols * 0.75))

    # Return
    return i_p38_row, i_p62_row, i_p25_col, i_p75_col


def vis_cropping(na_img, b_show=True):

    # Get crop metrics
    i_top_row, i_bottom_row, i_left_col, i_right_col = get_crop_metrics(na_img)

    # Add crop visualization
    na_img_w_crop = np.copy(na_img)
    na_img_w_crop[i_top_row, i_left_col: i_right_col, :] = (0, 0, 255)
    na_img_w_crop[i_bottom_row, i_left_col: i_right_col, :] = (0, 0, 255)
    na_img_w_crop[i_top_row: i_bottom_row, i_left_col, :] = (0, 0, 255)
    na_img_w_crop[i_top_row: i_bottom_row, i_right_col, :] = (0, 0, 255)

    # Show
    show(na_img_w_crop, 'Image w/ Crop')

    # Return
    return na_img_w_crop


def get_crop(na_img):

    # Get crop metrics
    i_top_row, i_bottom_row, i_left_col, i_right_col = get_crop_metrics(na_img)

    # Get crop
    na_img_crop = na_img[i_top_row: i_bottom_row, i_left_col: i_right_col, :]

    # Return
    return na_img_crop


def get_pixel_samples(na_img, i_pixel_samples_per_img):

    # Sample
    na_pixels = na_img.reshape(-1, 3)
    na_sample_idxs = np.random.choice(na_pixels.shape[0], size=i_pixel_samples_per_img, replace=False)
    na_pixel_samples = na_pixels[na_sample_idxs]

    # Return
    return na_pixel_samples


