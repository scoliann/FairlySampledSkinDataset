

# Do imports
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# Do local imports
import helper_functions as hf


def main():

    # Define key variables
    s_train_file = 'fairface/fairface_label_train.csv'
    s_val_file = 'fairface/fairface_label_val.csv'
    s_fairface_img_dir = 'fairface/fairface-img-margin025-trainval'
    i_img_samples = 10000
    i_pixel_samples_per_img = 100

    # Set random seed
    np.random.seed(0)

    # Read in dataframes
    df_train = pd.read_csv(s_train_file)
    df_val = pd.read_csv(s_val_file)

    # Combine
    df = pd.concat([df_train, df_val])

    # Determine number of images to sample per ethnic group
    d_ethnicity_count = {}
    for s_ethnicity in df['race'].unique():
        i_count = (df['race'] == s_ethnicity).sum()
        d_ethnicity_count[s_ethnicity] = i_count
    i_num_sample_imgs_in_smallest_group = min(list(d_ethnicity_count.values()))
    print(f'\n\nEach ethnic group has at least {i_num_sample_imgs_in_smallest_group} images, {i_img_samples} from each group will be used for building the dataset\n\n')

    # Randomize dataframe and keep specified number of samples for each ethnic group
    df = df.sample(frac=1, random_state=0)
    df = df.groupby('race').head(i_img_samples).reset_index(drop=True)

    # Get pixel samples
    tqdm.pandas(desc='Sampling Pixels')
    def get_pixel_samples_by_file_path(s_file, i_pixel_samples_per_img):
        na_img = cv2.imread(s_file)
        na_img_crop = hf.get_crop(na_img)
        na_pixel_samples = hf.get_pixel_samples(na_img, i_pixel_samples_per_img)
        ls_pixel_samples = '\t'.join([','.join(map(str, na_pixel_sample)) for na_pixel_sample in na_pixel_samples])
        return ls_pixel_samples
    df['bgr_pixel_samples'] = df['file'].progress_apply(lambda s_file: get_pixel_samples_by_file_path(os.path.join(s_fairface_img_dir, s_file), i_pixel_samples_per_img))
    
    # Randomize rows and save
    df = df.sample(frac=1, random_state=0)
    df.to_csv('fairly_sampled_skin_pixels.csv', index=False)


if __name__ == '__main__':
    main()
