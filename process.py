import os
import argparse
import numpy as np
import pandas as pd
import skimage

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Convert folder with .tif files to .png",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data',
        default='.'+os.sep+'epidermal_data_model'+os.sep,
        help='Location where the dataset is stored.')

    return parser.parse_args()


def main():
    """ Main function. """

    yourpath = ARGS.data
    count = 0
    for root, dirs, files in os.walk(yourpath, topdown=False):
        for name in files:
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
                if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                    print("A jpg file already exists for ", name)
                # If a jpeg is *NOT* present, create one from the tiff.
                else:
                    outfile = "processed_images/" + str(count) + ".jpg"
                    outfile_blurred = "processed_images/" + str(count) + "-blurred.jpg"
                    try:
                        im = np.array(skimage.io.imread(os.path.join(root, name)))
                        print("Generating jpg for ", name)
                        print(outfile)
                        im = skimage.transform.resize(im, (256, 256), anti_aliasing=True)
                        im_blurred = skimage.filters.gaussian(im, sigma=1)
                        skimage.io.imsave(outfile, im)
                        skimage.io.imsave(outfile_blurred, im_blurred)
                        count+=1
                    except Exception as e:
                        print(e)

# Make arguments global
ARGS = parse_args()

main()
