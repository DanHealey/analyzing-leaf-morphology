import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

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
                    outfile = os.path.splitext(os.path.join(root, str(count)))[0] + ".jpg"
                    try:
                        im = np.array(Image.open())
                        print("Generating jpg for ", name)
                        print(outfile)
                        im = im / np.amax(im) * 255
                        im = Image.fromarray(im.astype(np.uint8))
                        im.convert("L")
                        im.thumbnail(im.size)
                        im.save(outfile)
                        count+=1
                    except Exception as e:
                        print(e)

# Make arguments global
ARGS = parse_args()

main()
