from PIL import Image
import numpy as np
import os
import argparse


# from: https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array
def load_image(file):
    img = Image.open(file)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data


def main(args):
    subdirs = []
    if args.train_dir:
        subdirs.append("train")
    if args.test_dir:
        subdirs.append("test")
    if args.val_dir:
        subdirs.append("val")

    R = []
    G = []
    B = []

    for subdir in subdirs:
        main_path = os.path.join(args.root_dir, subdir)

        for root, dirs, files in os.walk(main_path):
            for file in files:
                # extract images
                if file.split(".")[-1] == "jpg":
                    image = load_image(os.path.join(main_path, root, file))

                    for x in image:
                        for pixel in x:
                            try:
                                if len(image.shape) == 3:
                                    R.append(pixel[0])
                                    G.append(pixel[1])
                                    B.append(pixel[2])

                                # grayscale
                                elif len(image.shape) == 2:
                                    R.append(pixel)
                                    G.append(pixel)
                                    B.append(pixel)
                            except:  # needed because images have weird sizes
                                print(os.path.join(main_path, root, file))
                                print(image.shape)

    return ([np.mean(np.array(R)), np.mean(np.array(G)), np.mean(np.array(B))],
            [np.std(np.array(R)), np.std(np.array(G)), np.std(np.array(B))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("get mean and stdev of (image) dataset",
                                     add_help=True)
    parser.add_argument("--root-dir", type=str)

    parser.add_argument("--train-dir", action="store_true", default=True)
    parser.add_argument("--val-dir", action="store_true", default=True)
    parser.add_argument("--test-dir", action="store_true", default=False)

    args = parser.parse_args()

    (mean, stdev) = main(args)

    print("mean:")
    print(mean)
    print("stdev:")
    print(stdev)
