import numpy as np
import cv2
import argparse

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Define the list of boundaries
    boundaries = [
        # ([17, 15, 100], [50, 56, 200]),
        # ([86, 31, 4], [220, 88, 50]),
        # ([25, 146, 190], [62, 174, 250]),
        # ([103, 86, 65], [145, 133, 128]),
        ([0, 0, 0], [50, 50, 255]),  # Red
        ([0, 0, 0], [255, 50, 50]),  # Blue
        ([0, 100, 0], [100, 255, 100]),  # Green
        ([120, 0, 120], [255, 50, 255]),  # Pink
        ([0, 120, 120], [50, 255, 255]),  # Yellow
    ]

    # Loop over the boundaries
    for (lower, upper) in boundaries:
        # Create NumPy arrays from the boundaries
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        # Find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # Show the images
        cv2.imshow("image", np.hstack([image, output]))
        cv2.waitKey(0)

if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    args = vars(ap.parse_args())

    # Process the image
    process_image(args["image"])
