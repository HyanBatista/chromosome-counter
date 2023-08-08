import cv2
from preprocessing import binarize
from matplotlib import pyplot as plt


def main():
    image = cv2.imread('/home/hyanbatista42/workspaces/chromosome-counter/data/41.tif')
    plt.figure(figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(cv2.cvtColor())

if __name__ == '__main__':
    main()
