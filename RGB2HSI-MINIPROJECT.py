import math
from matplotlib import pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm


def RGB2HSI(R, G, B):
    H = np.zeros(R.shape)
    S = np.zeros(R.shape)
    I = np.zeros(R.shape)

    eps = 1E-6

    for x in tqdm(range(R.shape[0])):
        for y in range(R.shape[1]):
            r = R[x, y]
            b = B[x, y]
            g = G[x, y]

            numer = float(0.5 * ((r - g) + (r - b)))
            denom = float(((r - g) ** 2 + ((r - b) * (g - b))) ** 0.5)  # when uplifting you can use ** as well!
            if (b <= g):
                h = math.acos(numer / (denom + eps))  # to not divide with 0  !
            if (b > g):
                h = (2 * math.pi) - math.acos(numer / (denom + eps))
                # h=h/360

            minimum = np.min([r, g, b])
            s = 1 - 3 * np.divide(minimum, (r + g + b) + eps)

            i = float(r + g + b) / float(3)

            pix_h = h * (180 / math.pi)
            pix_s = s  # * 100
            pix_i = i  # * 255

            H[x, y] = pix_h / 360
            S[x, y] = pix_s
            I[x, y] = pix_i

    return H, S, I

# Loads image
image = cv2.imread('pyramid.png')
IMG = image / 255

#  Loading each dimension of the R, G and B space.
b = IMG[:, :, 0]
g = IMG[:, :, 1]
r = IMG[:, :, 2]

# Using the definition of RGB2HSI with the converted values of R, G and B.
hue, saturation, intensity = RGB2HSI(r, g, b)

# swapping the BGR to RGB by rearranging the stacks
IMG_swap = np.stack([intensity, saturation, hue], axis=2)

# showing the three different outputs
# cv2.imshow('hue', hue)
# cv2.imshow('saturation', saturation)  # they are shown in one dimensional space and therefore as a default in greyscale.
# cv2.imshow('intensity', intensity)
cv2.imshow('merged', IMG_swap)
cv2.imshow('org. image', IMG)  # original input

cv2.waitKey(0)

cv2.destroyAllWindows()

# Showing each output with colors

plt.imshow(hue)
plt.show()
plt.imshow(saturation)
plt.show()
plt.imshow(intensity)

plt.show()
