import numpy as np
import scipy as scipy
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skimage import io, color


plt.clf()
plt.close('all')

image = io.imread("./Map_of_the_ways_to_say_in_different_european_countries.png")
img_shape = image.shape  
pixels = image.reshape(-1, 3)  

data = pixels.T.astype(np.float64)  

# c-means ale rozmyte
n_clusters = 2
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data=data,
    c=n_clusters,
    m=2.0,
    error=0.005,
    maxiter=1000,
    init=None
)

print("Cluster centers:\n", cntr)
print("Fuzzy partition coefficient (FPC):", fpc)

cluster_membership = np.argmax(u, axis=0)

segmented_pixels = cntr[cluster_membership]
segmented_image = segmented_pixels.reshape(img_shape).astype(np.uint8)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(segmented_image)
plt.title(f"Segmented Image (c={n_clusters})")
plt.axis('off')
plt.show()
