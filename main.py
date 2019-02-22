import matplotlib.pyplot as plt
import numpy as np
import cornermatching as cm

# Read in both images and set them to grayscale
I1 = plt.imread('im1.jpg')
I1 = I1.mean(axis=2)

I2 = plt.imread('im2.jpg')
I2 = I2.mean(axis=2)

# Gauss kernel
g_kernal = cm.gauss_kernal(5,2)

# Convolve the two images
#plt.imshow(I1, cmap="gray")
I1 = cm.convolve(I1, g_kernal)
#plt.show()

#plt.imshow(I2, cmap="gray")
I2 = cm.convolve(I2, g_kernal)
#plt.show()

H1 = cm.harris_response(I1)
H2 = cm.harris_response(I2)

H1sup = cm.nonmaxsup(H1)
H2sup = cm.nonmaxsup(H2)

H1descrips = cm.descriptorExtractor(I1,H1sup)
H2descrips = cm.descriptorExtractor(I2,H2sup)

#print(H1descrips)

best_matches = cm.get_best_matches(H1descrips, H2descrips)
secondbest_matches = cm.get_secondbest_matches(H1descrips, H2descrips, best_matches)

filtered_matches = cm.filter_matches(best_matches, secondbest_matches, H1descrips)

#print(filtered_matches[0][0][1],filtered_matches[0][0][2])
for match in filtered_matches:
    print(match[0][1:],match[1][0][1:])


#for point in H1sup:
#    plt.scatter(point[1],point[0], c= 'r', s = 10)
#plt.show()
#plt.imshow(H2, cmap="gray")
#plt.show()
