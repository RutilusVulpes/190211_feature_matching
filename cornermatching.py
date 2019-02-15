import numpy as np
import math as mt

def convolve(g,h): # h is kernel, g is the image
    I_gray_copy = g.copy()

    x,y = h.shape
    xl = int(x/2)
    yl = int(y/2)
    for i in range(xl,len(g[:,1])-xl):
        for j in range(yl, len(g[i,:])-yl):

            f = g[i-xl:i+(xl+1), j-yl:j+(yl+1)] #FIXME

            total = h*f
            I_gray_copy[i][j] = sum(sum(total))
    return I_gray_copy

def gauss_kernal(size, var):
    kernel = np.zeros(shape=(size,size))
    for i in range(size):
        for j in range(size):
            kernel[i][j] = mt.exp( -((i - (size-1)/2)**2 + (j - (size-1)/2)**2 )/(2*var*var))

    kernel = kernel / kernel.sum()
    return kernel

def harris_response(img, gmean = 5,var =2):
	sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	gauss = gauss_kernal(gmean,var)
    #calculate the harris response using sobel operator and gaussian kernel

	Iu = convolve(img,sobel)
	Iv = convolve(img,sobel.transpose())

	Iuu = convolve(Iu*Iu,gauss)
	Ivv = convolve(Iv*Iv,gauss)
	Iuv = convolve(Iu*Iv,gauss)

	H = (Iuu*Ivv - Iuv*Iuv)/(Iuu + Ivv + .0000000001)

	return H

def getmaxima (H,threshold):
    maxima = []
    localSearchWidth = 21
    p = localSearchWidth

    width,height = H.shape
    for i in range(int(p/2)+1,width-int(p/2)+1,p):
        for j in range(int(p/2)+1,height-int(p/2)+1,p):
            if H[i,j] < threshold:
                continue
            else:
                localMax = [0,0,0]
                for x in range(i-int(p/2),i+int(p/2)+1):
                    for y in range(j-int(p/2),j+int(p/2)+1):
                        if(H[x][y] > localMax[2]):
                            localMax = [x,y, H[x][y]]
                maxima.append(localMax)
    return maxima


def nonmaxsup(H,n=100,c=.9):

    mindistance = []
    threshold = np.mean(H) + np.std(H)
    maxima = np.array(getmaxima(H,threshold))

    x = 0
    y = 1
    z = 2
    for row in maxima:
        min = np.inf
        for row1 in maxima:
            if (row[z] < c*row1[z]):
                dist = np.sqrt((row[x]-row1[x])**2 + (row[y]-row1[y])**2 )
                if (dist < min) and (dist>0):
                    min = dist
                #xmin = row1[x]
                #ymin = row1[y]

        mindistance.append([row[x],row[y],min])
    mindistance.sort(key=lambda x:x[2])
    return mindistance[-n:]

#def descriptorExtractor(img, featureList):
