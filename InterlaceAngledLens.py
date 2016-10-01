'''                              InterlaceAngledLens.py
created: 26/09/2016

To solve some of the problems that arise from lenticlar lens multiview displays (low resolution, picket fence) it has been
proposed to place the lens at an angle with respect to the screen - see van Berkel and Clarke 1997 paper.

This code performs the image interlacing for a lenticular lens placed on an angle with respect to the screen based
on the image processing allgorithms presented in the van Berkel 1999 paper (Image processing for 3D-LCD) and the Lee and Ra
2005 paper (Reduction of the distortion due to non-ideal lens alignment in lenticular 3D displays).

This code is specifically designed for a 7 view system with the lens slanted at an angle of 1/3. Some small modifications
would make the code adaptable to other configurations.

@author: dmcauslan
'''
#Import packages
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
import time
np.set_printoptions(precision=4, threshold=None, edgeitems=10, linewidth=120, suppress=None, nanstr=None, infstr=None, formatter=None)

## Define paramaters of the lens and screen.
# Define lenticular parameters.
lensWidth = 0.63178                            # width of lens (mm)
lensOffset = 0                                 # offset (number of lenses from left-hand side of screen)
lensAngle = math.atan(1/3)                     # Angle of the lens, alpha (radians)
alphaOff = 0                                   # Tilt offset (degrees) - used to compensate for lenses not placed at the perfect angle
lensAngle = lensAngle + math.pi*alphaOff/180   # Total angle of the lens, including offset (radians)
lensWidth = lensWidth/math.cos(lensAngle)      # Width of the lens as seen by the pixels (because it is slanted)

# Define screen parameters.
pixelPitch = 0.294                                  # Pixel pitch (mm)
numRGB = 3                                          # Number of subpixels (RGB) per pixel
subpixelPitch = pixelPitch/numRGB                   # Divide by 3 to get the subpixel pitch
nViews = 7                                          # Number of views per lens
nLens = 1                                           # Number of lenses per 3D pixel
Xview = lensWidth/subpixelPitch                     # The number of subpixels underneath one lenticule
screenRes = np.array([1024, 1280])                  # Screen resolution [height width] (pixels)
#screenRes = np.array([100, 100])                    # Screen resolution [height width] (pixels)
imageRes = (math.ceil(screenRes[0]/numRGB), math.ceil(screenRes[1]*numRGB/(Xview*nLens))+2)    # Resolution per view [vertical horizontal]
kOff = lensOffset*Xview                             # Horizontal offset of the lens array with respect to the LCD pixels. - Number of subpixels

# Define filename etc.
imType         = 'calibration'       # type of image to be interlaced (e.g., 'calibration', 'general', etc.)
generalBase    = 'ferrari'           # base file name (for 'general' case)
generalDir     = 'H:/David/Google Drive/Lenticular Sheet/images/{}/'.format(generalBase)    # base directory (for 'general' case)
generalExt     = 'jpg'               # file format    (for 'general' case)


def imageLoader(imageRes, nViews, imType, generalDir, generalBase, generalExt, numRGB):
    """ Loads/creates the images to be interlaced. """    
    maxPixelColour = 255        # Scaling factor when the images are coded as uint8   
    print('> Loading the images to be interlaced...')
    # Loop over the number of views
    for i in range(nViews):
        print('  - Loading frame {} of {}...'.format(i+1, nViews))
        # Choose which set of images to create/load
        # Calibration image - all images are black, except the centre image which is white        
        if imType == 'calibration':
            # On first iteration create array lfImage to hold all images
            if i==0:
                nImages = np.zeros(np.hstack((imageRes,numRGB,nViews)), dtype = np.uint8)
                nImages[:,:,:,i] =  maxPixelColour*(np.ones(np.hstack((imageRes,numRGB)), dtype = np.uint8))
        # Image used for alignment - alternating red, green, blue images.
        elif imType == 'redgreenblue':
            # On first iteration create array lfImage to hold all images            
            if i==0:
                nImages = np.zeros(np.hstack((imageRes,numRGB,nViews)), dtype = np.uint8)
            if i%numRGB == 0:
                nImages[:,:,0,i] =  maxPixelColour*np.ones(imageRes)
            elif i%numRGB == 1:
                nImages[:,:,2,i] =  maxPixelColour*np.ones(imageRes)
            elif i%numRGB == 2:
                nImages[:,:,1,i] =  maxPixelColour*np.ones(imageRes)
        # Image used for measuring crosstalk between views
        elif imType == 'numbersCrosstalk':
            # How the numbers will be arranged, need to change this as we change nViews            
            nRow = 2
            nCol = 4
            # Load the numbers crosstalk images
            im1 =  maxPixelColour*mpimg.imread('H:/David/Google Drive/Lenticular Sheet/images/numbers crosstalk/numbers crosstalk 0{}.png'.format(i+1))
            sz = np.shape(im1)
            # Rearrange them so the different images are arranged in a grid
            im2 = np.zeros(np.hstack((nRow*sz[0], nCol*sz[1], numRGB)), dtype = np.uint8)
            rows = ((i%nRow)*sz[0]+np.arange(sz[0])).astype(int)
            cols = (np.floor(i/nRow)*sz[1]+np.arange(sz[1])).astype(int)
            im2[rows[:, np.newaxis], cols, :] = im1
            # On first iteration create array lfImage to hold all images
            if i == 0:
                nImages = np.zeros(np.hstack((np.shape(im2),nViews)),dtype = np.uint8)
            nImages[:,:,:,i] = im2
        # General image interlacing - for example ferrari images
        elif imType == 'general':
            # On first iteration create array lfImage to hold all images            
            if i == 0:
                tmp = mpimg.imread('{}{}-{:02d}.{}'.format(generalDir, generalBase, i+1, generalExt))
                nImages = np.zeros(np.hstack([np.shape(tmp),nViews]), dtype = np.uint8)
            nImages[:,:,:,i] = mpimg.imread('{}{}-{:02d}.{}'.format(generalDir, generalBase, i+1, generalExt))                     
        else:
            raise ValueError('You did not correctly choose a set of images to load!')   
    
    ## Resizes the image to fit the view resolution.
    nImagesB = np.zeros(np.hstack((imageRes, numRGB, nViews)), dtype = np.uint8)
    for i in range(nViews):
        nImagesB[:,:,:,i] = resize(nImages[:,:,:,i], imageRes, preserve_range=True, order = 3) # Order 3 means bicubic interpolation
    
    # Plot images
    fig = plt.figure(1)
    plt.clf()    
    for n in range(nViews):
        ax = fig.add_subplot(241+n)
        ax.imshow(nImagesB[:,:,:,n],interpolation="nearest", aspect='auto')
        plt.title("View {}".format(n))
    plt.show()
    
    return nImagesB


def generateMappingTable(screenRes, lensAngle, Xview, nViews, kOff, numRGB):
    """ Generate mapping table """
    # For each value of k and l calculates which View (V) they correspond to and  which 3D pixel (X, Y) they belong to.
    # k and l refer to the xth and yth upsampled sub-pixel. 
    kMax = numRGB*screenRes[1]
    lMax = screenRes[0]
    kMat = np.tile(np.arange(kMax),(lMax,1))
    lMat = np.tile(np.arange(lMax)[:,np.newaxis],(1,kMax)) + 0.5        # + 0.5 is so you take the centre of each subpixel.
    
    qdash = numRGB*math.tan(lensAngle)                                  # The number of subpixels that the lens edge shifts by as it tranverses one subpixel (see Lee05)
    qoff = kOff + lMat*qdash - np.floor((lMat*qdash)/Xview)*Xview       # The offset of the lens for each l (in subpixels).
    locFact = kMat + 0.5 + kOff - lMat*qdash                            # +0.5 is so it takes the location of the centre of each subpixel
    X = np.floor((kMat + 0.5-qoff)/Xview).astype(int)+1                 # Note here X is the lens number, not the 3D pixel number. +0.5 is so it takes the location of the cente of each subpixel. The first plus 1 is because Matlab indexes things starting at 1. The second 1 is because we basically throw away the first row of 3D pixels to fill the extra space on the left edge that comes from having slanted lenses.
    Y = np.floor(lMat/numRGB).astype(int)                               # Vertical index of each 3D pixel. (each 3D pixel takes 3 subpixels vertically).
    V = np.floor((locFact%Xview)*nViews/Xview).astype(int)              # Mapping of views to subpixels. Note each subpixels actually covers multiple views so just corresponds to which view is closest to the centre of the pixel.
    
    XMax = np.max(X)+1
    [VMat, XMat, lMatB] = np.mgrid[:nViews, :XMax,  :lMax]
    qoffVect = qoff[:,0]                                                # qoff is the same for all values of X and V, so just need a column.
    h = Xview/nViews                                                    # Width of each viewing zone (in term of subpixels.
    D_VXl = np.tile(qoffVect,(nViews,XMax,1)) + (XMat - 1)*Xview + (VMat + 0.5)*h       # Where the centre of the View is in term of subpixels number (fractional value) for V, X, and l.  
    D_VXl_plus = np.floor(D_VXl+h/2)                                    # Which subpixel the upper edge of the View is on in terms of X,V,l.
    D_VXl_minus = np.floor(D_VXl-h/2)                                   # Which subpixel the lower edge of the View is on in terms of X,V,l.
    
    X_sav, Y_sav, V_sav, w_kXVl = calculatePixelWeightings(lMax, kMax, XMax, X, Y, V, D_VXl, D_VXl_plus, D_VXl_minus, h, nViews)    
    return X_sav, Y_sav, V_sav, w_kXVl 


def calculatePixelWeightings(lMax, kMax, XMax, X, Y, V, D_VXl, D_VXl_plus, D_VXl_minus, h, nViews):    
    ''' w_kXVl, V_sav, X_sav, Y_sav are the mapping table for the screen to go from l, k subpixels to 3D pixels X, Y, V. 
    Note the matrices are l x k x 3 because if the view width, h, is smaller than half the subpixel pitch (but greater 
    than a third) as it is here then at most 3 views can contribute to each subpixel. The w_kXVl are the weightings of 
    those views to each subpixel.'''
    numWs = 3
    w_kXVl = np.zeros((lMax,kMax,numWs))
    V_sav = np.zeros((lMax,kMax,numWs), dtype = int)
    X_sav = np.zeros((lMax,kMax,numWs), dtype = int)
    Y_sav = np.dstack((Y,Y,Y))
     
    # Loop over each subpixel l,k calculating the weightings from each view.
    t0A = time.time()
    for l in range(lMax):
        print('.', end="")
        for k in range(kMax):        
            Xtmp = X[l,k]
            if Xtmp == -1:
                print('-1')
            Dtmp = D_VXl[:,Xtmp,l]
            Dtmp_plus = D_VXl_plus[:,Xtmp,l]
            Dtmp_minus = D_VXl_minus[:,Xtmp,l]   
    
            count = 0      # As you loop over the 7 views most of the weights will be 0. Use a counter so that only the non-zero ones are added to w_kXVl       
            # Deals with corner case if V = 0 then subpixel k will probably have a contribution from the previous lens
            if k == Dtmp_minus[0] and Xtmp>=0:               
               w_kXVl[l,k,count] = (D_VXl[nViews-1,Xtmp-1,l] + h/2 - D_VXl_plus[nViews-1,Xtmp-1,l])/h
               count+=1           
            # Loops over the views, adding to w_kXVl when a view overlaps a subpixel. See Lee05 for expressions.
            for nV in range(nViews):
                if k == Dtmp_minus[nV] and k == Dtmp_plus[nV]:                
                    w_kXVl[l,k,count] = 1
                    count+=1
                elif k == Dtmp_minus[nV]:                
                    w_kXVl[l,k,count] = (Dtmp_minus[nV] + 1 - Dtmp[nV] + h/2)/h
                    count+=1
                elif k == Dtmp_plus[nV]:
                    w_kXVl[l,k,count] = (Dtmp[nV] + h/2 - Dtmp_plus[nV])/h
                    count+=1            
            # Deals with corner case if V = 7 then subpixel k will probably have a contribution from the next lens
            if k == Dtmp_plus[nViews-1] and Xtmp<(XMax-1):
                w_kXVl[l,k,count] = (D_VXl_minus[0,Xtmp+1,l] + 1 - D_VXl[0,Xtmp+1,l] + h/2)/h
                count+=1
    
            # Assigning views and Xs to weightings.
            Vtmp = V[l,k]
            X_sav[l,k,:] = [Xtmp, Xtmp, Xtmp]
            ''' There is basically two cases when Vtmp corresponds to the first weighting value, and when it corresponds
             to the second weighting value. To figure out which case if is test whether w_kXVl[1]>
             w_kXVl[2]. If its the first case then the view assignment is basically [V, V+1, V+2], if its
             the second case then the view assignment is [V-1, V, V+1]; however, these values need to be
             calculated modulo the number of views (7).  
             Note that if this is expanded to a two lens per nViews arrangement there will be 4 weightings per subpixel
             and we need to test if sum(w_kXVl[l,k,2:4]) > sum(w_kXVl[l,k,0:2])'''
            if w_kXVl[l,k,1] > w_kXVl[l,k,0]:
                V_sav[l,k,:] = [(Vtmp-1)%nViews, Vtmp, (Vtmp+1)%nViews]
                # The X values will usually just be Xtmp; however, if Vtmp = 1 or 7, then 1 needs to be either
                # added or subtracted to the value in X_sav.
                if Vtmp == 0:
                    X_sav[l,k,0] = Xtmp-1
                elif Vtmp == nViews-1:
                    X_sav[l,k,2] = Xtmp+1
            else:
                V_sav[l,k,:] = [Vtmp, (Vtmp+1)%nViews, (Vtmp+2)%nViews]
                if Vtmp == nViews-1:
                    X_sav[l,k,1] = Xtmp+1
                    X_sav[l,k,2] = Xtmp+1
                elif Vtmp == nViews-2:
                    X_sav[l,k,2] = Xtmp+1
    
    t0B = time.time()
    print("\ntime = {}s".format(t0B-t0A)) 
    return X_sav, Y_sav, V_sav, w_kXVl


def mapImages(nImages, X_sav, Y_sav, V_sav, w_kXVl, numRGB):
    """ Takes the loaded images and interlaces them based on the mapping described by 
    X_sav, Y_sav, V_sav, w_kXVL """
    numWs = np.shape(w_kXVl)[2]
    patternImage = np.zeros([screenRes[0], numRGB*screenRes[1], numWs])
    patternFinal = np.zeros([screenRes[0], screenRes[1], numRGB], dtype = np.uint8)
    
    for rgb in range(numRGB):
        # The subpixel indices for R, G, B subpixels
        kVect = np.arange(rgb, numRGB*screenRes[1], numRGB)
        # Calculates the linear indexing vectors to correctly assign the view images (lfImageB) to the final screen image.
        # Clip mode solves a couple of errors, but not sure if it is good practice...
        indxVect = np.ravel_multi_index((Y_sav[:,kVect,:], X_sav[:,kVect,:], rgb, V_sav[:,kVect,:]), dims = np.shape(nImages), order = 'C', mode = 'clip')
        # Using the indices above assigns 3D views to their values of l, k, weightidx.
        patternImage[:,kVect,:] = nImages.ravel(order = 'C')[indxVect]
    
    # Calculates the sum of the squared weights.     
    sumWSquare = np.sum(w_kXVl**2, axis = 2)
    # Calculates the weighted sums for each screen pixel.
    summedPattern = (np.sum((w_kXVl**2)*patternImage, axis = 2)/sumWSquare).astype(np.uint8)
    
    # Finally converts to an (v x h x 3) RGB image.
    for rgb in range(numRGB):
        patternFinal[:,:,rgb] = summedPattern[:,rgb::numRGB]        
    return patternFinal

    
def plotImage(imageName, figureNumber):
    fig= plt.figure(figureNumber)
    fig.clf()
    plt.imshow(imageName.astype(np.uint8), interpolation = "nearest")
    fig.show()


def saveImage(imageTot, imType, generalBase, nViews):
    """ Save the image """
    if imType == 'general':
        fName = 'H:/David/Google Drive/Canopy/Interlaced Images/{}_{}view_slanted_{}x{}.png'.format(generalBase, nViews, screenRes[1], screenRes[0])
    else:
        fName = 'H:/David/Google Drive/Canopy/Interlaced Images/{}_{}view_slanted_{}x{}.png'.format(imType, nViews, screenRes[1], screenRes[0])
    mpimg.imsave(fName, imageTot) 
    

def saveMap(X_sav, Y_sav, V_sav, w_kXVl, nViews, screenRes):
    """ Saves the mapping so that it doesn't have to be calculated each time we want to interlace a set of images.
    We only need to run the mapping until the screen is calibrated properly. Once the screen is calibrated we can just
    load the mapping. """
    outputFilename = 'H:/David/Google Drive/Canopy/Calibration Files/screenCalibration_{}Views_{}x{}.npz'.format(nViews, screenRes[1], screenRes[0])
    np.savez(outputFilename, X_sav = X_sav, Y_sav = Y_sav, V_sav = V_sav, w_kXVl = w_kXVl)
    

def loadMap(nViews, screenRes):
    """ Loads the screen calibration """
    mapFilename = 'H:/David/Google Drive/Canopy/Calibration Files/screenCalibration_{}Views_{}x{}.npz'.format(nViews, screenRes[1], screenRes[0])
    try:    
        npzfile = np.load(mapFilename)
    except FileNotFoundError:
        print("Looks like you haven't actually performed the calibration yet.")
        raise
    X_sav = npzfile["X_sav"]
    Y_sav = npzfile["Y_sav"]
    V_sav = npzfile["V_sav"]
    w_kXVl = npzfile["w_kXVl"]
    return X_sav, Y_sav, V_sav, w_kXVl


# Running the code
nImages = imageLoader(imageRes, nViews, imType, generalDir, generalBase, generalExt, numRGB)
remapRequired = False
if remapRequired:       
    X_sav, Y_sav, V_sav, w_kXVl = generateMappingTable(screenRes, lensAngle, Xview, nViews, kOff, numRGB)
    saveMap(X_sav, Y_sav, V_sav, w_kXVl, nViews, screenRes)
else:
    X_sav, Y_sav, V_sav, w_kXVl = loadMap(nViews, screenRes)
imageTotal = mapImages(nImages, X_sav, Y_sav, V_sav, w_kXVl, numRGB)
plotImage(imageTotal, 2)
saveImage(imageTotal, imType, generalBase, nViews)