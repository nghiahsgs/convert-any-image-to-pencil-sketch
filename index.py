import cv2
import numpy as np
#B1: Convert image to gray scale

img_rgb = cv2.imread("goku.png")
cv2.imshow("img_rgb", img_rgb)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray", img_gray)
cv2.imwrite('img_gray.png',img_gray)

#B2: Obtain a negative
img_gray_inv=255-img_gray

#B3: Apply gausian blur #Effective way to reduce noise and reduce amount of detail in a image

img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),sigmaX=0, sigmaY=0)
cv2.imshow("img_blur", img_blur)
cv2.imwrite('img_blur.png',img_blur)
#B4: Blend the grayscale image with the blurred negative

 
def dodgeNaive(image, mask):
  # determine the shape of the input image
  width,height = image.shape [:2] 
 
  # prepare output argument with same size as image
  blend = np.zeros((width,height), np.uint8)
 
  for col in xrange(width):
    for row in xrange(height):
      # do for every pixel  
      if mask[c,r] == 255:
        # avoid division by zero 
        blend[c,r] = 255
      else:
        # shift image pixel value by 8 bits
        # divide by the inverse of the mask
        tmp = (image[c,r] << 8) / (255-mask)
 
        # make sure resulting value stays within bounds
        if tmp > 255:
          tmp = 255
          blend[c,r] = tmp
 
  return blend
 
def dodgeV2(image, mask):
	return cv2.divide(image, 255-mask, scale=256)
def dodge(front,back):
    result=front*255/(255-back) 
    result[np.logical_or(result > 255, back ==255)] =255
    return result.astype('uint8')
'''
def burnV2(image, mask):
	return 255–cv2.divide(255-image, 255-mask, scale=256)
'''
img_blend = dodgeV2(img_gray, img_blur)
cv2.imshow("pencil sketch", img_blend)
cv2.imwrite('pencil_sketch.png',img_blend)
cv2.waitKey()
