# USAGE
# python object_size.py --image images/example_01.png --width 0.955
# python object_size.py --image images/example_02.png --width 0.955
# python object_size.py --image images/example_03.png --width 3.5

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os
import matplotlib.pyplot as plt

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


for f in os.listdir('./'):
    if f.split('.')[-1]=='mp4' or f.split('.')[-1]=='mkv' or f.split('.')[-1]=='avi' :
        cap = cv2.VideoCapture(f)        
        if not os.path.exists(f.split('.')[0]):
            os.makedirs(f.split('.')[0])
        i=-1
        ret, image = cap.read()
        while(ret):
            i+=1           
            ret, image = cap.read()   
#            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       
            # perform edge detection, then perform a dilation + erosion to
            # close gaps in between object edges
            if not ret:
                break
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            videowrite = cv2.VideoWriter('./'+f.split('.')[0]+'/mark.avi',fourcc,5,(image.shape[0],image.shape[1]))
            BGR=np.array([10,10,10])
            upper=BGR+20
            lower=BGR-10
            mask=cv2.inRange(image,lower,upper)
#            cv2.imshow(str(i),mask)
#            cv2.waitKey(0)                  
#            plt.imshow(edged)
            # find contours in the edge map
            
            cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_LIST,
            	cv2.CHAIN_APPROX_SIMPLE)
#            cv2.drawContours(mask.copy(),cnts,-1,(0,0,255),2)

            #cnts = imutils.grab_contours(cnts)
#            for i in range (len(cnts)):
#                print(cnts[i])
#            cv2.drawContours(image,cnts,-1,(0,0,255),2)
#            cv2.imshow('1',image)
#            cv2.waitKey(0)
            # sort the contours from left-to-right and initialize the
            # 'pixels per metric' calibration variable
            #(cnts, _) = contours.sort_contours(cnts)
            pixelsPerMetric = 50  #50个像素1cm
            orig = image.copy()            
            # loop over the contours individually
            area=900
            dimA=0
            dimB=0
            for c in cnts:                                        	
                if cv2.contourArea(c)<area  or cv2.contourArea(c)>40000 :
                    continue                
                left_most = min(c[:, :, 0])
                right_most = max(c[:, :, 0])
                top_most = min(c[:, :, 1])
                bottom_most = max(c[:, :, 1])
                print(left_most,right_most,top_most,bottom_most,cv2.contourArea(c))
                if left_most>370 or right_most<230 or top_most>275 or bottom_most<240:
                    continue                
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                
                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                box = perspective.order_points(box)
#                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
                
                # loop over the original points and draw them
                """
                for (x, y) in box:
                	cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                """
                # unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                
                # compute the midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
                
                # draw the midpoints on the image
                """
                cv2.circle(orig, (int(tltrX), int(tltrY)), 3, (255, 0, 0), -1)
                cv2.circle(orig, (int(blbrX), int(blbrY)), 3, (255, 0, 0), -1)
                cv2.circle(orig, (int(tlblX), int(tlblY)), 3, (255, 0, 0), -1)
                cv2.circle(orig, (int(trbrX), int(trbrY)), 3, (255, 0, 0), -1)
                """
                # draw lines between the midpoints
                # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                # 	(255, 0, 255), 2)
                #  cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                # 	(255, 0, 255), 2)
                
                # compute the Euclidean distance between the midpoints
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                # if the pixels per metric has not been initialized, then
                # compute it as the ratio of pixels to supplied metric
                # (in this case, inches)
                if pixelsPerMetric is None:
                	pixelsPerMetric = dB 
                
                # compute the size of the object
                dimA = dA / pixelsPerMetric
                dimB = dB / pixelsPerMetric
                area=cv2.contourArea(c)
                #     cv2.drawContours(orig, c, -1, (0,255,0),5)
                weeks=((dimA+dimB)/2+2.54)/0.7
            # cv2.putText(orig, "{:.1f}cm".format(dimA),
            	# (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            	# 0.65, (255, 255, 255), 2)
            # cv2.putText(orig, "{:.1f}cm".format(dimB),
            	# (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            	# 0.65, (255, 255, 255), 2)
            # cv2.putText(orig, "{:.1f}*weeks".format(weeks),
            	# (550, 450), cv2.FONT_HERSHEY_SIMPLEX,
            	# 0.65, (255, 255, 255), 2)
            cv2.imwrite('./'+f.split('.')[0]+'/'+str(i).zfill(6)+'.jpg',orig)
            videowrite.write(orig)
        cap.release()
        videowrite.release()
        cv2.destroyAllWindows()
        
        
        
