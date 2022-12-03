import cv2
import numpy as np
import random as rng
from itertools import groupby
import matplotlib.pyplot as plt

#it looks like the lowest overhead way to do a rotation is with a port of the function from my C library...
def rotate(points, angle, center = (0,0)):
    points_out = []
    for i, p in enumerate(points):  
     c = np.cos(angle)
     s = np.sin(angle)     
     x = center[0] + (p[0] - center[0])*c - (p[1] - center[1])*s
     y = center[1] + (p[0] - center[0])*s + (p[1] - center[1])*c 
     points_out.append((x,y))
    return points_out


def getContours(image):
  #image_gray          = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_gray          = image[:,:,2] # red channel for white gears
  image_gray          = cv2.GaussianBlur(image_gray,(15,15),0)
  
  #cv2.imshow('Binary image', cv2.resize(image_gray, (1024,768)))
  #cv2.waitKey(6000)
  
  _,image_gray        = cv2.threshold(image_gray, 127, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_OTSU)
  
  #cv2.imshow('Binary image', cv2.resize(image_gray, (1024,768)))
  #cv2.waitKey(6000)
  
  contours, _         = cv2.findContours(image_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  print (len(contours), "contours found")
  return contours


def sortContoursOne(contours):
  #break down the contours for us into circles and polygons
  centers      = []
  radii        = []
  contours_out = []

  for i, c in enumerate(contours):
   # print("index",i,"has",len(c),"points")
    if (len(c) < 10):
      print("ditching profile", i, "because its got less than 10 points...") 
      continue
    center, radius = cv2.minEnclosingCircle(c)   
    if ((radius < 15) or (radius > 972)) :
      print("ditching profile", i, "becasue its radius is too ... wrong...", radius)
      continue   
    a = cv2.contourArea(c)
    if ((a < 17000) or (a > 2500000)):
      print("ditching profile", i, "because the area is... ugly...", a) 
      continue
    centers.append(center)
    radii.append(radius)
    contours_out.append(c)
     
  #print (len(centers), "Items remaining")
  return radii, centers, contours_out


# this might not be the best method... 
# we might be better off looping thru points rejecting ones that go backwards angles (or forward too far).
# the output should be of evenly spaced, increasing angles.
def correctPolarData(polar):

  # sort points by polar angle
  normal = polar[np.argsort(polar[:,1])]       

  #make a set of 2000 sample angles from -Pi to +Pi
  angles = np.linspace(-3.14, 3.14,500)

  #interpolate the data for even samples at angles
  normal = np.interp(angles, normal[:,1], normal[:,0])
  
  return normal
  
  
def _correctPolarData(polar):  
  normal = polar[:,0]  
  return normal  


# cartesian to polar conversion
def unrollContours(radii, centers, contours):
  centers_out = []
  polar_out = []
  radii_out = []
    
  for i, c in enumerate(contours):
    #make a poly gone
    ar = np.array( cv2.approxPolyDP(c, 2, True) )
    # recenter it on zero
    ar = ar - [centers[i][0],centers[i][1]] 
    #convert to polar. (distance, angle)
    polar = np.stack((np.hypot(ar[:,0,0],ar[:,0,1]), np.arctan2(ar[:,0,0],ar[:,0,1])), axis=1)                                          

    #print ("Polar is:", polar)   

    #calculate the tooth height       
    varia = np.var(polar, 0)
    #print (" variance is: ", varia)

    if (varia[0] < 5):
      print("ditching profile", i, "becasue teeth are smaller than 10 pixies...")   
      continue

    polar = correctPolarData(polar) # re-order and re-sample 
    centers_out.append(centers[i])
    polar_out.append(polar) 
    radii_out.append(radii[i])
   
  return  radii_out, centers_out, polar_out
    


def amplitudeToothCounter(radii, centers, ippolar):
  centers_out = []
  count_out = []
  radii_out = []
  
  for i, c in enumerate(ippolar):
    polar = np.array(c)
    
    average = np.mean(polar, 0 )
    #print("average is:", average)    
    #subtract the average radius
    polar = polar - average     
        
#    plt.plot(list(range(0, len(polar))), polar, [0,len(polar)], [0, 0] )
#    plt.show()         
        
    #clip 0-1 based on >0
    teeths = list(polar>0)
    
    #remove sequential dups  
    
    teeths = [i[0] for i in groupby(teeths)]
 
#    plt.plot(list(range(0, len(teeths))), teeths, [0,len(teeths)], [0, 0] )
#    plt.show()  
    
    #print (teeths)
    
    #end connections, if the first and last value of the array are the same, delete one of them
    if (teeths[0] == teeths[-1]): teeths.pop()
    
    # add everyone up
    toothCount = sum(teeths)
    
    if (toothCount < 5):
      print("ditching profile", i, "not enough toofs")
      continue
    
    print ("Amplitude toothcount is: ", toothCount)
    centers_out.append(centers[i])
    count_out.append(toothCount)
    radii_out.append(radii[i])
    
  return radii_out, centers_out, count_out

#make greyscale waves of the inside edge of the circle.
def makeRingWaves(image, radii, centers):
  inten_out           = []

  intenWave           = []
 # image_gray          = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_gray          = image[:,:,2] # red channel for white gears  
  color               = 255  
  angles              = np.linspace(-3.14, 3.14,500)
  
  for i, c in enumerate(centers):
    micro_grey          = image_gray[int(c[1]-radii[i]):int(c[1]+radii[i]), int(c[0]-radii[i]):int(c[0]+radii[i])]
    mask                = micro_grey.copy()
    
#    cv2.imshow('Binary image', micro_grey)
#    cv2.waitKey(6000)

    p0                  = (radii[i]*2-10, radii[i]) #middle 
    mask[:]             = 0
    cv2.circle(mask, (int(p0[0]), int(p0[1])), 5, color, -1) # make sample size mask
    maskBase = mask.sum()

    for a in angles:  
      mask[:]             = 0
      p1 = rotate([p0], a, (radii[i], radii[i]))[0] 
      cv2.circle(mask, (int(p1[0]), int(p1[1])), 5, color, -1) # circle along edge-5pix 
      sample              = cv2.bitwise_and(micro_grey,micro_grey,mask = mask)
      bri                 = sample.sum()/maskBase
      intenWave.append(bri*100)

    inten_out.append(intenWave)
    intenWave = []

  return radii, centers, inten_out   


def fftToothCounter(radii, centers, ippolar):
  centers_out = []
  count_out = []
  radii_out = []
  
  for i, c in enumerate(ippolar):
    fftpoints = np.array(c)

    fftResult = np.fft.fft(fftpoints)   
    #ditch the positive fequency values (and the dc value)
    fftResult = fftResult[0:250]     

    # print("Hi!", fftpoints)
    # fftResult[0] = 0

#    plt.plot(list(range(0, len(fftpoints))), fftpoints)
#    plt.show()
    
    #reduce to amplitudes
    fftResult = abs(fftResult)
    fftResult[0] = 0
    fftResult[1] = 0
    fftResult[2] = 0
    fftResult[3] = 0
    fftResult[4] = 0
    
#    plt.plot(list(range(0, len(fftResult))), fftResult)
#    plt.show()

    #draw a cutoff at 90% the peak value
    peak = np.max(fftResult)
    if (peak < 950): #1200):
      print("ditching profile", i, "low FFT amplitude", peak)
      continue

    #print ("max is: ", peak)
    peak = peak *0.75

    peakCount = (fftResult>peak).sum()
    #print ("peak count is: ", peakCount)

    # peaks = [x for x in fftResult if x<=peak]
    peaks = [(i, fftResult[i]) for i, item in enumerate(fftResult) if item > peak] 
    if (len(peaks) > 2):
      print("ditching profile", i, "too much noise", peaks)
      continue

#    print (">>> peaks are: ", fftResult[peaks] , "at", peaks, "<<<<")  
    print (">>> peaks are: ", peaks, "<<<<")           

#    indexes = list(range(0, len(fftResult)))
#    plt.plot( indexes[0:200], fftResult[0:200], [0,len(fftResult[0:200])], [peak, peak])
#    plt.show()

    centers_out.append(centers[i])
    count_out.append(peaks[0][0])
    radii_out.append(radii[i])
    
  return radii_out, centers_out, count_out


def drawCounts(drawing, radii, centers, counts):
  font = cv2.FONT_HERSHEY_SIMPLEX  
  for i, c in enumerate(counts):
    color = (rng.randint(128,256), rng.randint(10,256), rng.randint(10,256))
    p0 = (int(centers[i][0]), int(centers[i][1]))

    cv2.circle(drawing, p0, int(radii[i]), color, 3) # circle along edge
    cv2.circle(drawing, p0, 20, color, -1)           # filled circle in middle
            
    xf = (p0[0] < 1300)
    yf = (p0[1] < 980)

    p1 = np.add(p0, [radii[i]+60, 0])
    
    t1 = np.add(p0, [radii[i], 0])
    t2 = np.add(p0, [radii[i]+40, 10])
    t3 = np.add(p0, [radii[i]+40, -10])    
    
    if ([xf,yf] == [True, True]): # go right down
      angle = rng.randint(0,45)
    if ([xf,yf] == [True, False]): # go right up
      angle = rng.randint(315,360)
    if ([xf,yf] == [False, True]): # go left down
      angle = rng.randint(135,180)
    if ([xf,yf] == [False, False]): # go left up
      angle = rng.randint(180, 225)
    
    ps = rotate([p1, t1, t2, t3], np.deg2rad(angle), (p0[0], p0[1]) )
    p1 = ps[0];    
    
    triangle = np.array([ (int(ps[1][0]), int(ps[1][1]) ), (int(ps[2][0]), int(ps[2][1]) ), (int(ps[3][0]), int(ps[3][1]) ) ])
    cv2.drawContours(drawing, [triangle], 0, color, -1)
        
    p0 = (int(p0[0]), int(p0[1]) )
    p1 = (int(p1[0]), int(p1[1]) )
    
    cv2.line(drawing, p0, p1,  color, 3) #takeoff                      
           
    if (xf):
      x = p1[0] + 50
    else:
      x = p1[0] - 50
    
    cv2.line(drawing, p1, (x, p1[1]),  color, 3) #takeoff tail
    
    if (xf == 0): x = x - 100
    
    cv2.putText(drawing, str(c) ,(x, p1[1]+30) , font, 2.5, color, 4, cv2.LINE_AA) # gear count
        
  return drawing


def main():
  print("###################################################")
  np.set_printoptions(suppress=True)
  
  image                    = cv2.imread("foo.jpg")
  contours                 = getContours(image)
  radii, centers, contours = sortContoursOne(contours)
  
  radii, centers, polar    = unrollContours(radii, centers, contours)
  Iradii, Icens, Icounts   = makeRingWaves(image, radii, centers)  

  Aradii, Acens, Acounts   = amplitudeToothCounter(radii, centers, polar)
#  Fradii, Fcens, Fcounts   = fftToothCounter(radii, centers, polar)
  Fradii, Fcens, Fcounts   = fftToothCounter(radii, centers, Icounts)  
  
  image = drawCounts(image, Aradii, Acens, Acounts)
  image = drawCounts(image, Fradii, Fcens, Fcounts)
  
  cv2.imshow('Binary image', cv2.resize(image, (1024,768)))
  cv2.imwrite('image_counts.jpg', image)
  cv2.waitKey()
  cv2.destroyAllWindows()


main()
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
