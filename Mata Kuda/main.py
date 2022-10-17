import cv2
import numpy as np
import math

# tuplifies things for opencv
def tup(p):
    return (int(p[0]), int(p[1])) 

# returns the center of the box
def getCenter(box):
    x = box[0] 
    y = box[1] 
    x += box[2] / 2.0 
    y += box[3] / 2.0 
    return [x,y] 

# rescales image by percent
def rescale(img, scale):
    h,w = img.shape[:2] 
    h = int(h*scale) 
    w = int(w*scale) 
    return cv2.resize(img, (w,h)) 

# load video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
scale = 0.5 

# font stuff
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1 
font_color = (255, 255, 0) 
thickness = 2 

# set up tracker
tracker = cv2.TrackerCSRT_create()  # I'm using OpenCV 3.4
backup = cv2.TrackerCSRT_create() 

# grab the first frame
# success , frame = cap.read()
# frame = rescale(frame, scale)
#
# # init tracker
# box = cv2.selectROI(frame, False)
# tracker.init(frame, box)
# backup.init(frame, box)

# set center bounds
width = 75 
height = 60 

# save numbers
file_index = 0 

# blink counter
blinks = 0 
blink_thresh = 35 
blink_trigger = True

# show video

while True:
    # get frame
    success, frame = cap.read()
    box = cv2.selectROI(frame, False)
    tracker.init(frame, box)
    backup.init(frame, box)
    if not ret:
        break 
    frame = rescale(frame, scale) 

    # choose a color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    h,s,v = cv2.split(hsv) 
    channel = s 

    # grab tracking box
    ret, box = tracker.update(frame) 
    if ret:
        # get the center
        center = getCenter(box) 
        x, y = center 

        # make box on center
        tl = [x - width, y - height] 
        br = [x + width, y + height] 
        tl = tup(tl) 
        br = tup(br) 

        # get top left and bottom right
        p1 = [box[0], box[1]] 
        p2 = [p1[0] + box[2], p1[1] + box[3]] 
        p1 = tup(p1) 
        p2 = tup(p2) 

        # draw a roi around the image
        cv2.rectangle(frame, p1, p2, (255,0,0), 3) 
        cv2.rectangle(frame, tl, br, (0,255,0), 3) 
        cv2.circle(frame, tup(center), 6, (0,0,255), -1) 

        # get the channel average in the box
        slc = channel[tl[1]:br[1], tl[0]:br[0]] 
        ave = np.mean(slc) 

        # if it dips below a set value, then trigger a blink
        if ave < blink_thresh:
            if blink_trigger:
                blinks += 1 
                blink_trigger = False 
        else:
            blink_trigger = True 

        # draw blink count
        frame = cv2.putText(frame, "Blinks: " + str(blinks), org, font, fontScale,
                            font_color, thickness, cv2.LINE_AA)

    # show
    cv2.imshow("Frame", frame)

    # check keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# Destroy all the windows
cv2.destroyAllWindows()