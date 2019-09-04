import cv2
vidcap = cv2.VideoCapture('India vs Pakistan 2007 4th Odi Gwalior - Full Highlights 40m60s - 50m60s (elIYmH2sR3E).mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1