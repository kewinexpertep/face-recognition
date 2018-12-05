import face_recognition
import cv2
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import os
from os import listdir
from os.path import isfile, join
from imutils.video import VideoStream
import imutils
import pickle
     
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())
frameSize = (640, 360)
areaFrame = frameSize[0] * frameSize[1]
MinCountourArea = areaFrame * 0.0111  #Adjust ths value according to your usage
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()

#camera.crop = (0.25, 0.25, 0.5, 0.5)
camera.crop = (0.22, 0.10, 0.40, 0.40)
camera.rotation = 180
camera.hflip = False
camera.resolution = (640, 360)
camera.framerate = 30
camera.brightness = 65
#camera.roi = (0.5,0.5,0.25,0.25)
#camera.brightness = 60
rawCapture = PiRGBArray(camera, size=(640, 360))
# Get a reference to webcam #0 (the default one)
namefile = None
if args["video"] is None :
    #camera = VideoStream(src=0, usePiCamera=True, resolution=frameSize, framerate=15).start()
    namefile = 'camera'
else :
    camera = cv2.VideoCapture(args["video"])
    namefile = args["video"]
known_face_encodings = []
known_face_names = []
pathKnownImg = "encodings.pickle"
data = pickle.loads(open(pathKnownImg, "rb").read())
known_face_encodings = data["encodings"]
#known_face_names = data["name"]

# Create arrays of known face encodings and their names
fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=100, detectShadows=False)
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
motion_flag = False
idle_time = 0
namePre = ""
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    if args["video"] is None :
        #frame = camera.read()
        #frame = imutils.rotate(frame, 180)
        frame = image.array
    else:
        _,frame = camera.read()
        if (frame is None):
            # not connect camera
            break
        x = 300
        y = 300
        frame = frame[y:y+450, x:x+800]
        frame = imutils.resize(frame, width=frameSize[0])

    # Grab a single frame of video
    # Only process every other frame of video to save time
    if process_this_frame :
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.4)
            name = "Unknown"
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            face_names.append(name)
    process_this_frame = False
    if idle_time%5==0:
        process_this_frame = True

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    # Display the resulting image
    cv2.imshow('Video', frame)
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    idle_time += 1
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()