import numpy as np
import cv2 as cv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-link_v', type = str, required= False)
parse = parser.parse_args()

#1. If argument is given read the video else read webcam input
if parse.link_v:
    cap = cv.VideoCapture(parse.link_v)
    w1 = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    out = cv.VideoWriter("../results/sample_video_edited.mp4",
                         fourcc, 30.0,(w1,h1))

else:
    cap = cv.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    #2. Adding my name in videos
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)-50)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    font_face = cv.FONT_HERSHEY_PLAIN
    scale = 0.8
    color = (0, 0, 0)
    thickness = cv.FILLED
    margin = 2
    text = "zxy"
    pos = [w,10]
    bg_color= [255,255,255]
    txt_size = cv.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv.rectangle(frame, pos, (end_x, end_y), bg_color, thickness)
    cv.putText(frame, text, pos, font_face, scale, color, 1, cv.LINE_AA)

    #3. Display
    if frame.any()!= None:
        frame2= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
        cv.imshow('frame-1',frame)
        #4. Display Grayscale version
        cv.imshow('frame-2', frame2)
        #5.2 Both windows side by side
        cv.moveWindow('frame-1', 00,00)
        cv.moveWindow('frame-2', 700,00)
        if parse.link_v:
            out.write(frame)
    #5.1 Press q to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
