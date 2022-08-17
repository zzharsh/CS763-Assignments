1. Screen Name:  zxy

2. Which external person (human resource) other than group members or TA I consulted:  None

3. What resources I used on the Internet: 
a. www.stackoverflow.com for various errors
b. For k-means: Lecture Notes from my previous CS 725 course.
c. https://numpy.org/doc/ documentation of numpy
d. https://docs.opencv.org/4.x/d4/db1/tutorial_documentation.html opencv documentation
e. https://www.youtube.com/watch?v=cdblJqEUDNo for understanding arparse
f. https://www.youtube.com/watch?v=XYUXFR5FSxI for understanding argparse

4. Honor Code
I pledge on my honour that I have not given or received any unauthorized assistance on this assignment or any previous task.  

Signed by:  Harsh Diwakar

5.  If I am awarded 90 marks for this task, my perception  of how much the breakdown is: 

Question  Me
a   100%

6.  Commands to run the code
Commands are same as defined in the lab instructions. I am listing out the extra details:
a.  display_video.py: 
    if the program is run without arguments, it will capture video from webcam and then display it in two different window after putting the text and converting to greyscale as:
	$ python3 display_video.py
    But if camera is not present and a path to a video file is given, then it should be given as:
	$ python3 display_video.py -link_v path_to_video_file
    In this case the program will open the video, display as it dispayed in above method and also save the video to results directory.
	