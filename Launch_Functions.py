# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:33:18 2018

@author: Pranav Shah
"""

# %%

# Importing required libraries

import os
from win32com.client import Dispatch


# %%

# Setting up the paths
base_path = "D:\\download\\YOLO_v3_tutorial_from_scratch-master (1)\\"
presentations_path = "4Others\\"
videos_path = "4Video\\"

# Setting up the dictionary for file names
dict_files = {"Blue_ball": "Soothsayer Overview Slides for Integr8-v02.mp4",
              "Red_ball": "Soothsayer Manufacturing Slides for Integr8V2.mp4",
              "Pink_ball": "Soothsayer Healthcare Slides for Integr8V2.mp4",
              "Orange_ball": "Soothsayer Retail Slides for Integr8V2.mp4"}

# Name of the presentation to be launched
ppt_file = "Test_Ppt2.pptx"

# %%


# Function to launch the videos
def launch_videos(current_video_id, previous_video_id):
    """
    This function will launch the video as per the video IDs passsed
    as arguments.

    Arguments:
        current_video_id: The argmax value of the list of outputs of 2 seconds
                          for the current iteraiton. Values = {0, 1, 2, 3}
        previous_video_id: The argmax value of the list of outputs of 2 seconds
                           for the previous iteraiton. Values = {0, 1, 2, 3}

    Returns: None
    """
    # print("this is current_id {}, this is previous _ id".format(
    # ccurrent_video_id, previous_video_id))
    if(current_video_id != previous_video_id):
        print("Playing video for File ID: " + str(current_video_id))
        os.startfile(base_path + videos_path + dict_files[current_video_id])

        # # Show the video for 5 seconds
        # time.sleep(5)

# %%


def launch_presentation(ppt_file):
    """
    This function will launch the presentation in Windows PowerPoint
    Application.

    Arguments:
        ppt_file (String): Name of the Powerpoint file mentioned in
                           double quotes and extension '.pptx'

    Returns:
        PowerPoint: win32com.client.CDispatch Application object
        Presentation: win32com.client.CDispatch Presentation object
    """

    # Dispatching the application to be launched

    PowerPoint = Dispatch("PowerPoint.Application")

    # Launching the PowerPoint application
    Presentation = PowerPoint.Presentations.Open(
            base_path + presentations_path + ppt_file)
    PowerPoint.Visible = True

    # If multiple applications are launched using 'win32com.client' library
    # then this command will help navigate amongst the active applications
    # It will enact the function of 'Alt+Tab'
    PowerPoint.Windows(1).Activate()

    # Launch the slideshow
    Presentation.SlideShowSettings.Run()
    Presentation.SlideShowWindow.SlideNavigation.Visible = False

    # Show the slide for 3 seconds
    # time.sleep(3)

    # Return the Presentation object
    return PowerPoint, Presentation

# %%


def navigate_presentation(Presentation, gesture):
    """
    This function will navigate through the Slideshow of the presentation
    in Windows PowerPoint Application based on the argument value.

    Arguments:
        Presentation (win32com.client.CDispatch):
            Presentation object created by win32com.client library
        gesture(int):
            Integer corresponding the gesture. Performs action after 2 seconds
            * 1 for 'FIVE' hand gesture - Show the current slide
            * 0 for 'FIST' hand gesture - Show the next slide
            * 3 for 'SWING' hand gesture - Show the previous slide

    Returns: None
    """

#     Do the action according to the gesture
    if (gesture == 0):
        # Print the status
        print("Showing next slide in 2 seconds.")
        # Show the next slide
        Presentation.SlideShowWindow.View.Next()
    elif (gesture == 3):
        # Print the status
        print("Showing previous slide in 2 seconds.")
        # Show the previous slide
        Presentation.SlideShowWindow.View.Previous()
    elif (gesture == 1):
        # Print the status
        print("Showing the current slide.")
        # Show the current slide
        Presentation.SlideShowWindow.View.Slide

    # Show the slide for 2 seconds
    # time.sleep(2)

# %%


# Testing for the main code
if __name__ == "__main__":

    # %%

    # Setting the current video ID to launch the video
    video_ids = [1, 1, 1, 3, 3, 3, 3, 3, 2, 2, 2, 0, 0, 0, 0]

    # Setting the previous video ID to launch the video
    # This has to be set in the main frame code (Vivek's code)
    previous_video_id = -1

    # Looping for different video IDs
    for idx, current_video_id in enumerate(video_ids):

        # Testing the function call
        launch_videos(current_video_id, previous_video_id)

        # Making the current video ID as a previous video ID for next iteration
        previous_video_id = current_video_id

    # Print the status of exit of VLC player
    print("Shutting down the VLC Player")

    # Exit the application for the last request
    os.system("TASKKILL /F /IM vlc.exe")

# %%

#    # Code chunk to kill VLC player if it's open
#    list_programs = psutil.pids()
#    for i in range(0, len(list_programs)):
#        try:
#            p = psutil.Process(list_programs[i])
#            if p.cmdline()[0].find("vlc.exe") != -1:
#                print("VLC found at ", i, " position. Kill it")
#                p.kill()
#                break
#        except IndexError:
#            pass

    # %%

    # List of gesture IDs
    gesture_ids = [2, 2, 2, 1, 1, 1, 3]

    # Control Powerpoint application using Python COM objects
    a = launch_presentation(ppt_file)

    # Navigate through the presnetation
    navigate_presentation(gesture_ids, ppt_file)

# %%
