###################################################################################
# Author; John  Moran
# Android Eye Tool
# Tool interfaes between OpenCV and Android Debug Bridge
# to scroll through a multiple screen industrial device
# capturing the screen shots.
# The device must have the information in bounded cells less
# than the heigth of the device's screen.
# This version of the tool is used as demo on a Samsung Galaxy
# A51.
#
#
# Usage. 
# In a Linux env with ADB running start a Python3 shell
# Then create an Android Eye object:. E.g.:  e = android_eye(test_dir="test6")
# Then invoke the run_capture method. E.g.:  e.run_capture()
###################################################################################


import cv2
import subprocess
import numpy as np
import sys
from ppadb.client import Client as AdbClient
from enum import Enum
import os
from pathlib import Path
import sys
import difflib
import filecmp
import re
import time
from datetime import datetime
import cProfile
from collections import Counter

###################################################################################
########### Class to Inferface to ADB
###################################################################################


class adb_driver():

    def __init__(self):
        """
       Function initialize adb
       Note - move to class constructor
        """     
        self.client = AdbClient(host="127.0.0.1", port=5037)
        devices = self.client.devices()
        self.device = devices[0]
        return self.device 

    def swipe_adb(self, x1=0, y1=790, x2=0, y2=0, dur=1900):
        """
        Function to swipe
        the android devices screen
        """         
        self.device.input_swipe(x1, y1, x2, y2, dur)

    def capture_adb(self, device, screen_cnt):
        """
        Function to capture
        the android devices screen
        """
        screen = device.screencap()
        with open("screen.png", "wb") as fp:
            fp.write(screen)


###################################################################################
########### Android Eye Class
###################################################################################

class android_eye(adb_driver):

    # Constant to define crop points
    CROP_TOP = 730 
    CROP_BOT = 1950
    CROP_LEFT = 0
    CROP_RIGHT = 960
    # Scroll bar removal crop point
    SIM_CROP_RIGHT = 460
    SIM_CROP_BOTTOM = 800   
    # Adaptive scrolling crop points
    AJ_SC_FULL_SCN = 780
    AJ_SC_TUN_SCN=CROP_BOT-CROP_TOP  
    AJ_UNDERSWIPE_FACTOR = 0.97
    # Duplicate Screen Check Constants
    SIMILARITY_THRES = 8000  # 
    PIXEL_LEVEL_SIMILARITY_THRES = 30  
    # Constants for output files dirs
    IMAGES_PATH_LOG = "images/"
    TESSERACT_PATH_ROOT = "tesseract/"
    LOG_PATH_ROOT = "log/"
    GOLDEN_VEC_ROOT = "golden/"
    # Screen check location constants
    TRUN_SCREEN_THRESHOLD_CK = 30  #  This defines how far we check for a color transition to determine screen trunction
    RHS_MARGIN_THRES = 20         # This defines how far away from the RHS of the image we sample for transition detection, color detection etc
                                   # offset is needed to remove noise from fading effects on scroll bar etc.


    class Color(Enum):  
            UNKNOWN       = 0
            RED           = 1
            PURPLE        = 2
            WHITE         = 3
            GREEN         = 4
            BLUE          = 5
            NAVY_BLUE     = 6
            YELLOW        = 7

    class Header(Enum): 
            UNKNOWN         = 0
            LAST_TRAVEL     = 1
            TRANS_HIS       = 2



    def __init__(self, test_dir):
        self.device = super().__init__()   # Init the adb device
        self.color_lu_dict = self.setup_color_lu_dict()
        self.test_dir = test_dir
        self.TESSERACT_PATH = self.test_dir + "/" + android_eye.TESSERACT_PATH_ROOT
        Path(self.TESSERACT_PATH).mkdir(parents=True, exist_ok=True) 
        self.IMAGES_PATH = self.test_dir + "/" +  android_eye.IMAGES_PATH_LOG
        Path(self.IMAGES_PATH).mkdir(parents=True, exist_ok=True) 
        self.LOG_PATH = self.test_dir + "/" +  android_eye.LOG_PATH_ROOT
        Path(self.LOG_PATH).mkdir(parents=True, exist_ok=True) 
        self.CWD_PATH = os.getcwd()
        self.prev_string_dict = set()  # Set to track previous recorded strings



#   @classmethod    
    def setup_color_lu_dict(self):
        """
        Function initialize the color lookup
        dictonary, note - move to class constructor
        BGR values to be used as keys
        """
        color_lu_dict = {"[127   0 127]" : self.Color.PURPLE,
                         "[  0   0 255]" : self.Color.RED,
                         "[  0 255   0]" : self.Color.GREEN,
                         "[255 255 255]" : self.Color.WHITE,
                         "[255   0   0]" : self.Color.BLUE,
                         "[127   0  63]" : self.Color.NAVY_BLUE,
                         "[  0 180 255]" : self.Color.YELLOW
                         }
        return color_lu_dict


    def find_sub_images(self, image, page_trunc):
        """
      Function to find sub-images from full inpsection app image
      This function will scan down the RHS of the image and find 
      transitions on colors to identify sub-image boundaries.
      The location of the transition points are stored in array trans_pnt
      Note, the last segement of truncated screens (i.e. all screens that are not the last screen)
        """
        trans_pnt=[0]
        sub_image_lst=[]
        pre_boundary_row_idx=0
        cv2.imwrite(self.IMAGES_PATH + 'bot_tunc_debug.png',image)
        for row_idx in range(image.shape[0]-1): # Loop across all rows jmoran was -1 
            right_most_row = image[row_idx, image.shape[1]-1] # RHS Column of image
            nxt_right_most_row = image[row_idx+1, image.shape[1]-1] #Subsequent RHS Column of image
            if(right_most_row != nxt_right_most_row): # If a transition in color exists
                trans_pnt.append(row_idx+1)
                sub_image_lst.append(image[pre_boundary_row_idx:row_idx+1,:])
                pre_boundary_row_idx=row_idx+1       
        if(page_trunc and (image.shape[0] - trans_pnt[-1] >3)):   ## If on the last page, capture the final segment
            lst_segment = image[trans_pnt[-1]:-1]   
            sub_image_lst.append(lst_segment)
            trans_pnt.append(image.shape[0])
        return  sub_image_lst, trans_pnt    


    def find_color_sub_images(self, image, trans_pnt):
        """
         Function to find the color sub-images once they 
        greyscale images have already been identified
        """
        color_sub_image=[]
        for row_idx in range (len(trans_pnt)-1):
            color_sub_image.append(image[trans_pnt[row_idx]:trans_pnt[row_idx+1],:,:])
        return color_sub_image


    def find_image_color(self, image):
        """
      Function to the background color of an image
      This function will scan down the RHS of the image and find 
      color present
        """ 
        right_most_row = image[0, image.shape[1]-android_eye.RHS_MARGIN_THRES] # RHS Column of image
        key=str(right_most_row) 
        color = self.color_lu_dict.get(key, 0)   
        return color

    def invert_img(self, image):
        """
        Function to invert image
        Used when there's a red background 
        as tesseract stuggles in this case
        """ 
        image = cv2.bitwise_not(image)
        return image

    def thres_img(self, image):
        """
        Function to threshlod an image
        """ 
        image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
        return image

    def gen_screen_struct(self, inFile, screen_cap, section, color, segement_size):
        """
        Function to form the data structure containing cell info:
        (1) Text as output from Tesseract 
        (2) The back-grounf color of the cell
        (3) The height of the cell
    	The order of the element in the list reflects the 
    	ordering of the cells on the screen
        """
        string = ""
        screen_cap_sec_lst = [[],[],[]]
        for line in inFile:
            if line.strip():
                string += line  
        if(string !=""):     
            if(string not in self.prev_string_dict):   ## Dont insert duplicates   - check previous string dictonary
            	self.prev_string_dict.add(string)
            	screen_cap_sec_lst[0] = string  
            	screen_cap_sec_lst[1] = color           
            	screen_cap_sec_lst[2] = segement_size
            	screen_cap.append(screen_cap_sec_lst)
        else:
        	self.prev_string_dict.add("Dummy")                       
        	screen_cap_sec_lst[0] = "Dummy"             ## Identify text free segments with the keyword "Dummy", 
        	screen_cap_sec_lst[1] = self.Color.WHITE    ## we need to keep these as they may be the truncated segement      
        	screen_cap_sec_lst[2] = 0                   ## which we need to remove later.  Any remaining empty cells are filtered out 
        	screen_cap.append(screen_cap_sec_lst)       ## before the text file is created.
        return screen_cap


    def capture_screen(self, sub_image_lst, color_sub_image, screen_capture, page_trunc, trans_pnt, screen_cnt):    
        """
        Function to capture an indivual screen color and text
        Takes sub-images of the inspection screen 
        as input.
        Requires both the grey scale and color versions
        of the sub-images. 
        Returns a data structure containing the text and 
        color of each segment
        """
        for count, sub_image in enumerate(sub_image_lst):
            if(not((page_trunc and (screen_cnt !=1)) and (count==0))):    ##  Detect First Segment of Last Page and if its truncated
                color = self.find_image_color(color_sub_image[count])
                if(color == self.Color.RED or color == self.Color.NAVY_BLUE):
                    sub_image = self.invert_img(sub_image)
                cv2.imwrite(self.IMAGES_PATH + 'sub_image{}.png'.format(count, screen_cnt),sub_image)
                cv2.imwrite(self.IMAGES_PATH + 'color_sub_image{}.png'.format(count),color_sub_image[count])   
                img_path = self.IMAGES_PATH + "sub_image" + str(count) + ".png"         
                out_path = self.TESSERACT_PATH + "screen" + str(count)
                tesseract_cmd = "tesseract {0} {1}  --psm 6 quiet".format(img_path,out_path)
                os.system(tesseract_cmd)
                tesseract_file = open(self.TESSERACT_PATH + "screen%d.txt" % count, "r")
                segement_size  = trans_pnt[count+1] -  trans_pnt[count]
                segement_pos   = count
                screen_capture = self.gen_screen_struct(tesseract_file, screen_capture, count, color, segement_size)
        return screen_capture


    def check_trans_at_topscn(self, image): 
        """
        Function to check the transition point on a screen capture - 
        used to determine is a screen is truncated
        """
        page_trunc=(image[0:self.TRUN_SCREEN_THRESHOLD_CK, image.shape[1]-android_eye.RHS_MARGIN_THRES]==image[0, image.shape[1]-android_eye.RHS_MARGIN_THRES]).all()     
        return page_trunc   

    def element_parser(self, screen_capture, current_time):
        """
        Function to parse the screen capture into different categories
        This was used on the version of the tool I used for work purposed - 
        This is not used for the Samsumg Phone demo version of the tool.
        I will leave it here as a place holder
        """

        return screen_capture

    def print_op_file(self, screen_capture, op_file_name):
        """
        Function to print the captured screen contents 
        to an output file
        """
        textfile = open(op_file_name, "w")  
        for element in screen_capture:
            if("Dummy" not in element[0]):
            	textfile.write("--------------------------------------------------------------\n")            	
            	for sub_idx, sub_element in enumerate(element):
                    sub_element=str(sub_element)
                    if(sub_idx == 1):
                    	textfile.write("CELL COLOR: ")
                    elif(sub_idx == 2):
                    	textfile.write("CELL HEIGHT: ")
                    textfile.write(sub_element.rstrip("\n"))
                    textfile.write("\n")
            	textfile.write("--------------------------------------------------------------\n\n\n")
        textfile.close()

    def is_similar(self, image1, image2):
        """ 
        Function to check the similarity between 2 imgaes
        """
        sub_diff = np.absolute(np.subtract(image1,image2))
        print("Similarity count is",np.sum(np.count_nonzero(sub_diff > self.PIXEL_LEVEL_SIMILARITY_THRES)))
        return (np.sum(np.count_nonzero(sub_diff > self.PIXEL_LEVEL_SIMILARITY_THRES) < self.SIMILARITY_THRES))


    def run_capture(self):
    # Main Loop
    # Here the screen will be captured using ADB and written
    # to to a .png file.  Then it will be re-read using opencv
    # into a numpy array.  This will allow a comparision
    # between the current captured image and the previous one
    # if they are close => end of scroll reached in the inspection
    # device  
        current_time  = str(time.strftime("%d-%m-%Y %H:%M"))
        identical_img = False
        screen_cnt=1
        screen_capture = []
        screen_cap_complete = False
        while(not screen_cap_complete):
            print("Capturing screen {0}, plesae wait".format(screen_cnt))
            self.capture_adb(self.device, screen_cnt)
            inspec_screen = cv2.imread(self.CWD_PATH + '/screen.png')
            inspec_screen_bw = cv2.cvtColor(inspec_screen, cv2.COLOR_BGR2GRAY)
            inspec_screen_crop=inspec_screen_bw[android_eye.CROP_TOP:android_eye.CROP_BOT,android_eye.CROP_LEFT:android_eye.CROP_RIGHT]
            cv2.imwrite(self.IMAGES_PATH + 'crop' + str(screen_cnt) + '.png', inspec_screen_crop)
            inspec_screen_crop_color = inspec_screen[android_eye.CROP_TOP:android_eye.CROP_BOT, :]
            cv2.imwrite(self.LOG_PATH + 'screen' +str(screen_cnt) + '.png', inspec_screen_crop_color)
            page_trunc = self.check_trans_at_topscn(inspec_screen_crop)   # Detect if page is truncated at top => discard first element
            if(screen_cnt!=1):
                if(self.is_similar(inspec_screen_bw[:android_eye.SIM_CROP_BOTTOM, :android_eye.SIM_CROP_RIGHT], # Check if two consecutive screens are the same
                          prev_screen[:android_eye.SIM_CROP_BOTTOM, :android_eye.SIM_CROP_RIGHT])):             # Use greyscale to reduce computation effort
                    file_name = self.LOG_PATH  + "screen" + str(screen_cnt) + '.png'
                    os.remove(file_name)  # Remove the last screen printed out as it will be a duplicate
                    print("Screen Capture Complete")
                    screen_cap_complete = True
                else:     # Note Last Page => Remove Last Capscren entry from output list and prev string dictonary as this corresponds to a potentially truncated segment
                    del_str = screen_capture.pop(-1)[0]
                    self.prev_string_dict.remove(del_str)
            sub_image_lst, trans_pnt = self.find_sub_images(inspec_screen_crop, True)
            color_sub_image = self.find_color_sub_images(inspec_screen_crop_color, trans_pnt)
            screen_capture = self.capture_screen(sub_image_lst, color_sub_image, screen_capture, page_trunc, trans_pnt, screen_cnt)
            prev_screen = np.copy(inspec_screen_bw)
            adjusted_scroll = ((trans_pnt[-2])/inspec_screen_crop.shape[0])*(android_eye.AJ_SC_TUN_SCN)*android_eye.AJ_UNDERSWIPE_FACTOR
            self.swipe_adb(y1=adjusted_scroll)   # Swipe the device to the next screen
            screen_cnt +=1


# Now output the screen descriptors to output files
        screen_capture = self.element_parser(screen_capture, current_time)
        op_file_name = self.LOG_PATH +  "screen_cap" + '.txt'
        self.print_op_file(screen_capture, op_file_name)

