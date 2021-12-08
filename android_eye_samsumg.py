####################################################################
# Android Eye Tool
# Tool interfaes between OpenCV and Android Debug Bridge
# to scroll through a multiple screen industrial device
# capturing the screen shots.
# The device must have the information in bounded cells less
# than the heigth of the device's screen.
# This version of the tool is used as demo on a Samsung Galaxy
# A51.
####################################################################


import cv2
import subprocess
import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
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

################################################################
########### Class to Inferface to ADB
################################################################


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

    def swipe_adb(self, x1=0, y1=790, x2=0, y2=0, dur=1900):#2100 #2000  #5000
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
        with open(self.LOG_PATH + self.fpn_dir + "screen%d.png" % screen_cnt, "wb") as fp:
            fp.write(screen)


################################################################
########### Android Eye Class
################################################################

class android_eye(adb_driver):

    # Constant to define crop points
    CROP_TOP = 800
#   CROP_BOT = 1200
    CROP_BOT = 2000
    CROP_LEFT = 0
    CROP_RIGHT = 960
    # Scroll bar removal crop point
    SIM_CROP_RIGHT = 460
    SIM_CROP_BOTTOM = 800   
    # Adaptive scrolling crop points
    AJ_SC_FULL_SCN = 780
    AJ_SC_TUN_SCN=CROP_BOT-CROP_TOP  #990#1025#970 #753 #755  #773 #776
    AJ_SMALL_SCREEN_FACTOR = 1.02
    BOT_IMAGE_TRUN_THRES = 5 # Ingor transistions very close to the image end
    SIMILARITY_THRES = 8000  # 250 edit
    SEG_SIZE_THRESHOLD = 20
    PIXEL_LEVEL_SIMILARITY_THRES = 30  # 250 edit
    TRUN_SCREEN_THRESHOLD_CK = 30  #15 #  This defines how far we check for a color transition to determine screen trunction
    RHS_MARGIN_THRES = 4
    TIME_CHECK_THRES = 2  # Number of minutes the inspection app time must be within the current time
    IMAGES_PATH_LOG = "images/"
    TESSERACT_PATH_ROOT = "tesseract/"
    LOG_PATH_ROOT = "log/"
    GOLDEN_VEC_ROOT = "golden/"
    # OpenCV BGR Constants
    BGR_PURPLE   = [127,   0, 127]
    BGR_RED      = [  0,   0, 255]
    BGR_GREEN    = [  0, 255,   0]
    BGR_WHITE    = [255, 255, 255]


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



    def __init__(self, test_dir, golden_dir="", lt_offset=0, bcf_elasped=0, fpn=False, ticket_removal=False):
    #   self.device = super().init_adb()   # Init the adb device
        self.device = super().__init__()   # Init the adb device
        self.test_dir = ""         # This will hold the output directtoy
        self.golden_file = ""      # This will hold the golden comparsion directory 
        self.color_lu_dict = self.setup_color_lu_dict()
        self.test_dir = test_dir
        self.golden_dir = golden_dir
        self.TESSERACT_PATH = self.test_dir + "/" + android_eye.TESSERACT_PATH_ROOT
        Path(self.TESSERACT_PATH).mkdir(parents=True, exist_ok=True) 
        self.IMAGES_PATH = self.test_dir + "/" +  android_eye.IMAGES_PATH_LOG
        Path(self.IMAGES_PATH).mkdir(parents=True, exist_ok=True) 
        self.LOG_PATH = self.test_dir + "/" +  android_eye.LOG_PATH_ROOT
        Path(self.LOG_PATH).mkdir(parents=True, exist_ok=True) 
        self.CWD_PATH = os.getcwd()
        self.fpn_dir=""
        if(fpn):
            self.fpn_dir = "fpn"
        self.golden_cmp = False    # This will determine if a comparision is done
        if(golden_dir):
            self.golden_cmp = True
        self.golden_file = self.CWD_PATH + "/" + android_eye.GOLDEN_VEC_ROOT + self.golden_dir + "/screen_cap.txt"
        self.lt_check = False    # This will determine if a comparision is done
        if(lt_offset != 0):
            self.lt_check = True 
            self.lt_offset  = int(lt_offset)
        if(bcf_elasped !=0):
            self.bcf_check = True 
            self.bcf_elasped  = int(bcf_elasped)            
        self.time_window_ck_assert = True   
        self.time_bcf_assert = True 
        self.ticket_removal = ticket_removal



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

    ## Class varible - Color Look-up table
    #color_lu_dict = android_eye.setup_color_lu_dict()

    def find_sub_images(self, image, page_trunc):
        """
      Function to find sub-images from full inpsection app image
      This function will scan down the RHS of the image and find 
      transitions on colors to identify sub-image boundaries.
      The location of the transition points are stored in array trans_pnt
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
        color = self.color_lu_dict.get(key,0)
        return color

    def detrunc_img(self, image, top):
        """
      Function to remove truncated text from inpsection app image
      This function will scan from either  the top ot the base  of the
      image and find the first row with uniform pixels  
      The fuction can work from the top or bottom of the image 
      depending on whether top is set to true or false
        """
        if(top==True): 
            row_idx=0
        else:
            row_idx=image.shape[0]-1
        row = image[row_idx]
        ## Loop thru the image to find the first uniform row
        while (not np.all(row == row[-1])):  
            if top:
                row_idx+=1 
            else:
                row_idx-=1 
            row = image[row_idx]
        detrunc_idx=row_idx
        image[0:detrunc_idx,:] = 0
        return image

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

    def gen_screen_dict(self, inFile, screen_cap, section, color, segement_size):
        """
        Function to tesseract output text file and
        generate the dictonary which contains test 
        and color 
        """
        string = ""
        screen_cap_sec_lst = [[],[],[]]
        for line in inFile:
            if line.strip():
                string += line  
        if(string !=""):
            if (not any(string in x for x in screen_cap)):  ## Dont insert duplicates                   
                    screen_cap_sec_lst[0] = string  
                    screen_cap_sec_lst[1] = color           
                    screen_cap_sec_lst[2] = segement_size
                    screen_cap.append(screen_cap_sec_lst)
        else:                       
                    screen_cap_sec_lst[0] = "Dummy"         ## Identify test free segments with the keyword "Dummy" 
                    screen_cap_sec_lst[1] = self.Color.WHITE            
                    screen_cap_sec_lst[2] = 0
                    screen_cap.append(screen_cap_sec_lst)       
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
                cv2.imwrite(self.IMAGES_PATH + 'sub_image{}.png'.format(count),sub_image)
                cv2.imwrite(self.IMAGES_PATH + 'color_sub_image{}.png'.format(count),color_sub_image[count])   
                img_path = self.IMAGES_PATH + "sub_image" + str(count) + ".png"         
                out_path = self.TESSERACT_PATH + "screen" + str(count)
                tesseract_cmd = "tesseract {0} {1}  --psm 6 quiet".format(img_path,out_path)
                os.system(tesseract_cmd)
                tesseract_file = open(self.TESSERACT_PATH + "screen%d.txt" % count, "r")
                segement_size  = trans_pnt[count+1] -  trans_pnt[count]
                segement_pos   = count
                screen_capture = self.gen_screen_dict(tesseract_file, screen_capture, count, color, segement_size)
        return screen_capture


    def check_trans_at_topscn(self, image): 
        """
        Function to check the transition point on a screen capture - 
        used to determine is a screen is truncated
        """
        #page_trunc=not((image[0:self.TRUN_SCREEN_THRESHOLD_CK,:]==image[0,:]).all())term
        page_trunc=(image[0:self.TRUN_SCREEN_THRESHOLD_CK,image.shape[1]-android_eye.RHS_MARGIN_THRES]==image[0,image.shape[1]-android_eye.RHS_MARGIN_THRES]).all()
        #print(image[0:self.TRUN_SCREEN_THRESHOLD_CK,image.shape[1]-android_eye.RHS_MARGIN_THRES])
        #print(image[0:self.TRUN_SCREEN_THRESHOLD_CK+10,image.shape[1]-android_eye.RHS_MARGIN_THRES])       
        cv2.imwrite(self.IMAGES_PATH + 'tunc_debug.png',image)
        return page_trunc   

    def element_parser(self, screen_capture, current_time):
        """
        Function to parse the screen capture into static and dymanic
        elements. 
        Static elements are not time dependant.
        This was used on the version of the tool I used for wor purposed - not used for the Samsumg Phone version of the tool
        """

        return screen_capture

    def print_op_file(self, screen_capture,op_file_name):
        """
        Function to print the captured screen contents 
        to an output file
        """
        textfile = open(op_file_name, "w")  
        for element in screen_capture:
            if("Dummy" not in element[0]):
                for sub_element in element:
                    sub_element=str(sub_element)
                    textfile.write(sub_element.rstrip("\n"))
                    textfile.write(" ")
                textfile.write("\n")    
        textfile.close()

    def compare_static(self, op_file_name):
        if(self.golden_cmp):    
            if(filecmp.cmp(self.golden_file, op_file_name)):
                print("Test Pass")
            else:
                print("Test Failed")
            text1 = open(self.golden_file).readlines()
            text2 = open(op_file_name).readlines()
            for line in difflib.unified_diff(text1, text2):
                print(line)


    def check_times(self, ref_time, current_time, lt_offset):
        """
        Function to check the curren time against 
        the inspection app time
        This was used on the version of the tool I used for wor purposed - not used for the Samsumg Phone version of the tool
        """
        current_time = datetime.strptime(current_time, '%d-%m-%Y %H:%M')
        ref_time = datetime.strptime(ref_time.lstrip(), '%d-%m-%Y %H:%M')
        time_thres =str(android_eye.TIME_CHECK_THRES)
        tdiff = current_time - ref_time 
        #assert abs(tdiff.total_seconds() - lt_offset*60) < int(time_thres)*60,  "Inspection Time is Outside Permissable Window"
        if((tdiff.total_seconds() - lt_offset*60) > int(time_thres)*60):
            self.time_window_ck_assert = False
    
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
        while(1):
            print("Capturing screen {0}, plesae wait".format(screen_cnt))
            self.capture_adb(self.device, screen_cnt)
            inspec_screen = cv2.imread(self.CWD_PATH + '/screen.png')
            inspec_screen_bw = cv2.cvtColor(inspec_screen, cv2.COLOR_BGR2GRAY)
            inspec_screen_crop=inspec_screen_bw[android_eye.CROP_TOP:android_eye.CROP_BOT,android_eye.CROP_LEFT:android_eye.CROP_RIGHT]
            cv2.imwrite(self.IMAGES_PATH + 'crop_deubg.png', inspec_screen_crop)
            inspec_screen_crop_color = inspec_screen[android_eye.CROP_TOP:android_eye.CROP_BOT, :]
            page_trunc = self.check_trans_at_topscn(inspec_screen_crop)   # Detect if page is truncated => discard first element
            if(screen_cnt!=1):
                cv2.imwrite(self.IMAGES_PATH + 'stop_prev.png', prev_screen[:, :android_eye.SIM_CROP_RIGHT])
                cv2.imwrite(self.IMAGES_PATH + 'stop_curr.png', inspec_screen_bw[:, :android_eye.SIM_CROP_RIGHT])
                if(self.is_similar(inspec_screen_bw[:android_eye.SIM_CROP_BOTTOM, :android_eye.SIM_CROP_RIGHT], prev_screen[:android_eye.SIM_CROP_BOTTOM, :android_eye.SIM_CROP_RIGHT])): # Use greyscale to reduce computation effort
                    file_name = self.LOG_PATH + self.fpn_dir + "screen" + str(screen_cnt) + '.png'
                    os.remove(file_name)  # Remove the last screen printed out as it will be a duplicate
                    print("Screen Capture Complete")
                    break
            # Not Lat Page => Remove Last Capscren entry
                else:
                    del screen_capture[-1]
            sub_image_lst, trans_pnt = self.find_sub_images(inspec_screen_crop, True)
            color_sub_image = self.find_color_sub_images(inspec_screen_crop_color, trans_pnt)
            screen_capture = self.capture_screen(sub_image_lst, color_sub_image, screen_capture, page_trunc, trans_pnt, screen_cnt)
            prev_screen = np.copy(inspec_screen_bw)
            print(trans_pnt[-2], inspec_screen_crop.shape[0])
            if(trans_pnt[-2] > inspec_screen_crop.shape[0]/1.85):
                adjusted_scroll = ((trans_pnt[-2])/inspec_screen_crop.shape[0])*(android_eye.AJ_SC_TUN_SCN)
            else:
                adjusted_scroll = ((trans_pnt[-2])/inspec_screen_crop.shape[0])*(android_eye.AJ_SC_TUN_SCN)*android_eye.AJ_SMALL_SCREEN_FACTOR
            self.swipe_adb(y1=adjusted_scroll)
            screen_cnt +=1


# Now Parse the elements into static and dymanic elements
        screen_capture = self.element_parser(screen_capture, current_time)
        op_file_name = self.LOG_PATH + self.fpn_dir + "screen_cap" + '.txt'
        self.print_op_file(screen_capture, op_file_name)
        self.compare_static(op_file_name)
        assert self.time_window_ck_assert,  "Time check is Outside Permissable Window"