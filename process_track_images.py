#---------------------------------------------------------------
# TOP LEVEL CODE
#---------------------------------------------------------------
# PROCESS_TRACK_IMAGES
#
# Description
# General program to read MODIS reflectance and shiptrack hand-logged
# files with some basic plots to illustrate where ship tracks are located
# in the satellite granule. Uses PyTroll to plot/read the MODIS image.
#
# Notes: this code currently crops 250 pixels off the edge of each
#        image. Ship tracks are logged based on the lower left corner
#        starting at x=0 and y=0. The ML algorithm requires the top-left
#        corner to start at x=0 and y=0 therefore we tranform the y-coord
#        for bounding box locations.
#
# Output
# 1) Plots of NIR composite images for each MODIS granule
#       path ---> /images
# 2) Plots of the bounding boxes (used as a sanity check)
#       path ---> /images_bbox/
# 3) Text File: contains the bounding boxes according to DIGITS
#       path --> /labels/
#    (FORMAT: see 
#    https://github.com/NVIDIA/DIGITS/blob/master/digits/extensions/data/objectDetection/README.md)
#
# Example
# python2.7 -i process_track_images.py
#
# History
# 11/12/18, MC: upload initial version of the code to the repo
#---------------------------------------------------------------

#---------------------------------------------------------------
# Libraries
#---------------------------------------------------------------
from subroutines_track_images import *
import datetime
import numpy as np
import os,sys,glob
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
import numpy.ma as ma
import pdb
from satpy import Scene
from satpy.composites import GenericCompositor
from satpy.writers import to_image
from pyhdf.SD import SD, SDC

#Crop 250 pixels off each edge of the MODIS granule
cropped_area = 250

#---------------------------------------------------------------
# Paths
#---------------------------------------------------------------
#Ship Track Hand-logged Files
path_track_root = '/group_workspaces/jasmin2/aopp/mchristensen/shiptrack/shiptrack_logged_files/combined/'

#MODIS Files
path_modis_root = '/group_workspaces/cems2/nceo_generic/satellite_data/modis_c61/'

#Name of the experiment (here we are calling it crop_partial)
expName = 'crop'
expName = 'crop_partial'

#Output Path (user-defined)
root_output = '/group_workspaces/jasmin2/acpc/public/mchristensen/shiptrack/machine_learning/'

#images directory
path_output_images = root_output+'/'+expName+'/images/'
ferr = file_mkdir(path_output_images)

#bounding box directory
path_output_images_bbox = root_output+'/'+expName+'/images_bbox/'
ferr = file_mkdir(path_output_images_bbox)

#create labels based on full extent of bounding box
path_output_labels = root_output+'/'+expName+'/labels/'
ferr = file_mkdir(path_output_labels)

#---------------------------------------------------------------
# Fetch Ship Track Files
#---------------------------------------------------------------
trackfiles = file_search_tracks( path_track_root )
tfiles = trackfiles['tfiles']
lfiles = trackfiles['lfiles']
fct = len(tfiles)


# Loop over each track file
CT = 0
for i in range(fct):

    # Read track lat/lon locations
    fileInfo = os.path.split(lfiles[i])
    lfilename = fileInfo[0]+'/'+fileInfo[1]
    track_geo = read_osu_shiptrack_file(lfilename)

    # Read track locations from file
    fileInfo = os.path.split(tfiles[i])
    tfilepath = fileInfo[0]
    tfile = fileInfo[1]
    tfilename = tfilepath+'/'+tfile
    track_points = read_osu_shiptrack_file(tfilename)
    
    print(tfilename)

    # Fetch corresponding MODIS granule
    mtype=''
    if 'MOD' in tfile:
        mtype = 'mod'
    if 'MYD' in tfile:
        mtype = 'myd'

    ed=len(tfile)
    YYYY = tfile[ed-16:ed-12]
    DDD  = tfile[ed-12:ed-9]
    HHHH = tfile[ed-8:ed-4]

    print(tfile,'  ',mtype,'  ',YYYY,'  ',DDD,'  ',HHHH)   
    # Now fetch MODIS file for selected instrument and time
    file02 = glob.glob( path_modis_root + mtype+'021km/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf') # calibrated radiances
    file03 = glob.glob( path_modis_root + mtype+'03/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf')   # geolocation
    file04 = glob.glob( path_modis_root + mtype+'04_l2/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf') # aerosol
    file06 = glob.glob( path_modis_root + mtype+'06_l2/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf') # cloud

    # All MODIS L2 files must exist to proceed
    if len(file02) > 0 and len(file03) > 0 and len(file04) > 0 and len(file06) > 0:
        file02 = file02[0]
        file03 = file03[0]
        file04 = file04[0]
        file06 = file06[0]

        # Read MODIS Attribute Data (to extract size of image)
        file = SD(file02, SDC.READ)
        sds_obj = file.select('EV_500_Aggr1km_RefSB')
        xN = int((sds_obj.dimensions(0))['Max_EV_frames:MODIS_SWATH_Type_L1B'])
        yN = int((sds_obj.dimensions(0))['10*nscans:MODIS_SWATH_Type_L1B'])

        #Calculate bounding boxes
        #trainData = track_to_bbox(track_points,xN,yN,cropped_area,'full')
        trainData = track_to_bbox(track_points,xN,yN,cropped_area,'partial')

        #Extract flags from trainData (array of dictionaries)
        flags = []
        for iTRK in range(len(trainData)):
            flags.append( (trainData[iTRK])['flag'] )

        #Process image and label files if at least 1 ship track satisfies the cropped region condition
        if np.sum(flags) < len(trainData):

            #Create image
            pngFile = path_output_images + str(CT).zfill(4)+'.png'
            if os.path.isfile(pngFile):
                print('exists: '+pngFile)
            else:
                global_scene = (Scene(reader="hdfeos_l1b", filenames=[file02,file03]))
                global_scene.load(['1','20','32'], resolution=1000)
                global_scene = global_scene[0:yN,cropped_area:xN-cropped_area]
                compositor = GenericCompositor("rgb")
                composite = compositor([global_scene['1'],global_scene['20'],global_scene['32']])
                img = to_image(composite)
                img.stretch_hist_equalize("linear")
                img.save(pngFile)
                print(pngFile)

            #Create labels - valid (within cropped area) training data only
            txtFile = path_output_labels + str(CT).zfill(4)+'.txt'
            text_file = open(txtFile, "w")
            for iTRK in range(len(trainData)):
                if (trainData[iTRK])['flag'] == 0:
                    text_file.write((trainData[iTRK])['type']+' '+
                                    (trainData[iTRK])['truncated']+' '+
                                    (trainData[iTRK])['occluded']+' '+
                                    (trainData[iTRK])['alpha']+' '+
                                    (trainData[iTRK])['bbox_left']+' '+
                                    (trainData[iTRK])['bbox_top']+' '+
                                    (trainData[iTRK])['bbox_right']+' '+
                                    (trainData[iTRK])['bbox_bottom']+' '+
                                    (trainData[iTRK])['dimensions_height']+' '+
                                    (trainData[iTRK])['dimensions_width']+' '+
                                    (trainData[iTRK])['dimensions_length']+' '+
                                    (trainData[iTRK])['location_x']+' '+
                                    (trainData[iTRK])['location_y']+' '+
                                    (trainData[iTRK])['location_z']+' '+
                                    (trainData[iTRK])['rotation_y']+' '+
                                    (trainData[iTRK])['score']+"\n")
            text_file.close()

            #read bounding boxes
            text_file = open(txtFile, "r")
            lines = text_file.readlines()
            text_file.close()
            pngFileBBox = path_output_images_bbox + str(CT).zfill(4)+'.png'
            if os.path.isfile(pngFileBBox):
                print('exists: '+pngFileBBox)
            else:

                #plot bounding boxes over the top of the satellite image
                pil_im = img.pil_image() #convert XRIMAGE to PIL
                draw = ImageDraw.Draw(pil_im)

                #loop over each ship track
                for j in range(len(lines)):
                    line = lines[j].split()
                    xL = int(line[4])
                    yT = int(line[5])
                    xR = int(line[6])
                    yB = int(line[7])

                    #draw bounding boxes over image
                    truncateFlag = int(line[1])
                    if truncateFlag == 0:
                        col = "black"
                    if truncateFlag == 1:
                        col = "white"

                    print(xL,yT,xR,yB,col)
                    draw.line((xL,yT, xR, yT), fill=col, width=5)
                    draw.line((xL,yB, xR, yB), fill=col, width=5)
                    draw.line((xL,yB, xL, yT), fill=col, width=5)
                    draw.line((xR,yB, xR, yT), fill=col, width=5)
                pil_im.save(pngFileBBox)

            CT = CT+1
        else:
            print('outside of cropped area')

    else:
        print('missing: ',mtype+'021km/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf')

#tarball directory for easier downloading
os.system("tar -czvf "+root_output+expName+".tar.gz"+" "+root_output+expName+"/")
