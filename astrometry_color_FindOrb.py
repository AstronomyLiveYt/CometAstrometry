import numpy as np
import math
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io.fits import getheader
import cv2
import os
import sys
import fnmatch

def mouse_click(event, x, y, flags, params):
    global xcoord, ycoord
    if event == cv2.EVENT_LBUTTONDOWN:
        xcoord, ycoord = x,y
        cv2.destroyAllWindows()

if len(sys.argv) != 2:
    print('Proper use: python astrometry.py magnification')
    exit()
pattern = '*.new*'
filelist = os.listdir('.')
mag = float(sys.argv[1])
with open('CometID.txt') as f:
    lines = [line.rstrip('\n') for line in f]
    objid = lines[0].split(': ')[1]
    year = objid.split('/')[1].split(' ')[0]
    if int(year) > 1999:
        year = str('K'+year[2:4])
    elif int(year) > 1899:
        year = str('J'+year[2:4])
    elif int(year) > 1799:
        year = str('I'+year[2:4])
    elif int(year) > 1699:
        year = str('H'+year[2:4])
    desig1 = objid.split('/')[1].split(' ')[1][0]
    desig2 = str(int(objid.split('/')[1].split(' ')[1][1:3])).zfill(2)
    objid2 = str('    ' + objid[0] + year + desig1 + desig2 + '0  C')
    site = lines[1].split(': ')[1]

with open('astrometryoutput.txt', 'a') as s:
    for idx, entry in enumerate(filelist):
        if fnmatch.fnmatch(entry,pattern):
            skip = False
            hdu = fits.open(entry)[0]
            hdu.header['NAXIS'] = 2
            wcs = WCS(hdu)
            hdr = getheader(entry, 0)
            dateobs = hdr['DATE-OBS']
            idate, itime = dateobs.split('T')
            year, month, day = idate.split('-')
            hour, minute, seconds = itime.split(':')
            dayfraction = ((((float(seconds)/60)+int(minute))/60)+int(hour))/24
            day = int(day) + dayfraction
            second, millisecond = seconds.split('.')
            #print(dateobs)
            cv2.namedWindow(entry)
            cv2.setMouseCallback(entry, mouse_click)
            image_file = entry
            #Convert FITS to 8 bit image
            image_data = fits.getdata(entry)
            imagenew = np.array(image_data,dtype = np.float32)
            imagenew = np.moveaxis(imagenew, 0, 2)
            frame_height, frame_width, channels = imagenew.shape
            tonemap = cv2.createTonemapReinhard(1, 0,0,1)
            imagenormalized = tonemap.process(imagenew)
            eightbit =  np.clip(imagenormalized*255, 0, 255).astype('uint8')
            hsvImg = cv2.cvtColor(eightbit,cv2.COLOR_BGR2HSV)
            hsvImg[...,1] = hsvImg[...,1]*1.5
            eightbit=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
            
            cl1 = cv2.resize(eightbit,None,fx=mag, fy=mag, interpolation = cv2.INTER_LINEAR)
            print(entry,end='\r')
            cv2.imshow(entry,cl1)
            k = cv2.waitKey(0)& 0xFF
            if k == ord('s'):
                skip = True
                cv2.destroyAllWindows()
            elif k == ord('q'):
                cv2.destroyAllWindows()
                exit()
            cv2.destroyAllWindows()
            if skip is False:
                roibox = [(int((xcoord/mag)-10),int((ycoord/mag)-10)), (int((xcoord/mag)+10),int((ycoord/mag)+10))]
                imageroi = eightbit[roibox[0][1]:roibox[1][1],roibox[0][0]:roibox[1][0]]
                imageroi = cv2.cvtColor(imageroi, cv2.COLOR_BGR2GRAY)
                imageroi = cv2.GaussianBlur(imageroi, (9, 9), 0)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imageroi)
                newxcoord = (max_loc[0]-10) + (xcoord/mag)
                newycoord = (max_loc[1]-10) + (ycoord/mag)
                eightbit[int(newycoord),int(newxcoord)] = (0,0,255)
                print(newxcoord, newycoord)
                cv2.imshow('bright point',eightbit)
                cv2.waitKey(2000)
                lon, lat = wcs.all_pix2world((newxcoord), (newycoord), 0)
                rhr = math.trunc(lon/15)
                rminute = math.trunc(((lon/15)-rhr)*60)
                rsecond = math.trunc(((((lon/15)-rhr)*60)-rminute)*60)
                rtenths = math.trunc((round((((((lon/15)-rhr)*60)-rminute)*60),3)-rsecond)*1000)
                if rtenths > 999:
                    rtenths = 999
                if lat > 0:
                    sign = '+'
                else:
                    sign = ''
                lat = float(lat)
                lon = float(lon)
                ddegree = math.trunc(lat)
                dminute = abs(math.trunc(((abs(lat))-abs(ddegree))*60))
                dsecond = abs(round(((((abs(lat))-abs(ddegree))*60)-abs(dminute))*60))
                dtenths = abs(math.trunc((round(abs(((((abs(lat))-abs(ddegree))*60)-abs(dminute))*60),2)-dsecond)*100))
                if dtenths > 99:
                    dtenths = 99
                outfile = str(str(objid2)+str(year)+' '+ str(month).zfill(2)+' '+str('{0:.5f}'.format(day)).zfill(8)+' '+str(rhr).zfill(2)+' '+str(rminute).zfill(2)+' '+str(rsecond).zfill(2)+'.'+str(rtenths)[::-1].zfill(3)[::-1]+''+sign+str(ddegree).zfill(2)+' '+str(dminute).zfill(2)+' '+str(dsecond).zfill(2)+'.'+str(dtenths).zfill(2)[::-1]+'                     '+site+'\n')
                s.write(outfile)
