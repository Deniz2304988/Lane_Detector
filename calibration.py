import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mat_img
import os

img = mat_img.imread("C:/Users/user/PycharmProjects/Computer_Vision_Projects/calibration/calibration1.jpg")

def point_extractor(img):
    obj_points=[]
    image_points=[]


    obj_p=np.zeros((6*9,3),np.float32)
    obj_p[:,:2]=np.mgrid[0:9 , 0:6].T.reshape(-1,2)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # RGB2GRAY for matplotlib.image.imread , BGR2GRAY for cv2.imread

    ret, corners = cv2.findChessboardCorners(gray,(9,6))

    if ret==True:
        obj_points.append(obj_p)
        image_points.append(corners)

    new_img=cv2.drawChessboardCorners(gray,(9,6),corners,ret)

    return obj_points,image_points,gray.shape[::-1]

def load_calibration_photos(data_path):
    obj_points=[]
    image_points=[]
    for path,subdirs,files in os.walk(data_path):
        for file in files:
            if file[-4:] == ".jpg" and file != "calibration1.jpg":
                objp,imgp,shape=point_extractor(mat_img.imread(os.path.join(path,file)))
                if objp != []:
                    obj_points.append(objp)
                if imgp != []:
                    image_points.append(imgp)


    return np.squeeze(obj_points),np.squeeze(image_points),shape


DATA_PATH="C:/Users/user/PycharmProjects/Computer_Vision_Projects/calibration"
obj_points,image_points,shape=load_calibration_photos(data_path=DATA_PATH)

ret,mat,dist,rvecs,tvecs = cv2.calibrateCamera(obj_points,image_points,shape,None,None)

dst=cv2.undistort(img,mat,dist,None,mat)

dictionary = {'matrix': mat}
np.save('my_file1.npy', dictionary)
dictionary = {'dist':dist}
np.save('my_file2.npy', dictionary)
plt.imshow(img)
plt.show()
plt.imshow(dst)
plt.show()

