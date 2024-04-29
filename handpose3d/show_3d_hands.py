import numpy as np
import matplotlib.pyplot as plt
from utils import DLT
import os
import cv2
import pandas as pd

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (21, -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts


def visualize_3d(p3ds):
    
    fixed_azi= 30
    fixed_ele = 20

    """Apply coordinate rotations to point z axis as up"""
    Rz = np.array(([[0., -1., 0.],
                    [1.,  0., 0.],
                    [0.,  0., 1.]]))
    Rx = np.array(([[1.,  0.,  0.],
                    [0., -1.,  0.],
                    [0.,  0., -1.]]))

    p3ds_rotated = []
    for frame in p3ds:
        frame_kpts_rotated = []
        for kpt in frame:
            kpt_rotated = Rz @ Rx @ kpt
            frame_kpts_rotated.append(kpt_rotated)
        p3ds_rotated.append(frame_kpts_rotated)

    """this contains 3d points of each frame"""
    p3ds_rotated = np.array(p3ds_rotated)

    """Now visualize in 3D"""
    thumb_f = [[0,1],[1,2],[2,3],[3,4]]
    index_f = [[0,5],[5,6],[6,7],[7,8]]
    middle_f = [[0,9],[9,10],[10,11],[11, 12]]
    ring_f = [[0,13],[13,14],[14,15],[15,16]]
    pinkie_f = [[0,17],[17,18],[18,19],[19,20]]
    fingers = [pinkie_f, ring_f, middle_f, index_f, thumb_f]
    fingers_colors = ['red', 'blue', 'green', 'black', 'orange']

    from mpl_toolkits.mplot3d import Axes3D
    
    if not os.path.exists('figs'):
        os.makedirs('figs')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, kpts3d in enumerate(p3ds_rotated):
        if i%2 == 0: continue #skip every 2nd frame
        for finger, finger_color in zip(fingers, fingers_colors):
            for _c in finger:
                ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = finger_color)

        #draw axes
        ax.plot(xs = [0,5], ys = [0,0], zs = [0,0], linewidth = 2, color = 'red')
        ax.plot(xs = [0,0], ys = [0,5], zs = [0,0], linewidth = 2, color = 'blue')
        ax.plot(xs = [0,0], ys = [0,0], zs = [0,5], linewidth = 2, color = 'black')

        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(0, 100)
        ax.set_xlabel('x')
        ax.set_ylim3d(0, 100)
        ax.set_ylabel('y')
        ax.set_zlim3d(0, 100)
        ax.set_zlabel('z')
        # ax.elev = 0.2*i
        # ax.azim = 0.2*i
        ax.elev = fixed_ele
        ax.azim = fixed_azi
        plt.savefig('figs/fig_' + str(i) + '.png')
        plt.pause(0.01)
        ax.cla()
        
    img_folder = 'figs'
    video_name = 'video.avi'
    
    images = [img for img in os.listdir(img_folder) if img.endswith('.png')]
    frame = cv2.imread(os.path.join(img_folder, images[0]))
    h, w, l = frame.shape
    
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 15, (w, h))
    
    for image in images:
        video.write(cv2.imread(os.path.join(img_folder, image)))
    
    cv2.destroyAllWindows()
    video.release()
        
    
  
def out():
    
    data = np.loadtxt("kpts_3d.csv", dtype='float', delimiter=',', skiprows=1)
    time = np.loadtxt("time.csv", dtype='float', delimiter=',', skiprows=1) 
    #4 is thumb and 8 is index

    # Extract x, y, z coordinates for thumb, index, and base
    x_data = data[:, [0, 4, 8]]
    y_data = data[:, [21, 25, 29]]  
    z_data = data[:, [42, 46, 50]] 
    t_val = time
    
    for i in range(1, len(x_data)):
        if x_data[i, 0] == -1:
            x_data[i, :] = x_data[i-1, :]
            y_data[i, :] = y_data[i-1, :]
            z_data[i, :] = z_data[i-1, :]
            
            
    #x_data = gaussian_filter1d(x_data, sigma=1, axis=0, mode='constant')
    #y_data = gaussian_filter1d(y_data, sigma=1, axis=0, mode='constant')
    #z_data = gaussian_filter1d(z_data, sigma=1, axis=0, mode='constant')
    
    data1 = np.stack([x_data, y_data, z_data], axis = 2)
    disp_ind_thumb = data1[:, 1, :] - data1[:, 2 , :]
    
    norm_disp_ind_thumb = np.linalg.norm(disp_ind_thumb, axis = 1)
    
    
    disp_along_axis = []
    vel = []
    disp = []
    time_diff = []
    
    for i in range(1, len(x_data)):
        time_diff.append((t_val[i] - t_val[i-1]))
        disp_along_axis.append((data1[i, :, :] - data1[i-1, :, :])) 
        disp.append(np.linalg.norm(disp_along_axis[-1], axis = 1))
        vel.append(disp[-1]/time_diff[-1])
    
    vel = np.array(vel)
    vel_df = pd.DataFrame(vel, columns=['cent', 'index', 'thumb'])
    norm_disp_ind_thumb_df = pd.DataFrame(norm_disp_ind_thumb, columns = ['Index to Thumb'])
    
    vel_df.to_csv('velocity.csv', index=False)
    norm_disp_ind_thumb_df.to_csv('dist_index_thumb.csv', index=False)

if __name__ == '__main__':
   
    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)
    out()
    
  