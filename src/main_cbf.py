#! /usr/bin/env python
import rospy
import numpy as np
from raspi import *
from transform import *
from controller import *
import csv
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from shapely import geometry
from shapely.geometry import Polygon
from shapely.geometry import Point
from geovoronoi import coords_to_points, points_to_coords, voronoi_regions_from_coords
from geovoronoi.plotting import plot_voronoi_polys_with_points_in_area, _plot_polygon_collection_with_color

# This script used for experimenting with simulation. (CBF)

def csv_initial_headers(coords):
    headers = ['Time']
    for i, p in enumerate(coords):
        headers.append('The %dth agent_x'%(i+1))
        headers.append('The %dth agent_y'%(i+1))
    return headers

def csv_initial_row(coords):
    row = [0]
    for i in range(len(coords)):
        row.append(coords[i][0]) # row: i / column: 0
        row.append(coords[i][1]) # row: i / column: 1
        #row.append(coords[0][i])
        #row.append(coords[1][i])
    return row

N = 4
j = 0 # csv and gif related

if __name__ == '__main__':
    try:
        rospy.init_node('control_node', anonymous = False)
        radius = 1.5
        xybound = radius*np.array([-1, 1, -1, 1])
        p_theta = 2*np.pi*(np.arange(0, 2*N, 2)/(2*N))
        p_circ = np.vstack([
            np.hstack([xybound[1]*np.cos(p_theta), xybound[1]*np.cos(p_theta+np.pi)]),
            np.hstack([xybound[3]*np.sin(p_theta), xybound[3]*np.sin(p_theta+np.pi)])
            ])


        ######################## Plotting ########################
        #plotting
        fig = plt.figure(figsize=(6,5))
        fig.subplots_adjust(wspace=0.3, hspace=0, top=0.9, bottom=0.2)
        ax = plt.subplot()
        major_locator=MultipleLocator(1)
        ax.xaxis.set_major_locator(major_locator)
        ax.yaxis.set_major_locator(major_locator)
        #ax.set_xlim([0,400])
        #ax.set_ylim([0,400])
        #date_x_label = ['0', '0', '100', '200', '300', '400']
        #date_y_label = ['0', '0', '100', '200', '300', '400']
        #font2 = {'fontsize': 20}
        #ax.set_xticklabels(date_x_label)
        #ax.set_yticklabels(date_y_label)
        font = {'size':15}
        ax.set_xlabel('Coordinate X', font, labelpad=15, fontweight = 'bold')
        ax.set_ylabel('Coordinate Y', font, labelpad=15, fontweight = 'bold')
        plt.axis('scaled')

            ###plot
        #ax.set_xlim([-0.3,0.3])  # here
        #ax.set_ylim([-0.3,0.3])  # here
        ax.set_xlim([-3,3])
        ax.set_ylim([-3,3])
        ax.tick_params(labelsize=18)

        #ax.scatter(dxu[0, :], dxu[1, :], color = 'black', s= 30, marker='o', zorder =20)
        plt.tick_params(labelsize=18) #设置刻度字体大小
        plt.pause(0.001)
        ax.clear()

        ###CSV
        #headers = csv_initial_headers(x) # trajectory 
        rows = []
        #rows.append(csv_initial_row(x))

        new_array = np.ndarray(shape=(2,4), dtype=float)
        new_array_2 = np.ndarray(shape=(4,2), dtype=float)

        flag = 0
        x_goal = p_circ[:, :N] 
        while not rospy.is_shutdown():
            ###plot
            #ax.set_xlim([-0.3,0.3]) # here
            #ax.set_ylim([-0.3,0.3]) # here
            ax.set_xlim([-3,3])
            ax.set_ylim([-3,3])
            ax.xaxis.set_major_locator(major_locator)
            ax.yaxis.set_major_locator(major_locator)
            #ax.set_xticklabels(date_x_label, font2)
            #ax.set_yticklabels(date_y_label, font2)
            ax.set_xlabel('Coordinate X', font, labelpad=15, fontweight = 'bold')
            ax.set_ylabel('Coordinate Y', font, labelpad=15, fontweight = 'bold')

            # csv
            row = [j]

            pose = getposition(N) # get the position of the robots in gazebo
            coords = np.dstack((pose[0], pose[1]))[0]
            print (np.column_stack((pose[0:2])))

            # csv
            headers = csv_initial_headers(coords)

            for i in range(len(coords)):  
                row.append(coords[i][0])
                row.append(coords[i][1])
            rows.append(row)
            row = []
            ####
            pose_si = uni_to_si_states(pose)

            ax.scatter(pose_si[0][:], pose_si[1][:], color = 'black', s= 30, marker='o', zorder =20)

            new_array = pose_si
            pose_si[0][0]
            pose_si[0][0]
            pose_si[0][0]
            pose_si[0][0]

            pose_si[1][0]
            pose_si[1][0]
            pose_si[1][0]
            pose_si[1][0]

            for i in range(N):
                new_array_2[i][0] = pose_si[0][i]
                new_array_2[i][1] = pose_si[1][i]

            sensor_range = 0.58
            sensor_region = []
            for coord in list(new_array_2):
                circ = geometry.Point(coord[0], coord[1]).buffer(sensor_range, cap_style=1)
                sensor_region.append(circ)
            
            _plot_polygon_collection_with_color(ax, sensor_region, color='red', alpha=0.3, zorder=10)

            if(np.linalg.norm(x_goal - pose_si) < 0.05):
                flag = 1-flag

            if(flag == 0):
                x_goal = p_circ[:, :N]
            else:
                x_goal = p_circ[:, N:]


            dxi = si_position_controller(pose_si, x_goal)
            dxi = si_barrier_cert(dxi, pose_si, 1.2)
            dxu = si_to_uni_dyn(dxi, pose)
            k = set_velocities(N, dxu)
            print("k: ", k)
            ####
            put_velocities(N, k) # go into the gazebo robot velocity

            plt.savefig('src/raspberrypimouse_controlbarrierfunction/scripts/PNG_CBF/FIG_'+str(j)+'.png')
            plt.savefig('src/raspberrypimouse_controlbarrierfunction/scripts/EPS_CBF/FIG_'+str(j)+'.eps', format = 'eps')
            plt.tick_params(labelsize=18) #设置刻度字体大小
            plt.pause(0.001)
            ax.clear()

            if j == 500:
                break
            else:
                j += 1

    except rospy.ROSInterruptException:
        rospy.signal_shutdown('End of testing')
        pass

with open('src/raspberrypimouse_controlbarrierfunction/scripts/Data_CBF/data.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
    f.close()