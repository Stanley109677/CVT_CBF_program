#! /usr/bin/env python
import rospy
import matplotlib.pyplot as plt
import numpy as np
import time
from cvt_uniform import *
from raspi import *
from transform import *
from controller import *
from geovoronoi.plotting import plot_voronoi_polys_with_points_in_area
import csv

#Coverage control in certain area using Voronoi-based algorithm & Control barrier function for avoiding the collision.
#This script used for experimenting with simulation.
# This is CBF and CVT program (UNIFORM)

N = 13
#LOG_DIR = '/src/coverage_control_cbf/src'
def csv_initial_headers(coords):
    headers = ['Time']
    for i, p in enumerate(coords):
        headers.append('The %dth agent_x'%(i+1))
        headers.append('The %dth agent_y'%(i+1))
    return headers

def csv_initial_headers_2(coords):
    headers = ['Time']
    for i, p in enumerate(coords):
        headers.append('The %dth agent'%(i+1))
    return headers

def csv_initial_headers_3(coords):
    headers = ['Time']
    for i, p in enumerate(coords):
        headers.append('The %dth agent_x'%(i+1))
        headers.append('The %dth agent_y'%(i+1))
    return headers

def csv_initial_row(coords):
    row = [0]
    for i in range(len(coords)):
        row.append(coords[i][0])
        row.append(coords[i][1])
    return row


def setup_csv(N):
    x_traj = np.empty((0, N), float)
    y_traj = np.empty((0, N), float)
    t = ['time']
    data = []

    x, y = [], []
    for i in range(N):
        x.append('x_traj_'+str(i))
        y.append('y_traj_'+str(i))
    x_traj = np.append(x_traj, np.array([x]), axis=0)
    y_traj = np.append(y_traj, np.array([y]), axis=0)
    return x_traj, y_traj, t, data

if __name__ == '__main__':
    try:
        rospy.init_node('control_node', anonymous = False)
        rate = rospy.Rate(100)
        fig = plt.figure(figsize=(5.7,5))
        fig.subplots_adjust(wspace=0, hspace=0, top=0.95, bottom=0.15)
        ax = plt.subplot()
        plt.axis('scaled')
        iter = 0
        
        ''' Initialize it first '''
        pose = getposition(N)
        old_centroids = (np.zeros((N, 2))) # N = 10
        coords = np.column_stack((pose[0:2]))
        safety_radius = 1.2
        outer = [(7, 7), (-7, 7), (-7, -7), (7, -7)]
        x_traj, y_traj, t, data = setup_csv(N)
        start = time.time()
        j = 0

        # Voronoi partition
        (area, poly_shape, poly2pt, new_centroids, new_coords) = gen_voronoi_first(coords, outer) # Voronoi + density function

        # csv
        headers = csv_initial_headers(new_coords)
        desired_headers = csv_initial_headers_2(new_centroids)
        desired_headers_2 = csv_initial_headers_3(new_centroids)
        rows = []
        desired_rows = []
        desired_rows_2 = []
        list_of_u = []
        iteration = []

        while not rospy.is_shutdown():
            # csv
            row = [j]
            desired_row = [j]
            desired_row_2 = [j]

            plotting(fig, ax, iter)
            plt.pause(0.001)
            plt.savefig('src/coverage_control_cbf/src/PNG_uniform/FIG_'+str(iter)+'.png')
            plt.savefig('src/coverage_control_cbf/src/EPS_uniform/FIG_'+str(iter)+'.eps', format = 'eps')
            ax.clear()
            pose = getposition(N) # three value --> (x,y,z)
            
            pose_si = uni_to_si_states(pose) # transformer --> (x,y)

            x_traj = np.append(x_traj, pose[0:1], axis=0)
            y_traj = np.append(y_traj, pose[1:2], axis=0)

            # csv
            for i in range(len(coords)): # Trajectory_data 
                #new_coords[i] = new_coords[i][1]
                row.append(coords[i][0])
                row.append(coords[i][1])
            rows.append(row)
            row = []

            for i in range(len(new_centroids)):
                desired_row_2.append(new_centroids[i][0])
                desired_row_2.append(new_centroids[i][1])
            desired_rows_2.append(desired_row_2)
            desired_row_2 = []

            coords = np.column_stack((pose[0:2]))

            norm = np.linalg.norm(np.column_stack((new_coords)) - pose_si)

            (area, poly_shape, poly2pt, new_centroids, new_coords) = gen_voronoi_upd(coords, outer) # Voronoi + density function
            new_coords = cal_tra_fatii_update(new_coords, new_centroids, old_centroids) # already came in to cvt format --> (x,y)\
            old_centroids = new_centroids # the old_centroid must have the value of the previous iteration of the new_centroids
            u_all = cal_fatii_u(new_coords, new_centroids, old_centroids)
            print("uuuuuuuu:   ", u_all)

            for i in range(len(new_coords)): # desired Trajectory_data
                    desired_row.append(u_all[i])
            desired_rows.append(desired_row)
            desired_row = []

            # if norm < 0.01:
            #     end = time.time()
            #     t.append(end - start)
            #     rate.sleep()

            #     data.append(t)
            #     print (len(t))
            #     print (len(x_traj))
            #     for i in range(N):
            #         data.append(x_traj[:, i])
            #         data.append(y_traj[:, i])
            #     rospy.signal_shutdown('End of testing')
            #     np.savetxt('src/coverage_control_cbf/src/Data_Uniform/test.csv', np.column_stack(data), delimiter=' , ', fmt='%s')
            #     pass

            plot_voronoi_polys_with_points_in_area(ax, area, poly_shape, np.array(coords), poly2pt, voronoi_edgecolor='black', points_color='black', 
                                        points_markersize=30, voronoi_and_points_cmap=None)

            plot_new_coords(new_centroids, ax) #plot goal of each robot
            plot_sensor_range(coords, ax, safety_radius)

            x_goal = np.dstack(new_coords)[0] # np.dstack to fit into robotarium controller format from cvt
            dxi = si_position_controller(pose_si, x_goal)
            dxi = si_barrier_cert(dxi, pose_si, safety_radius)
            dxu = si_to_uni_dyn(dxi, pose) # transformer
            k = set_velocities(N, dxu)
            put_velocities(N, k)
            end = time.time()
            t.append(end - start)
            rate.sleep()

            j += 1
            if(j >= 450):
                break
            else:
                iter +=1

    except rospy.ROSInterruptException:
        rospy.signal_shutdown('End of testing')
        pass

with open('src/coverage_control_cbf/src/Data_Uniform/data.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
    f.close()
  
with open('src/coverage_control_cbf/src/Data_Uniform/U_data.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(desired_headers) # This method is used when we want to write only a single row at a time in our CSV file.
    f_csv.writerows(desired_rows) # This method is used to write multiple rows at a time. writerows() can be used to write a list of rows. 
    f.close()

with open('src/coverage_control_cbf/src/Data_Uniform/data_desired.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(desired_headers_2)
    f_csv.writerows(desired_rows_2)
    f.close()