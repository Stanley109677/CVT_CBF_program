#! /usr/bin/env python
import rospy
import matplotlib.pyplot as plt
import numpy as np
from cvt_non_uniform import *
from raspi import *
import time
from transform import *
from controller import *
from geovoronoi.plotting import plot_voronoi_polys_with_points_in_area, _plot_polygon_collection_with_color
import csv

#Coverage control in certain area using Voronoi-based algorithm & Control barrier function for avoiding the collision.
#This script used for experimenting with simulation.
# This is CBF and CVT program (NON_UNIFORM)

N = 13
LOG_DIR = '/home/'

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
        x_traj = np.empty((0, N), float)
        y_traj = np.empty((0, N), float)
        rate = rospy.Rate(100)
        fig = plt.figure(figsize=(8,8))
        fig.subplots_adjust(wspace=0.5, hspace=0, top=0.95, bottom=0.15)
        ax1 = fig.add_subplot(121,projection='scatter_density')
        ax2 = fig.add_subplot(122,projection='scatter_density')
        #ax3 = fig.add_subplot(223,projection='scatter_density')
        ax1.axis('scaled')
        ax2.axis('scaled')
        #ax3.axis('scaled')
        colors = plt.cm.jet(np.linspace(0,1,N))
        norm = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())

        ''' Initialize it first '''
        pose = getposition(N)
        old_centroids = (np.zeros((N, 2))) # N = 10
        coords = np.column_stack((pose[0:2]))
        safety_radius = 1.2
        target = [0, 0]
        iter = 0
        j = 0
        #u_all = []
        #start = time.time()
        #x_traj_2, y_traj_2, t, data = setup_csv(N)

        # Voronoi partition
        (area, poly_shape, poly2pt, new_centroids, new_coords, x_unit, y_unit, ori_centroid) = gen_voronoi_first(coords, target) # Voronoi + density function
        
        # csv
        headers = csv_initial_headers(new_coords)
        desired_headers = csv_initial_headers_2(new_centroids)
        desired_headers_2 = csv_initial_headers_3(new_centroids)
        rows = []
        #rows.append(csv_initial_row(new_coords))
        desired_rows = []
        desired_rows_2 = []
        #desired_rows.append(csv_initial_row(new_centroids))
        list_of_u = []
        iteration = []
        while not rospy.is_shutdown():
            # csv
            row = [j]
            desired_row = [j]
            desired_row_2 = [j]

            plotting_density(fig, ax2, 50, x_unit, y_unit, norm) # from here (putting the plotting of density function in) 
            plotting_voronoi(fig, ax1, iter)
            plt.pause(0.001)
            plt.savefig('src/coverage_control_cbf/src/PNG_nonUniform/FIG_'+str(iter)+'.png')
            plt.savefig('src/coverage_control_cbf/src/EPS_nonUniform/FIG_'+str(iter)+'.eps', format = 'eps')
            ax1.clear()
            ax2.clear()
            
            pose = getposition(N) # three value --> (x,y,z)
            pose_si = uni_to_si_states(pose) # transformer --> (x,y)

            #x_traj_2 = np.append(x_traj_2, pose[0:1], axis=0)
            #y_traj_2 = np.append(y_traj_2, pose[1:2], axis=0)

            coords = np.column_stack((pose[0:2]))

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

            # it slow down, so that target update a certain interval
            if(iter % 8 == 0):
                if(target[1] <= 2.95): # 3.4
                    target[0] += 0.08
                    target[1] += 0.08
                else:
                    target[0] = 3
                    target[1] = 3
                    pass
                    #if target[0] >= 1: # 3.4
                        #end = time.time()
                        #t.append(end - start)
                        #rate.sleep()

                        #data.append(t)
                        #for i in range(N):
                            #data.append(x_traj_2[:, i])
                            #data.append(y_traj_2[:, i])
                        #rospy.signal_shutdown('End of testing')
                        #np.savetxt('src/coverage_control_cbf/src/Data_nonUniform/test.csv', np.column_stack(data), delimiter=' , ', fmt='%s')
                        #pass
            
            print("Target: ", target[0], target[1])
            print("iter: ", iter)
            
            if(iter % 4 == 0):
                (area, poly_shape, poly2pt, new_centroids, new_coords, x_unit, y_unit, ori_centroid) = gen_voronoi_upd(coords, target, ori_centroid) # Voronoi + density function
                #(area, poly_shape, poly2pt, new_centroids, new_coords, ori_centroid) = gen_voronoi_upd(coords, target, ori_centroid) # Voronoi + static density function

                # CVT controller here
                new_coords = cal_tra_fatii_update(new_coords, new_centroids, old_centroids) # already came in to cvt format --> (x,y)\
                old_centroids = new_centroids # the old_centroid must have the value of the previous iteration of the new_centroids
                u_all = cal_fatii_u(new_coords, new_centroids, old_centroids)
                print("u_all; ", u_all)
                #print("new_coords: ", new_coords)
                #print("new_centroids: ", new_centroids)
                # plt.figure(figsize = [8,8])
                # for idx in range(len(u_all)):
                #     plt.plot(iteration, np.array(list_of_u)[:,idx])
                # plt.savefig('/home/robolab/catkin_ws3/src/coverage_control_cbf/src/nonUniformTrajectory/trajectory_plot_' + str(iter) + '.png')
                # print("uuuuuuuu:   ", u_all)
                #ax3.plot(x_unit, u_all[iter], '-', color=colors[iter], label='robot'+str(iter), MarkerSize=4)
                
            for i in range(len(new_coords)): 
                    desired_row.append(u_all[i])
            desired_rows.append(desired_row)
            desired_row = []

            plot_voronoi_polys_with_points_in_area(ax1, area, poly_shape, np.array(coords), poly2pt, voronoi_edgecolor='black', points_color='black', 
                                        points_markersize=12, voronoi_and_points_cmap=None)

            #plot_new_coords(new_coords, ax1) #plot goal of each robot
            plot_sensor_range(coords, ax1, safety_radius)

            x_goal = np.dstack(new_coords)[0] 
            dxi = si_position_controller(pose_si, x_goal)
            dxi = si_barrier_cert(dxi, pose_si, safety_radius)
            dxu = si_to_uni_dyn(dxi, pose) # transformer
            k = set_velocities(N, dxu)
            put_velocities(N, k)
            #end = time.time()
            #t.append(end - start)
            rate.sleep()

            j += 1
            if(j >= 450):
                break
            else:
                iter +=1



    except rospy.ROSInterruptException:
        rospy.signal_shutdown('End of testing')
        pass

with open('src/coverage_control_cbf/src/Data_nonUniform/data.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
    f.close()
  
with open('src/coverage_control_cbf/src/Data_nonUniform/U_data.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(desired_headers) # This method is used when we want to write only a single row at a time in our CSV file.
    f_csv.writerows(desired_rows) # This method is used to write multiple rows at a time. writerows() can be used to write a list of rows. 
    f.close()

with open('src/coverage_control_cbf/src/Data_nonUniform/data_desired.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(desired_headers_2)
    f_csv.writerows(desired_rows_2)
    f.close()
