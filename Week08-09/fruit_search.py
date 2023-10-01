# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import matplotlib.pyplot as plt

#from Practical03_Support.Obstacle import *
# from Practical03_Support.path_animation import *

#import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['apple', 'pear', 'lemon']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('M4_prac_shopping_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def get_distance_to_goal(waypoint, robot_pose):
    # calc distance to waypoint
    x_goal, y_goal = waypoint
    try:
        x, y, theta = robot_pose
    except:
        x,y = robot_pose
    x_diff = x_goal - x
    y_diff = y_goal - y
    distance_to_goal = np.hypot(x_diff, y_diff)
    return distance_to_goal

def get_desired_heading(waypoint, robot_pose):
    # calc desired heading to waypoint (we will clamp between -np.pi and np.pi)
    x_goal, y_goal = waypoint
    x, y, theta = robot_pose
    x_diff = x_goal - x
    y_diff = y_goal - y
    angle_to_goal = np.arctan2(y_diff,x_diff) - theta
    desired_heading = (angle_to_goal+np.pi) % (2*np.pi) + (-np.pi)
    return desired_heading

def call_umeyama(detected_lms, aruco_true_pos):
    if (len(detected_lms)) > 1:
        detected_lms_pos = np.zeros([len(detected_lms), 2])
        aruco_poses = np.zeros([len(detected_lms), 2])
        for i in range(len(detected_lms_pos)):
            #print(detected_lms_pos)
            detected_lms_pos[i] = [detected_lms[i].position[0],detected_lms[i].position[1]]
            tag = detected_lms[i].tag-1
            aruco_poses[i] = aruco_true_pos[tag]
        R, t = ekf.umeyama(detected_lms_pos.T, aruco_poses.T)
        robot_state = ekf.get_state_vector()
        x, y, theta = robot_state[0:3]
        new_robot_pose = R @ np.array([x,y]).reshape(2,1) + t
        robot_state[0:3] = [new_robot_pose[0], new_robot_pose[1], theta]
        print(robot_state[0:3])
        ekf.set_state_vector(robot_state)

# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose, aruco_true_pos):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    #baseline = np.loadtxt(fileB, delimiter=',')
    baseline = 1.703317591322501832e-01
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    wheel_vel = 30 # tick
    wheel_w = 5
    K_pv = 1
    K_pw = 1
    
    # get distance to waypoint
    distance_to_goal = get_distance_to_goal(waypoint, robot_pose)
    # print("Distance_to_goal:")
    # print(distance_to_goal)
    #print("Robot_pose:")
    #print(robot_pose)

    # get desired heading
    desired_heading = get_desired_heading(waypoint, robot_pose)
    # print("Desired_heading:")
    # print(desired_heading)


    rot_threshold = 0.05 # might want to decrease for better acc
    dist_threshold = 0.05

    #while abs(orientation_diff) > rot_threshold:
    w_k = K_pw*desired_heading
    if desired_heading <0:
        r = -1
    else:
        r = 1
    turn_time =  abs(baseline*np.pi/(scale*wheel_w)*((w_k)/(2*np.pi)))
    # print("Turn time: ")
    # print(turn_time)
    lv, rv = ppi.set_velocity([0, r], turning_tick=wheel_w, time=turn_time)
    # print('left-right v:',lv,rv)
    
    drive_meas = measure.Drive(lv, rv,turn_time, 1, 1)
    ekf.predict(drive_meas)
    robot_pose = get_robot_pose(aruco_true_pos)
    # print("Robot_pose:")
    # print(robot_pose)
    #desired_heading = get_desired_heading(waypoint, robot_pose)
    orientation_diff = desired_heading - robot_pose[2]
    # print(orientation_diff)
    ppi.set_velocity([0,0])
        
           
    #while distance_to_goal > dist_threshold:
    v_k = K_pv*distance_to_goal
    drive_time = v_k/(scale*wheel_vel)
    #print(v_k, scale, wheel_vel, waypoint, robot_pose)
    #print("Drive time: ")
    #print(drive_time)
    lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    drive_meas = measure.Drive(lv, rv,drive_time, 1, 1)
    ekf.predict(drive_meas)
    robot_pose = get_robot_pose(aruco_true_pos)
    # print("Robot_pose:")
    # print(robot_pose)
    distance_to_goal = get_distance_to_goal(waypoint, robot_pose)
    ppi.set_velocity([0,0])
    #print("Distance_to_goal:")
    #print(distance_to_goal)


    # turn towards the waypoint
    #phi = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0])

    '''if phi-robot_pose[2] < 0 or phi < 0:
        phi += 2*np.pi-robot_pose[2]
    else:'''
    #phi -= robot_pose[2]
    #phi = (phi+2*np.pi)%(2*np.pi)
    #if phi <0:
       # r = -1
    #else:
        #r = 1
    # refer to wheel calibration equation
    #print(robot_pose[2])
    #print(phi)
    #turnt = baseline*np.pi/(scale*wheel_vel)*((phi)/(2*np.pi))
    #turn_time = abs(turnt) # replace with your calculation
    #print(turn_time)
    #print("Turning for {:.2f} seconds".format(turn_time))
    
    #lv, rv = ppi.set_velocity([0, r], turning_tick=wheel_vel, time=turn_time)
    #drive_meas = measure.Drive(lv, rv,turn_time, 1, 1)
    #ekf.predict(drive_meas)
    #robot_pose = get_robot_pose(aruco_true_pos)
    
    # after turning, drive straight to the waypoint
    #x_dist = waypoint[0]-robot_pose[0]
    #y_dist = waypoint[1]-robot_pose[1]
    #dist = np.sqrt(x_dist**2 + y_dist**2)
    #drivet = dist/(scale*wheel_vel)
    #drive_time = drivet[0] # replace with your calculation
    #print(drive_time)
    #print("Driving for {:.2f} seconds".format(drive_time))
    #lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    #drive_meas = measure.Drive(lv, rv,drive_time, 1, 1)
    #ekf.predict(drive_meas)
    #robot_pose = get_robot_pose(aruco_true_pos)
    #print(ekf.get_state_vector())
    ####################################################

    # print("Arrived at [{}, {}]".format(robot_pose[0], robot_pose[1]))

    return robot_pose


def get_robot_pose(aruco_true_pos,from_true_map=False):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    wheel_vel = 10 # tick

    #print(robot_pose)
    
    
    # refer to wheel calibration equation
    #for i in range(7):
        #turnt = baseline*np.pi/(scale*wheel_vel)*((np.pi/7)/(2*np.pi))
        #turn_time = turnt # replace with your calculation
    #print("Turning for {:.2f} seconds".format(turn_time[0]))
    
        #lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
        #drive_meas = measure.Drive(lv, rv,turn_time, 1, 1)
        #ekf.predict(drive_meas)
    #lms = []
    '''lms, aruco_img = aruco_det.detect_marker_positions(ppi.get_image())
        for i in range(len(lms)):
            lms[i].position = aruco_true_pos[lms[i].tag-1].reshape(2,1)''' 
    #for i in range(len(aruco_true_pos)):
        #print(aruco_true_pos[i])
        #new_marker = measure.Marker(aruco_true_pos[i].reshape(2,1),i+1)
        #lms.append(new_marker)
    #ekf.add_landmarks(lms)
    image = ppi.get_image()
    lms, aruco_img = aruco_det.detect_marker_positions(image)
    # call_umeyama(lms, aruco_true_pos) -> still needs work
    
    
    if not from_true_map:
        for i in range(len(lms)):
            lms[i].position = aruco_true_pos[lms[i].tag-1].reshape(2,1) 
        ekf.add_landmarks(lms)
        ekf.update(lms)
    else:
        lms = []
        for i in range(len(aruco_true_pos)):
            #print(aruco_true_pos[i])
            new_marker = measure.Marker(aruco_true_pos[i].reshape(2,1),i+1)
            lms.append(new_marker)
            ekf.add_landmarks(lms)
            ekf.update(lms)
       
    pose = ekf.get_state_vector()[0:3]
    #print(pose)
    '''if pose[2] > 2*np.pi:
        pose[2] -= 2*np.pi'''
    #pose[2] = (pose[2]+2*np.pi)%(2*np.pi)
    #print(pose)
    
    # update the robot pose [x,y,theta]
    robot_pose = [pose[0][0], pose[1][0], pose[2][0]] # return as a flattened array or will get errors in drive to point bc of numpy arrays and scalars
    ####################################################

    return robot_pose

def find_obs_wrt_goal(goal_next, goal_list, obstacle_list, obs_radius,checking = False): # this is to ensure the goal is not in the obstacle list
    all_obstacles = []
    all_obs_coord = []
    # print('goalnext:',goal_next)
    goal_x = goal_next[0]
    goal_y = goal_next[1]

    for obs in obstacle_list:
        obs_x = obs['x'] 
        obs_y = obs['y'] 
        # print(obs_x,obs_y)
        all_obstacles.append(Circle(obs_x, obs_y, obs_radius))
        all_obs_coord.append([obs_x, obs_y])
        
    for obs in goal_list:
        obs_x = obs['x'] 
        obs_y = obs['y'] 
        if abs(goal_x - obs_x) < 0.05 and abs(goal_y - obs_y) < 0.05:
            continue
        else: 
            all_obstacles.append(Circle(obs_x, obs_y, obs_radius))
            all_obs_coord.append([obs_x, obs_y])
            # print('goal:',[obs_x, obs_y])
    if checking:
        print('all_obs_coord:', all_obs_coord)
    return all_obstacles # with respect to the next goal

def plot_path(est,obs, name, pose=None, count = None):
    if not os.path.exists('path_vis'):
        os.makedirs('path_vis')
    plot = plt.figure()
    a = [x[0] for x in est]
    a1 = [x[1] for x in est]
    plt.plot(a, a1,'-o')
    b = [x[0] for x in obs]
    b1 = [x[1] for x in obs]
    plt.scatter(b, b1,c='r')
    if pose != None:
        # print('plotting pose')
        c = [x[0] for x in pose]
        c1 = [x[1] for x in pose]
        plt.plot(c,c1,'-x',c='y')
    ax = plt.gca()
    ax.invert_yaxis()
    ax.invert_xaxis()
    

    plt.savefig(f'path_vis/{name}{count}.png') 

# This is an adapted version of the RRT implementation done by Atsushi Sakai (@Atsushi_twi)
class RRTC:
    """
    Class for RRT planning
    """
    class Node:
        """
        RRT Node
        """
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start=np.zeros(2),
                 goal=np.array([120,90]),
                 obstacle_list=None,
                 width = 160,
                 height=100,
                 expand_dis=3.0, 
                 path_resolution=0.5, 
                 max_points=200):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacle_list: list of obstacle objects
        width, height: search area
        expand_dis: min distance between random node and closest node in rrt to it
        path_resolion: step size to considered when looking for node to expand
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.width = width
        self.height = height
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_nodes = max_points
        self.obstacle_list = obstacle_list
        self.start_node_list = [] # Tree from start
        self.end_node_list = [] # Tree from end
        
    def planning(self):
        """
        rrt path planning
        """
        self.start_node_list = [self.start]
        self.end_node_list = [self.end]
        while len(self.start_node_list) + len(self.end_node_list) <= self.max_nodes:

        #TODO: Complete the planning method ----------------------------------------------------------------

            ## 1 - Sample random node and add to "start" tree
            rnd_node = self.get_random_node() # sample random node

            expansion_ind = self.get_nearest_node_index(self.start_node_list, rnd_node) # get nearest node in start tree to random node
            expansion_node = self.start_node_list[expansion_ind]

            new_node = self.steer(expansion_node, rnd_node, self.expand_dis) # find node closer to random node - MAKE SURE TO SPECIFY EXPANSION DISTANCE

            #! ######
            # print('found new node:', [new_node.x,new_node.y])
            
            if self.is_collision_free(new_node) or len(self.start_node_list)<2:
                self.start_node_list.append(new_node) # if collision free add to start list
                
                ## 2 - Using the new node added to start tree, add a node to "end" tree in this direction (follow pseudocode)
                end_exp_ind = self.get_nearest_node_index(self.end_node_list, new_node) 
                end_exp_node = self.end_node_list[end_exp_ind]

                new_end_node = self.steer(end_exp_node, new_node,self.expand_dis)

                if self.is_collision_free(new_end_node):
                    self.end_node_list.append(new_end_node)

                ## 3 - Check if we can merge
                closest_ind = self.get_nearest_node_index(self.end_node_list, new_node)
                closest_node = self.end_node_list[closest_ind]
                d, _ = self.calc_distance_and_angle(closest_node, new_node)
                
                #! ######
                # print('distance:',d)
                
                if d < self.expand_dis:
                    self.end_node_list.append(new_node)
                    self.start_node_list.append(closest_node)
                    return self.generate_final_course(len(self.start_node_list) - 1, len(self.end_node_list) - 1)

                ## 4 - Merge trees
                self.start_node_list, self.end_node_list = self.end_node_list, self.start_node_list
            
        return None  # cannot find path
    
    # ------------------------------DO NOT change helper methods below ----------------------------
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Given two nodes from_node, to_node, this method returns a node new_node such that new_node 
        is “closer” to to_node than from_node is.
        """
        
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        # How many intermediate positions are considered between from_node and to_node
        n_expand = math.floor(extend_length / self.path_resolution)

        # Compute all intermediate positions
        for _ in range(n_expand):
            new_node.x += self.path_resolution * cos_theta
            new_node.y += self.path_resolution * sin_theta
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node

        return new_node

    def is_collision_free(self, new_node):
        """
        Determine if nearby_node (new_node) is in the collision-free space.
        """
        if new_node is None:
            return True
        
        points = np.vstack((new_node.path_x, new_node.path_y)).T
        for obs in self.obstacle_list:
            in_collision = obs.is_in_collision_with_points(points)
            if in_collision:
                return False
        
        return True  # safe
    
    def generate_final_course(self, start_mid_point, end_mid_point):
        """
        Reconstruct path from start to end node
        """
        # First half
        node = self.start_node_list[start_mid_point]
        path = []
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        
        # Other half
        node = self.end_node_list[end_mid_point]
        path = path[::-1]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        x = self.width * np.random.random_sample()
        y = self.height * np.random.random_sample()
        rnd = self.Node(x, y)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):        
        # Compute Euclidean disteance between rnd_node and all nodes in tree
        # Return index of closest element
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta        

def rotation(waypoint,robot_pose,aruco_true_pos):
        # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    #baseline = np.loadtxt(fileB, delimiter=',')
    baseline = 1.703317591322501832e-01
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    wheel_vel = 30 # tick
    wheel_w = 5
    K_pv = 1
    K_pw = 1
    
    # get distance to waypoint
    distance_to_goal = get_distance_to_goal(waypoint, robot_pose)
    # print("Distance_to_goal:")
    # print(distance_to_goal)
    #print("Robot_pose:")
    #print(robot_pose)

    # get desired heading
    desired_heading = get_desired_heading(waypoint, robot_pose)
    # print("Desired_heading:")
    # print(desired_heading)


    rot_threshold = 0.05 # might want to decrease for better acc
    dist_threshold = 0.05

    #while abs(orientation_diff) > rot_threshold:
    w_k = K_pw*desired_heading
    if desired_heading <0:
        r = -1
    else:
        r = 1
    turn_time =  abs(baseline*np.pi/(scale*wheel_w)*((w_k)/(2*np.pi)))
    # print("Turn time: ")
    # print(turn_time)
    lv, rv = ppi.set_velocity([0, r], turning_tick=wheel_w, time=turn_time)
    # print('left-right v:',lv,rv)
    
    drive_meas = measure.Drive(lv, rv,turn_time, 1, 1)
    ekf.predict(drive_meas)
    robot_pose = get_robot_pose(aruco_true_pos)
    # print("Robot_pose:")
    # print(robot_pose)
    #desired_heading = get_desired_heading(waypoint, robot_pose)
    orientation_diff = desired_heading - robot_pose[2]
    # print(orientation_diff)
    ppi.set_velocity([0,0])
    
    return robot_pose
    
def rotation360(robot_pose,aruco_true_pos):
    print('START Rotating 360')
    # robot_pose = rotation([0,0.1],robot_pose,aruco_true_pos)
    # robot_pose = rotation([-0.1,0],robot_pose,aruco_true_pos)
    # robot_pose = rotation([0,-0.1],robot_pose,aruco_true_pos)
    # robot_pose = rotation([0.1,0],robot_pose,aruco_true_pos)
    # robot_pose = rotation([0.1,0],robot_pose,aruco_true_pos)
    
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    #baseline = np.loadtxt(fileB, delimiter=',')
    baseline = 1.703317591322501832e-01
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    wheel_vel = 30 # tick
    wheel_w = 30
    K_pv = 1
    K_pw = 1
    
    # get distance to waypoint
    distance_to_goal = get_distance_to_goal(waypoint, robot_pose)
    # print("Distance_to_goal:")
    # print(distance_to_goal)
    #print("Robot_pose:")
    #print(robot_pose)

    # get desired heading
    desired_heading = get_desired_heading(waypoint, robot_pose)
    # print("Desired_heading:")
    # print(desired_heading)


    rot_threshold = 0.05 # might want to decrease for better acc
    dist_threshold = 0.05

    #while abs(orientation_diff) > rot_threshold:
    w_k = K_pw*desired_heading
    if desired_heading <0:
        r = -1
    else:
        r = 1
    turn_time =  abs(baseline*np.pi/(scale*wheel_w)*((w_k)/(2*np.pi)))
    # print("Turn time: ")
    # print(turn_time)
    lv, rv = ppi.set_velocity([0, r], turning_tick=wheel_w, time=4)
    # print('left-right v:',lv,rv)
    
    drive_meas = measure.Drive(lv, rv,turn_time, 1, 1)
    ekf.predict(drive_meas)
    robot_pose = get_robot_pose(aruco_true_pos)
    # print("Robot_pose:")
    # print(robot_pose)
    #desired_heading = get_desired_heading(waypoint, robot_pose)
    orientation_diff = desired_heading - robot_pose[2]
    # print(orientation_diff)
    ppi.set_velocity([0,0])
    
    return robot_pose
    
    
    
 
    
    
# main loop
if __name__ == "__main__":
    
    #! ROTATE or NOT
    rotate_start = False
    rotate_iter = False
    
    #! !!!!!!!!!!!!!
    parser = argparse.ArgumentParser("Fruit searching")
    #parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--map", type=str, default='M4_prac_map_full.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")    
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)
    datadir = args.calib_dir
    fileK = "{}intrinsic.txt".format(datadir)
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fileD = "{}distCoeffs.txt".format(datadir)
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    fileS = "{}scale.txt".format(datadir)
    scale = np.loadtxt(fileS, delimiter=',')
    #if ip == 'localhost':
     #   scale /= 2
    fileB = "{}baseline.txt".format(datadir)  
    baseline = np.loadtxt(fileB, delimiter=',')
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    ekf =  EKF(robot)

    aruco_det = aruco.aruco_detector(
        ekf.robot, marker_length = 0.07) # size of the ARUCO markers
    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    # print(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    
    #! Nyan ########
    lms = []
    for i in range(len(aruco_true_pos)):
        new_marker = measure.Marker(aruco_true_pos[i].reshape(2,1),i+1)
        lms.append(new_marker)
    ekf.add_landmarks(lms)
    # print(ekf.get_state_vector()[0:3])
    #!##############

    #!### TIN new ##############################################################################################################
    # reading the data from the file
    with open(args.map) as f:
        data = f.read()
    # reconstructing the data as a dictionary
    js = json.loads(data)
    # print(js)
    object_name_list = ['apple','aruco','orange','lemon','pear','strawberry']
    goal_list = []
    obstacle_list = []
    obs_list = []
    for key,value in js.items():
        if 'apple' in key or 'lemon' in key or 'orange' in key or  'pear' in key or 'strawberry' in key :
            # print(key)
            goal_list.append(value) # a dictionary with x,y
        else: # not including the fruits    
            obstacle_list.append(value) # a dictionary with x,y
            obs_list.append([value['x'],value['y']])
    count = 0
    # print(obs_list)
    
    for goal_obj in goal_list:
        goal = np.array([goal_obj['x'], goal_obj['y']])
        
        if rotate_start:
            robot_pose=rotation360(get_robot_pose(aruco_true_pos),aruco_true_pos)
        
        start = get_robot_pose(aruco_true_pos)[0:2] # only need x,y no theta
        
  
        # testing
        # start = [1.1992355075676007, -1.0007954680562197]
        # goal = [1.2,1]
        
        print('starting coord', start)
        print('current goal:', goal)
        


        all_obstacles_wrt_current_goal = find_obs_wrt_goal(goal, goal_list, obstacle_list, 0.3) # last number is the radius of obstacle
        # print('obs without goal:', all_obstacles_wrt_current_goal)
        

        rrtc = RRTC(start=start, goal=goal, width=3, height=3, obstacle_list=all_obstacles_wrt_current_goal,
                    expand_dis=0.05, path_resolution=0.0005)


        path_robot_to_goal = rrtc.planning() # this is a list of coordinates [x,y]
        print('\nfinish rrtc for goal ', count)

        plot_path(path_robot_to_goal, obs_list,'path',count=count)
        if any(path_robot_to_goal[-1] !=goal):
            path_robot_to_goal=path_robot_to_goal[::-1]
        plt.savefig
        print('rrtc path from:',path_robot_to_goal[0],'to',path_robot_to_goal[-1])
        
        
        robot_pose_list =[]

        robot_pose = get_robot_pose(aruco_true_pos)
        robot_pose_list.append(robot_pose)

        for i,coord in enumerate(path_robot_to_goal):

            waypoint =coord # coord is according to the path planning
            # print('\nwaypoint dest:',waypoint)
            distance_to_next_waypoint =get_distance_to_goal(waypoint, robot_pose)
            # if i>0 and distance_to_next_waypoint > 0.125:
            #     robot_pose = get_robot_pose(aruco_true_pos)
            #     sth = path_robot_to_goal[i-1]
            #     print('copyfrompath:',sth, type(sth))
            #     print('robot theta:',robot_pose[2], type(robot_pose[2]))
            #     sth.append(robot_pose[2])
            #     robot_pose = sth
            #     print('SPECIAL:', robot_pose, sth)
                
            robot_pose = drive_to_point(waypoint,robot_pose,aruco_true_pos)
            robot_pose_list.append(robot_pose[0:2])
            print("\nreaching waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
            # stop 
            ppi.set_velocity([0, 0])
            
            if rotate_iter:
                if i!=0 and i%10==0:
                    robot_pose=rotation360(get_robot_pose(aruco_true_pos),aruco_true_pos)
            
            distance_to_goal =get_distance_to_goal(waypoint, path_robot_to_goal[-1])
            print('distance to goal remaining:', distance_to_goal)
            if distance_to_goal < 0.4:
                print('within stoping range to collect goal', count)
                plot_path(path_robot_to_goal, obs_list,'path',robot_pose_list,count)
                
                break
            # if i % 10 == 0:
            plot_path(path_robot_to_goal, obs_list,'path',robot_pose_list,count)
        count+=1
        
        # b= [x[0] for x in a]
        # b1= [x[1] for x in a]

        # plt.plot(b,b1)



    #!###########################################################################################################################


    # # The following code is only a skeleton code the semi-auto fruit searching task
    # while True:
    #     # enter the waypoints
    #     # instead of manually enter waypoints, you can get coordinates by clicking on a map, see camera_calibration.py
    #     x,y = 0.0,0.0
    #     x = input("X coordinate of the waypoint: ")
    #     try:
    #         x = float(x)
    #     except ValueError:
    #         print("Please enter a number.")
    #         continue
    #     y = input("Y coordinate of the waypoint: ")
    #     try:
    #         y = float(y)
    #     except ValueError:
    #         print("Please enter a number.")
    #         continue

    #     # estimate the robot's pose
    #     robot_pose = get_robot_pose()

    #     # robot drives to the waypoint
    #     waypoint = [x,y]
    #     drive_to_point(waypoint,robot_pose)
    #     robot_pose = get_robot_pose()
    #     print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

    #     # exit
    #     ppi.set_velocity([0, 0])
    #     uInput = input("Add a new waypoint? [Y/N]")
    #     if uInput == 'N':
    #         break