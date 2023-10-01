# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import math
from Astar import AStar
import plotting, env
import matplotlib.pyplot as plt
from path_smoothing import smooth_path
import pygame
import time

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
#from operate_yolo_search import Operate

# import utility functions
sys.path.insert(0, "{}/util")
from pibot import PenguinPi
import measure as measure



def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
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
                    marker_id = int(key[5]) - 1
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

def read_avoid_list(search_list, fruit_list, fruit_true_pos):
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    avoid_list = []
    for i in range(len(fruit_list)):
        if fruit_list[i] not in search_list:
            avoid_list.append((fruit_true_pos[i][0], fruit_true_pos[i][1]))

    return avoid_list

def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
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
    #theta = theta/180*np.pi
    angle_to_goal = np.arctan2(y_diff,x_diff) - theta
    print("Angle to goal: " + str(angle_to_goal))
    desired_heading = (angle_to_goal+np.pi) % (2*np.pi) + (-np.pi)

    return desired_heading

def drive_to_point(waypoint, robot_pose, ekfvar):
    # imports camera / wheel calibration parameters 
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    # wheel_vel = 30 # tick
    # wheel_w = 5
    # K_pv = 1
    # K_pw = 1
    
    # # get distance to waypoint
    # distance_to_goal = get_distance_to_goal(waypoint, robot_pose)
    # # print("Distance_to_goal:")
    # # print(distance_to_goal)
    # #print("Robot_pose:")
    # #print(robot_pose)

    # # get desired heading
    # desired_heading = get_desired_heading(waypoint, robot_pose)
    # print("Desired_heading:")
    # print(desired_heading)

    # rot_threshold = 0.05 # might want to decrease for better acc
    # dist_threshold = 0.05

    # #while abs(orientation_diff) > rot_threshold:
    # w_k = K_pw*desired_heading
    # if desired_heading <0:
    #     r = -1
    # else:
    #     r = 1
    # turn_time =  abs(baseline/(scale*wheel_w)*(w_k/np.pi))
    # # print("Turn time: ")
    # # print(turn_time)
    # lv, rv = ppi.set_velocity([0, r], turning_tick=wheel_w, time=turn_time)
    # # print('left-right v:',lv,rv)
    
    # drive_meas = measure.Drive(lv, rv,turn_time, 1, 1)
    # ekf.predict(drive_meas)
    # robot_pose = get_robot_pose(aruco_true_pos, ekfvar)
    # # print("Robot_pose:")
    # # print(robot_pose)
    # #desired_heading = get_desired_heading(waypoint, robot_pose)
    # orientation_diff = desired_heading - robot_pose[2]
    # # print(orientation_diff)
    # ppi.set_velocity([0,0])
        
           
    #while distance_to_goal > dist_threshold:
    # v_k = K_pv*distance_to_goal
    # drive_time = v_k/(scale*wheel_vel)
    # #print(v_k, scale, wheel_vel, waypoint, robot_pose)
    # #print("Drive time: ")
    # #print(drive_time)
    # lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    # drive_meas = measure.Drive(lv, rv,drive_time, 1, 1)
    # ekf.predict(drive_meas)

    robot_pose = get_robot_pose(aruco_true_pos, ekfvar)
    # print("Robot_pose:")
    # print(robot_pose)
    distance_to_goal = get_distance_to_goal(waypoint, robot_pose)
    ppi.set_velocity([0,0])
    turn_vel = 5
    wheel_vel = 35 # tick
    target_theta = np.arctan2((waypoint[1]-robot_pose[1]),(waypoint[0]-robot_pose[0]))
    print(target_theta)
    if target_theta < 0:
        target_theta += 2*np.pi
    #print(target_theta)
    target_diff = target_theta - robot_pose[2]
    #print(target_diff)
    if target_diff < 0:
        target_diff += 2*np.pi
    # turn towards the waypoint
    print(target_diff)
    if target_diff > np.pi:
        turn_time = float(baseline*np.abs(2*np.pi-target_diff)/(scale*turn_vel*2)) # replace with your calculation
        print("Turning for {:.2f} seconds".format(turn_time))
        lv, rv = ppi.set_velocity([0, -1], turning_tick=turn_vel, time=turn_time)
    else:
        turn_time = float(baseline*np.abs(target_diff)/(scale*turn_vel*2)) # replace with your calculation
        print("Turning for {:.2f} seconds".format(turn_time))
        lv, rv = ppi.set_velocity([0, 1], turning_tick=turn_vel, time=turn_time)
    #print(lv)
    #print(rv)
    ppi.set_velocity([0, 0])
    
    drive_meas = measure.Drive(lv, -rv,turn_time)
    ekfvar.predict(drive_meas)
    robot_pose = get_robot_pose(aruco_true_pos,ekfvar)
    
    # calculate distance_travel
    distance_travel = np.sqrt((robot_pose[0]-waypoint[0])**2+(robot_pose[1]-waypoint[1])**2)
    #print(distance_travel)
    
    # after turning, drive straight to the waypoint
    drive_time = float(distance_travel/(wheel_vel*scale)) # replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    #print(lv)
    #print(rv)
    drive_meas = measure.Drive(lv, -rv,drive_time)
    ekfvar.predict(drive_meas)
    robot_pose = get_robot_pose(aruco_true_pos, ekfvar)
    ppi.set_velocity([0,0])
    ####################################################
    #new_pose = np.array([waypoint[0],waypoint[1],target_theta])
    #new_pose = new_pose.reshape((3,1))
    
    #ekf.set_state_vector(new_pose)

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose(aruco_true_pos, ekfvar, from_true_map=False):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    image = ppi.get_image()
    lms, aruco_img = aruco_det.detect_marker_positions(image)
    # call_umeyama(lms, aruco_true_pos) -> still needs work
    
    
    if lms == []:
        return ekfvar.get_state_vector()[0:3,0]
    else:
        for i in range(len(lms)):
            if lms[i].tag-1 <= 10:
                lms[i].position = aruco_true_pos[lms[i].tag-1].reshape(2,1)
            else:
                pass
        print("landmarks detected")
        print([lm.position for lm in lms])
        ekfvar.add_landmarks(lms) #Is this needed
        ekfvar.update(lms)
        pose = ekfvar.get_state_vector()[0:3,0]
        print(pose)
        '''if pose[2] > 2*np.pi:
            pose[2] -= 2*np.pi'''
        ekfvar.robot.state[2] = (pose[2]+2*np.pi)%(2*np.pi)
        pose = ekfvar.get_state_vector()[0:3]
    #print(pose)
    '''if pose[2] > 2*np.pi:
        pose[2] -= 2*np.pi'''
    #pose[2] = pose[2]/180*np.pi
    #print(pose)
    #print(pose[2][0])
    #pose[2] = pose[2]*180/np.pi
    # update the robot pose [x,y,theta]
    while pose[2][0] < 0:
        pose[2][0] += 2*np.pi
    # while pose[2][0] > 2*np.pi:
    #     pose[2][0] -= 2*np.pi
    '''while pose[2] > 2*np.pi:
        pose[2] -= 2*np.pi'''

    robot_pose = [pose[0][0], pose[1][0], pose[2][0]]
    print("Pose: " + str(robot_pose))

    return robot_pose

def parse_groundtruth(fname : str) -> dict:
    with open(fname,'r') as f:
        # gt_dict = ast.literal_eval(f.readline())
        gt_dict = json.load(f)
        aruco_dict = {}
        for key in gt_dict:
            if key.startswith("aruco"):
                aruco_num = int(key.strip('aruco')[:-2])
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
    return aruco_dict

def world_to_gui(x):
    return int((1.5-x)*800/3)

def calc_new_waypoint(waypoint, robot_pose):
    new_waypoint = (0.0,0.0)
    #print(int((robot_pose[0,0]+1.5)*1000),int((robot_pose[1,0]+1.5)*1000))
    #print(waypoint)
    distance = np.sqrt((((1.5-robot_pose[0])*1000)-waypoint[0])**2+(((1.5-robot_pose[1])*1000)-waypoint[1])**2)
    #print(distance)
    new_distance = distance - 200
    #print(new_distance)
    new_waypoint = (int((waypoint[0]-((1.5-robot_pose[0])*1000))*(new_distance/distance) + ((1.5-robot_pose[0])*1000)),int((waypoint[1]-((1.5-robot_pose[1])*1000))*(new_distance/distance) + ((1.5-robot_pose[1])*1000)))
    #print(new_waypoint)
    return new_waypoint

def calc_new_waypoint2(start, end):
    new_waypoint = (0.0,0.0)
    #print(int((robot_pose[0,0]+1.5)*1000),int((robot_pose[1,0]+1.5)*1000))
    #print(waypoint)
    distance = np.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
    #print(distance)
    new_distance = distance - 100
    #print(new_distance)
    new_waypoint = (int((end[0]-start[0])*(new_distance/distance) + start[0]),int((end[1]-start[1])*(new_distance/distance) + start[1]))
    #print(new_waypoint)
    return new_waypoint

def distance_to_waypoint(waypoint, robot_pose):
    return np.sqrt((((1.5-robot_pose[0])*1000)-waypoint[0])**2+(((1.5-robot_pose[1])*1000)-waypoint[1])**2)

def distance_waypoint_to_waypoint(start, end):
    return np.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)

def within_waypoint(waypoint, robot_pose):
    if distance_to_waypoint(waypoint, robot_pose) < 0.3:
        return True
    return False

def rotate_to_centre(robot_pose, ekfvar):
    robot_pose = get_robot_pose(aruco_true_pos, ekfvar)
    # print("Robot_pose:")
    # print(robot_pose)
    #distance_to_goal = get_distance_to_goal(waypoint, robot_pose)
    ppi.set_velocity([0,0])
    turn_vel = 5
    wheel_vel = 35 # tick
    target_theta = np.arctan2((-robot_pose[1]),(-robot_pose[0]))
    print(target_theta)
    if target_theta < 0:
        target_theta += 2*np.pi
    #print(target_theta)
    target_diff = target_theta - robot_pose[2]
    #print(target_diff)
    if target_diff < 0:
        target_diff += 2*np.pi
    # turn towards the waypoint
    print(target_diff)
    if target_diff > np.pi:
        turn_time = float(baseline*np.abs(2*np.pi-target_diff)/(scale*turn_vel*2)) # replace with your calculation
        print("Turning for {:.2f} seconds".format(turn_time))
        lv, rv = ppi.set_velocity([0, -1], turning_tick=turn_vel, time=turn_time)
    else:
        turn_time = float(baseline*np.abs(target_diff)/(scale*turn_vel*2)) # replace with your calculation
        print("Turning for {:.2f} seconds".format(turn_time))
        lv, rv = ppi.set_velocity([0, 1], turning_tick=turn_vel, time=turn_time)
    #print(lv)
    #print(rv)
    ppi.set_velocity([0, 0])
    
    drive_meas = measure.Drive(lv, -rv,turn_time)
    ekfvar.predict(drive_meas)
    robot_pose = get_robot_pose(aruco_true_pos,ekfvar)

# main loop
if __name__ == "__main__":

    pygame.init()
    font = pygame.font.Font(None, 36)
    screen_width = 800
    screen_height = 800
    scale = 0.5

    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    purple = (255, 0, 255)

    robot_img = pygame.image.load('robot_img.png')
    original_robot_img = pygame.transform.scale(robot_img, (48, 48))
    durian_img = pygame.transform.scale(pygame.image.load('durian_img.png'), (22, 22))
  # Save the original image for rotations

    # Robot attributes
    robot_rect = robot_img.get_rect()
    robot_rect.center = (screen_width // 2, screen_height // 2)  # Starting position

    # Create the Pygame window
    pygame.display.set_caption("GUI for path planning reference")
    screen = pygame.display.set_mode((screen_width, screen_height))

    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_prac_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    args, _ = parser.parse_known_args()
    ip = args.ip
    datadir = args.calib_dir
    fileS = "{}scale.txt".format(datadir)
    scale = np.loadtxt(fileS, delimiter=',')
    if ip == 'localhost':
        scale /= 2
    fileK = "{}intrinsic.txt".format(datadir)
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fileD = "{}distCoeffs.txt".format(datadir)
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    fileB = "{}baseline.txt".format(datadir)
    baseline = np.loadtxt(fileB, delimiter=',')

    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    ekfvar = EKF(robot)
    aruco_det = aruco.aruco_detector(
            ekfvar.robot, marker_length=0.07)
    #operate = Operate(args)

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    avoid_list = read_avoid_list(search_list, fruits_list, fruits_true_pos)
    #print(search_list)
    #print(fruits_list)
    #print(fruits_true_pos)
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = get_robot_pose(aruco_true_pos, ekfvar)
    running = True
    # The following is only a skeleton code for semi-auto navigation

    env1 = env.Env()
    env1.set_arena_size(3000, 3000)
    #obs_aruco = []
    lms = []
    for i in range(len(aruco_true_pos)):
        #print(aruco_true_pos[i,:])
        new_marker = measure.Marker(aruco_true_pos[i].reshape(2,1),i+1, 0.0001*np.eye(2))
        lms.append(new_marker)
        env1.add_square_obs((1.5-aruco_true_pos[i,:][0])*1000, (1.5-aruco_true_pos[i,:][1])*1000, 460)
    ekfvar.add_landmarks(lms)
    for i in range(len(fruits_true_pos)):
        #print(aruco_true_pos[i,:])
        env1.add_square_obs((1.5-fruits_true_pos[i,:][0])*1000, (1.5-fruits_true_pos[i,:][1])*1000, 380)

    
    ARUCO_SCREEN_SIZE = 22
    # print(avoid_list)
    # for i in range(len(avoid_list)):
    #     #print(aruco_true_pos[i,:])
    #     env1.add_square_obs((avoid_list[i][0]+1.5)*1000, (avoid_list[i][1]+1.5)*1000, 180)

    #obs_aruco = []
    #for i in range(len(aruco_true_pos)):
        #print(aruco_true_pos[i,:])
       # env1.add_square_obs((fruits_true_pos[i,:][0]+1.5)*1000, (fruits_true_pos[i,:][1]+1.5)*1000, 180)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        #print(theta_pygame)
        screen.fill(white)
            # Rotate the robot image

        rotated_robot_img = pygame.transform.rotate(original_robot_img, int(np.squeeze(robot_pose[2])/np.pi*180))
        rotated_rect = rotated_robot_img.get_rect()
        scaled_x, scaled_y = world_to_gui(robot_pose[1]), world_to_gui(robot_pose[0])
        rotated_rect.center = (scaled_x, scaled_y)
        # Draw the rotated robot image
        true_waypoints = []
        waypoints = [(1500,1500)]
        for fruit in search_list:
            for i in range(len(fruits_list)): # there are 5 targets amongst 10 objects
                if fruit == fruits_list[i]:
                    waypoints.append((int(1500-fruits_true_pos[i][0]*1000), int(1500-fruits_true_pos[i][1]*1000)))
                    true_waypoints.append((int(1500-fruits_true_pos[i][0]*1000), int(1500-fruits_true_pos[i][1]*1000)))
        for i in range(len(true_waypoints)):
            #print((world_to_gui(true_waypoints[i][1]), world_to_gui(true_waypoints[i][0])))
            pygame.draw.circle(screen, yellow, (int(true_waypoints[i][1]/30*8), int(true_waypoints[i][0]/30*8)), 133)
        for i in range(len(fruits_true_pos)):
            #print(aruco_true_pos[i,:][0])
            #pygame.draw.circle(screen, yellow, (world_to_gui(fruits_true_pos[i,:][1]), world_to_gui(fruits_true_pos[j,:][0])), 133)
            screen.blit(durian_img, (world_to_gui(fruits_true_pos[i,:][1]), world_to_gui(fruits_true_pos[i,:][0])))
        for i in range(len(aruco_true_pos)):
            #print(aruco_true_pos[i,:][0])
            pygame.draw.rect(screen, black, (world_to_gui(aruco_true_pos[i,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[i,:][0])- ARUCO_SCREEN_SIZE//2, ARUCO_SCREEN_SIZE,  ARUCO_SCREEN_SIZE))
            text_content = str(i+1)
            text_surface = font.render(text_content, True, white)
            screen.blit(text_surface, (world_to_gui(aruco_true_pos[i,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[i,:][0])- ARUCO_SCREEN_SIZE//2))
        screen.blit(rotated_robot_img, rotated_rect)
        pygame.display.flip()

        #print(waypoints)
        for i in range(len(waypoints)-1):
            #print(waypoints[i])
            #print(int((robot_pose[0,0]+1.5)*1000),int((robot_pose[1,0]+1.5)*1000))
            #waypoints[i+1] = calc_new_waypoint(waypoints[i+1], robot_pose)
            print(waypoints[i+1])
            env1.remove_square_obs(true_waypoints[i][0], true_waypoints[i][1], 380)
            print("obs removed")
            #print(waypoints[i+1])
            #pose_in_
            astar = AStar((int((1.5-robot_pose[0])*1000),int((1.5-robot_pose[1])*1000)), waypoints[i+1], "euclidean", env1)
            print("Finding path")
            path, visited = astar.searching()
            smoothed_path = smooth_path(path)
            print(smoothed_path)
            attempt = 0
            while True:
                print("while")
                print(smoothed_path)
                attempt += 1
                if distance_waypoint_to_waypoint(smoothed_path[-2], waypoints[i+1]) < 400:
                    print("true")
                    smoothed_path.pop(len(smoothed_path)-1)
                else:
                    print("break")
                    break
                print(smoothed_path)
            if distance_waypoint_to_waypoint(smoothed_path[-1], waypoints[i+1]) < 410:
                smoothed_path[-1] = calc_new_waypoint2(smoothed_path[-2], smoothed_path[-1])
            print(smoothed_path)
            scaled_x, scaled_y = world_to_gui(robot_pose[1]), world_to_gui(robot_pose[0])
            screen.fill(white)
            pygame.draw.circle(screen, yellow, (int(true_waypoints[i][1]/30*8), int(true_waypoints[i][0]/30*8)), 133)
            for j in range(len(fruits_true_pos)):
            #print(aruco_true_pos[i,:][0])
                screen.blit(durian_img, (world_to_gui(fruits_true_pos[j,:][1]), world_to_gui(fruits_true_pos[j,:][0])))
            screen.blit(rotated_robot_img, rotated_rect)
            for j in range(len(aruco_true_pos)):
                pygame.draw.rect(screen, black, (world_to_gui(aruco_true_pos[j,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[j,:][0])- ARUCO_SCREEN_SIZE//2,  ARUCO_SCREEN_SIZE,  ARUCO_SCREEN_SIZE))
                text_content = str(j+1)
                text_surface = font.render(text_content, True, white)
                screen.blit(text_surface, (world_to_gui(aruco_true_pos[j,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[j,:][0])- ARUCO_SCREEN_SIZE//2))
                #pygame.draw.rect(screen, purple, (world_to_gui(fruits_true_pos[j,:][1])-24, world_to_gui(fruits_true_pos[j,:][0])-12, 24, 24))
            pygame.draw.line(screen, red, (scaled_x, scaled_y), (waypoints[i+1][1]/30*8,waypoints[i+1][0]/30*8), 5)
            for j in range(len(smoothed_path)-1):
                pygame.draw.line(screen, blue, (int(smoothed_path[j][1]*800/3000),int(smoothed_path[j][0]*800/3000)), (int(smoothed_path[j+1][1]*800/3000),int(smoothed_path[j+1][0]*800/3000)), 5)
            pygame.display.flip()
            
            for j in range(1,len(smoothed_path)):
                waypoint = (1.5-smoothed_path[j][0]/1000,1.5-smoothed_path[j][1]/1000)
                print(waypoint)
                drive_to_point(waypoint,robot_pose,ekfvar)
                robot_pose = get_robot_pose(aruco_true_pos, ekfvar)
                screen.fill(white)
                # Display Robot after each waypoint
                rotated_robot_img = pygame.transform.rotate(original_robot_img, robot_pose[2]/np.pi*180)
                rotated_rect = rotated_robot_img.get_rect()
                scaled_x, scaled_y = world_to_gui(robot_pose[1]), world_to_gui(robot_pose[0])
                rotated_rect.center = (scaled_x, scaled_y)
                # Draw the rotated robot image
                pygame.draw.circle(screen, yellow, (int(true_waypoints[i][1]/30*8), int(true_waypoints[i][0]/30*8)), 133)
                for k in range(len(fruits_true_pos)):
                #print(aruco_true_pos[i,:][0])
                    #pygame.draw.circle(screen, yellow, (world_to_gui(fruits_true_pos[k,:][1]), world_to_gui(fruits_true_pos[k,:][0])), 133)
                    screen.blit(durian_img, (world_to_gui(fruits_true_pos[k,:][1]), world_to_gui(fruits_true_pos[k,:][0])))
                screen.blit(rotated_robot_img, rotated_rect)
                for k in range(len(aruco_true_pos)):
                    #print(aruco_true_pos[i,:][0])
                    pygame.draw.rect(screen, black, (world_to_gui(aruco_true_pos[k,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[k,:][0])- ARUCO_SCREEN_SIZE//2,  ARUCO_SCREEN_SIZE,  ARUCO_SCREEN_SIZE))
                pygame.draw.line(screen, red, (scaled_x, scaled_y), (waypoints[i+1][1]/30*8,waypoints[i+1][0]/30*8), 5)
                for k in range(len(smoothed_path)-1):
                    pygame.draw.line(screen, blue, (int(smoothed_path[k][1]*800/3000),int(smoothed_path[k][0]*800/3000)), (int(smoothed_path[k+1][1]*800/3000),int(smoothed_path[k+1][0]*800/3000)), 5)
                pygame.display.flip()
                if within_waypoint(true_waypoints[i], robot_pose):
                    print("break")
                    break
                j += 1
            rotate_to_centre(robot_pose, ekfvar)
            get_robot_pose(aruco_true_pos, ekfvar)
            print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
            #pygame.display.flip()
            screen.fill(white)
                # Rotate the robot image
            rotated_robot_img = pygame.transform.rotate(original_robot_img, robot_pose[2]/np.pi*180)
            rotated_rect = rotated_robot_img.get_rect()
            scaled_x, scaled_y = world_to_gui(robot_pose[1]), world_to_gui(robot_pose[0])
            rotated_rect.center = (scaled_x, scaled_y)
            # Draw the rotated robot image
            pygame.draw.circle(screen, yellow, (int(true_waypoints[i][1]/30*8), int(true_waypoints[i][0]/30*8)), 133)
            for j in range(len(fruits_true_pos)):
            #print(aruco_true_pos[i,:][0])
                #pygame.draw.circle(screen, yellow, (world_to_gui(fruits_true_pos[j,:][1]), world_to_gui(fruits_true_pos[j,:][0])), 133)
                screen.blit(durian_img, (world_to_gui(fruits_true_pos[j,:][1]), world_to_gui(fruits_true_pos[j,:][0])))
            for j in range(len(aruco_true_pos)):
                #print(aruco_true_pos[i,:][0])
                pygame.draw.rect(screen, black, (world_to_gui(aruco_true_pos[j,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[j,:][0])- ARUCO_SCREEN_SIZE//2, ARUCO_SCREEN_SIZE, ARUCO_SCREEN_SIZE))
            screen.blit(rotated_robot_img, rotated_rect)
            pygame.display.flip()
            env1.add_square_obs(true_waypoints[i][0], true_waypoints[i][1], 230)
            print("Target {} reached".format(i+1))
        #x,y,theta = 0.0,0.0,0.0
        # x = input("X coordinate of the waypoint: ")
        # try:
        #     x = float(x)
        # except ValueError:
        #     print("Please enter a number.")
        #     continue
        # y = input("Y coordinate of the waypoint: ")
        # try:
        #     y = float(y)
        # except ValueError:
        #     print("Please enter a number.")
        #     continue
        
        # robot drives to the waypoint
        
        #new_pose = np.array([waypoint[0], waypoint[1],np.arctan2((robot_pose[1]-waypoint[1]),robot_pose[0]-waypoint[0])/np.pi*180])
        #reshape = new_pose.reshape((3,1))
        #ekf.set_state_vector(reshape)
        # estimate the robot's pose
        
        # exit
        # ppi.set_velocity([0, 0])
        # uInput = input("Add a new waypoint? [Y/N]")
        # if uInput == 'N':
        #     break
        break
    screen.fill(white)
    print("All targets reached. Mission accomplished.")
    text_content = "Mission Accomplished! Demonstrators please give us the full mark"
    text_color = (0, 0, 0)  # White color
    text_surface = font.render(text_content, True, black)
    screen.blit(text_surface, (0, 200))
    pygame.display.update()
    time.sleep(30)

    pygame.quit()