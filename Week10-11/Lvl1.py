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

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

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
    with open('search_list.txt', 'r') as fd:
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
def drive_to_point(waypoint):
    robot_pose = get_robot_pose(ekfvar)
    # print("Robot_pose:")
    # print(robot_pose)
    ppi.set_velocity([0,0])
    turn_vel = 25
    wheel_vel = 50 # tick
    target_theta = np.arctan2((waypoint[1]-robot_pose[1]),(waypoint[0]-robot_pose[0]))
    print(target_theta)
    if target_theta < 0:
        target_theta += 2*np.pi
    #print(target_theta)
    target_diff = target_theta - robot_pose[2]
    print("target diff before adjustment: " + str(target_diff))
    while target_diff < 0 or target_diff > 2*np.pi:
        if target_diff < 0:
            target_diff += 2*np.pi
        elif target_diff > 2*np.pi:
            target_diff -= 2*np.pi
    # turn towards the waypoint
    print("target diff after adjustment: " + str(target_diff))
    if target_diff > np.pi:
        turn_time = float(baseline*np.abs(2*np.pi-target_diff)/(scale*turn_vel*2)) # replace with your calculation
        print("Turning for right {:.2f} seconds".format(turn_time))
        lv, rv = ppi.set_velocity([0, -1], turning_tick=turn_vel, time=turn_time)
    else:
        turn_time = float(baseline*np.abs(target_diff)/(scale*turn_vel*2)) # replace with your calculation
        print("Turning for left {:.2f} seconds".format(turn_time))
        lv, rv = ppi.set_velocity([0, 1], turning_tick=turn_vel, time=turn_time)
    #print(lv)
    #print(rv)
    ppi.set_velocity([0, 0])
    
    drive_meas = measure.Drive(lv, -rv,turn_time)
    ekfvar.predict(drive_meas)
    robot_pose = get_robot_pose(ekfvar)
    
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
    robot_pose = get_robot_pose( ekfvar)
    ppi.set_velocity([0,0])
    ####################################################
    #new_pose = np.array([waypoint[0],waypoint[1],target_theta])
    #new_pose = new_pose.reshape((3,1))
    
    #ekf.set_state_vector(new_pose)

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose(ekfvar):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    #image = ppi.get_image()
    #lms, aruco_img = aruco_det.detect_marker_positions(image)
    # call_umeyama(lms, aruco_true_pos) -> still needs work
    

        #ekfvar.add_landmarks(lms) #Is this needed
        #ekfvar.update(lms)
    pose = ekfvar.get_state_vector()[0:3,0]
    print(pose)

    ekfvar.robot.state[2] = (pose[2]+2*np.pi)%(2*np.pi)
    pose = ekfvar.get_state_vector()[0:3]

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
    return int((x+1.5)*800/3)

# main loop
if __name__ == "__main__":

    pygame.init()
    screen_width = 800
    screen_height = 800
    scale = 0.5

    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)

    robot_img = pygame.image.load('robot_img.png')
    original_robot_img = pygame.transform.scale(robot_img, (50, 50))
    robot_img = pygame.image.load('robot_img.png')
    original_robot_img = pygame.transform.scale(robot_img, (48, 48))
    durian_img = pygame.transform.scale(pygame.image.load('durian_img.png'), (22, 22))
  # Save the original image for rotations

    # Robot attributes
    robot_rect = robot_img.get_rect()
    robot_rect.center = (screen_width // 2, screen_height // 2)  # Starting position

    # Create the Pygame window
    pygame.display.set_caption("visualisation")
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

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    #search_list = read_search_list()
    #print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = get_robot_pose(ekfvar)
    running = True
    # The following is only a skeleton code for semi-auto navigation

    env1 = env.Env()
    env1.set_arena_size(3000, 3000)
    obs_aruco = []
    for i in range(len(aruco_true_pos)):
        #print(aruco_true_pos[i,:])
        env1.add_square_obs((aruco_true_pos[i,:][0]+1.5)*1000, (aruco_true_pos[i,:][1]+1.5)*1000, 180)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        #print(theta_pygame)
        screen.fill(white)
            # Rotate the robot image

        rotated_robot_img = pygame.transform.rotate(original_robot_img, int(np.squeeze(robot_pose[2])))
        rotated_rect = rotated_robot_img.get_rect()
        scaled_x, scaled_y = world_to_gui(robot_pose[1]), world_to_gui(robot_pose[0])
        rotated_rect.center = (scaled_x, scaled_y)
        # Draw the rotated robot image
        screen.blit(rotated_robot_img, rotated_rect)
        for i in range(len(aruco_true_pos)):
            #print(aruco_true_pos[i,:][0])
            pygame.draw.rect(screen, black, (world_to_gui(aruco_true_pos[i,:][0])-24, world_to_gui(aruco_true_pos[i,:][1])-24, 48, 48))
        pygame.display.flip()
        x,y,theta = 0.0,0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue
        screen.fill(white)
        screen.blit(rotated_robot_img, rotated_rect)
        pygame.draw.line(screen, red, (scaled_x, scaled_y), (int((y+1.5)*800/3),int((x+1.5)*800/3)), 5)
        pygame.display.flip()
        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        #new_pose = np.array([waypoint[0], waypoint[1],np.arctan2((robot_pose[1]-waypoint[1]),robot_pose[0]-waypoint[0])/np.pi*180])
        #reshape = new_pose.reshape((3,1))
        #ekf.set_state_vector(reshape)
        # estimate the robot's pose
        robot_pose = get_robot_pose(ekfvar)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
        pygame.display.flip()
        screen.fill(white)
            # Rotate the robot image
        rotated_robot_img = pygame.transform.rotate(original_robot_img, robot_pose[2])
        rotated_rect = rotated_robot_img.get_rect()
        scaled_x, scaled_y = world_to_gui(robot_pose[1]), world_to_gui(robot_pose[0])
        rotated_rect.center = (scaled_x, scaled_y)
        # Draw the rotated robot image
        screen.blit(rotated_robot_img, rotated_rect)
        pygame.display.flip()
        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
    pygame.quit()