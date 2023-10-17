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
import shutil 
from copy import deepcopy

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
from YOLO.detector import Detector
#from operate_yolo_search import Operate

# import utility functions
sys.path.insert(0, "{}/util")
import util.DatasetHandler as dh 
from pibot import PenguinPi
import measure as measure

class Operate:
    def __init__(self, args):
        self.level = 0
        self.camera_matrix = None
        self.scale = None
        self.baseline = None
        self.dist_coeffs = None
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers
        self.tick=50
        self.turning_tick=20

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = True
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 3000
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        
        if args.yolo_model == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.yolo_model)
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

        self.last_keys_pressed = [False, False, False, False, False]
        self.vis_id = 0
        self.fruit_search = False
        self.fruit_lists = None
        self.fruits_true_pos = None
        self.aruco_true_pos = None

    # wheel control
    def control(self, motion, drive_time):
        if not self.fruit_search:       
            if args.play_data:
                lv, rv = self.pibot.set_velocity()            
            else:
                lv, rv = self.pibot.set_velocity(
                    self.command['motion'], tick=self.tick, turning_tick=self.turning_tick)
            if not self.data is None:
                self.data.write_keyboard(lv, rv)
            dt = time.time() - self.control_clock
            drive_meas = measure.Drive(lv, rv, dt)
            self.control_clock = time.time()
            return drive_meas
        else:
            lv, rv = self.pibot.set_velocity(motion, time = drive_time, tick=self.tick, turning_tick=self.turning_tick)
            drive_meas = measure.Drive(lv, -rv, drive_time)
            self.control_clock = time.time()
            return drive_meas
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        print('Update slam running')
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        # cv2.imshow("img", self.aruco_img)
        # cv2.waitKey(0)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            print(lms)
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            image = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            output, output_img = self.detector.detect_single_image(image)
            self.detector_output = output[0].numpy()
            print(self.detector_output)
            self.network_vis = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            self.command['inference'] = False
            self.file_output = (self.detector_output, deepcopy(self.ekf.robot.state[:,0].tolist()))
            self.notification = f'{self.detector_output.shape[0]} target(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        self.camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        self.dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        self.scale = np.loadtxt(fileS, delimiter=',')
        # if ip == 'localhost':
        #     self.scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        self.baseline = np.loadtxt(fileB, delimiter=',')
        # self.baseline = 0.12
        robot = Robot(self.baseline, self.scale, self.camera_matrix, self.dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)

                bboxs = self.file_output[0][:,0:4]
                labels = self.file_output[0][:,5]
                confs = self.file_output[0][:,4]

                out = {
                    "pose" : self.file_output[1],
                    "bbox" : bboxs.tolist(),
                    "confs" : confs.tolist(),
                    "labels" : labels.tolist()
                }
                cv2.imwrite('lab_output/pred_img'+str(self.vis_id)+'.png', cv2.cvtColor(self.network_vis, cv2.COLOR_RGB2BGR))
                self.vis_id += 1
                self.notification = f'Prediction is saved to lab_output/output.txt'
                with open('lab_output/output.txt', 'a') as f:
                    f.write(json.dumps(out))
                    f.write('\n')


            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    def scale_speed(self):
        keys_pressed = pygame.key.get_pressed()
        shift_pressed = keys_pressed[pygame.K_LSHIFT] or keys_pressed[pygame.K_RSHIFT]
        if shift_pressed == True:
            speedscale = 3
        else:
            speedscale = 1
        return speedscale 

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():

            if not self.fruit_search:
    ########### replace with your M1 codes ###########
                keys = pygame.key.get_pressed()
                up = keys[pygame.K_UP]
                down = keys[pygame.K_DOWN]
                left = keys[pygame.K_LEFT]
                right = keys[pygame.K_RIGHT]
                shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
                
            ############### add your codes below ###############
                v = 1
                keys_pressed = [up, down, left, right, shift]
                if keys_pressed != self.last_keys_pressed:
                    if up:
                        if (not left) and (not right): # up only
                            self.command['motion'] = [self.scale_speed()*v,0]
                            print("Moving Forward")
                        elif left and (not right): # up left
                            self.command['motion'] = [self.scale_speed()*v,self.scale_speed()*v]
                            print("Moving Forward-Left")
                        elif (not left) and right: # up right
                            self.command['motion'] = [self.scale_speed()*v,-self.scale_speed()*v]
                            print("Moving Forward-Right")
                        else:
                            self.command['motion'] = [0, 0]
                            print("Moving Forward")
                    elif down:
                        if (not left) and (not right): # down only
                            self.command['motion'] = [-self.scale_speed()*v,0]
                            print("Moving Backward")
                        elif left and (not right): # down left
                            self.command['motion'] = [-self.scale_speed()*v,-self.scale_speed()*v]
                            print("Moving Backward-Left")
                        elif (not left) and right: # down right
                            self.command['motion'] = [-self.scale_speed()*v,self.scale_speed()*v]
                            print("Moving Backword-Right")
                        else:
                            self.command['motion'] = [0, 0]
                            print("Moving Backward")
                    elif left: 
                        if not right: # left only
                            self.command['motion'] = [0,self.scale_speed()*v]
                            print("Spinning Left")
                        else:
                            self.command['motion'] = [0, 0]
                            print("Stopping")
                    elif right: # right only
                        self.command['motion'] = [0,-self.scale_speed()*v]
                        print("Spinning Right")
                    else:
                        self.command['motion'] = [0, 0]
                        print("Stopping")
                    
                    self.last_keys_pressed = [up, down, left, right, shift]


            ####################################################
            # stop
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True

                #     else:
                #         self.notification = '> 2 landmarks is required for pausing'
                # elif n_observed_markers < 3:
                #     self.notification = '> 2 landmarks is required for pausing'
                # else:
                #     if not self.ekf_on:
                #         self.request_recover_robot = True
                #     self.ekf_on = not self.ekf_on
                #     if self.ekf_on:
                #         self.notification = 'SLAM is running'
                #     else:
                #         self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

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

    robot_pose = get_robot_pose()
    Kt, Kd = 1, 1
    # print("Robot_pose:")
    print(robot_pose)
    distance_to_goal = get_distance_to_goal(waypoint, robot_pose)
    ppi.set_velocity([0,0])
    turn_vel = operate.turning_tick
    wheel_vel = operate.tick # tick
    target_theta = np.arctan2((waypoint[1]-robot_pose[1]),(waypoint[0]-robot_pose[0]))
    print(target_theta)
    while target_theta < 0:
        target_theta += 2*np.pi
    #print(target_theta)
    target_diff = target_theta - robot_pose[2]
    print("target diff before adjustment: " + str(target_diff))
    if target_diff < 0:
        target_diff += 2*np.pi
    elif target_diff > 2*np.pi:
        target_diff -= 2*np.pi
    # turn towards the waypoint
    print("target diff after adjustment: " + str(target_diff))
    if target_diff > np.pi:
        turn_time = float(operate.baseline*np.abs(2*np.pi-target_diff)/(operate.scale*turn_vel*2))*Kt # replace with your calculation
        print("Turning for right {:.2f} seconds".format(turn_time))
        drive_meas = operate.control([0, -1], turn_time)
    else:
        turn_time = float(operate.baseline*np.abs(target_diff)/(operate.scale*turn_vel*2))*Kt # replace with your calculation
        print("Turning for left {:.2f} seconds".format(turn_time))
        drive_meas = operate.control([0, 1], turn_time)
    #print(lv)
    #print(rv)
    #ppi.set_velocity([0, 0])
    
    #drive_meas = measure.Drive(lv, -rv,turn_time)
    operate.take_pic()
    operate.update_slam(drive_meas)
    operate.control([0, 0], 0.1)
    robot_pose = get_robot_pose()
    
    # calculate distance_travel
    distance_travel = np.sqrt((robot_pose[0]-waypoint[0])**2+(robot_pose[1]-waypoint[1])**2)
    #print(distance_travel)
    
    # after turning, drive straight to the waypoint
    drive_time = float(distance_travel/(wheel_vel*operate.scale))*Kd # replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    drive_meas = operate.control([1, 0], drive_time)
    #print(lv)
    #print(rv)
    #drive_meas = measure.Drive(lv, -rv,drive_time)
    operate.take_pic()
    operate.update_slam(drive_meas)
    operate.control([0, 0], 0.1)
    robot_pose = get_robot_pose()
    ####################################################
    #new_pose = np.array([waypoint[0],waypoint[1],target_theta])
    #new_pose = new_pose.reshape((3,1))
    
    #ekf.set_state_vector(new_pose)

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    robot_pose = operate.ekf.get_state_vector()[0:3,0]
    while robot_pose[2] < 0 or robot_pose[2] > 2*np.pi:
        if robot_pose[2] < 0:
            robot_pose[2] += 2*np.pi
        else:
            robot_pose[2] -= 2*np.pi
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
    new_distance = distance - 200
    #print(new_distance)
    new_waypoint = (int((end[0]-start[0])*(new_distance/distance) + start[0]),int((end[1]-start[1])*(new_distance/distance) + start[1]))
    #print(new_waypoint)
    return new_waypoint

def distance_to_waypoint(waypoint, robot_pose):
    return np.sqrt((((1.5-robot_pose[0])*1000)-waypoint[0])**2+(((1.5-robot_pose[1])*1000)-waypoint[1])**2)

def distance_waypoint_to_waypoint(start, end):
    return int(np.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2))

def within_waypoint(waypoint, robot_pose):
    if distance_to_waypoint(waypoint, robot_pose) < 0.3:
        return True
    return False



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
    yellow = (255, 200, 0)
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
    parser.add_argument("--map", type=str, default='Map.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt')
    parser.add_argument("--manual", action='store_false')
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

    operate = Operate(args)
    operate.fruit_search = args.manual
    operate.ekf_on = True

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    avoid_list = read_avoid_list(search_list, fruits_list, fruits_true_pos)
    #print(search_list)
    #print(fruits_list)
    #print(fruits_true_pos)
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    operate.fruit_lists = fruits_list
    operate.fruits_true_pos = fruits_true_pos
    operate.aruco_true_pos = aruco_true_pos

    #x,y = 0.0,0.0
    robot_pose = get_robot_pose()
    running = True
    # The following is only a skeleton code for semi-auto navigation

    #env1 = env.Env()
    #env1.set_arena_size(3000, 3000)

    lms = []
    ARUCO_SCREEN_SIZE = 22

    for i in range(len(aruco_true_pos)):
        #print(aruco_true_pos[i,:])
        new_marker = measure.Marker(aruco_true_pos[i].reshape(2,1),i+1, 0.001*np.eye(2))
        lms.append(new_marker)
        #env1.add_square_obs((1.5-aruco_true_pos[i,:][0])*1000, (1.5-aruco_true_pos[i,:][1])*1000, ARUCO_OBS)
    ekfvar.add_landmarks(lms)
    #for i in range(len(fruits_true_pos)):
        #print(aruco_true_pos[i,:])
        #env1.add_square_obs(int(1500-fruits_true_pos[i,:][0]*1000), int(1500-fruits_true_pos[i,:][1]*1000), FRUIT_OBS)

    #threshold_stopping = input("Enter stopping distance in mm: ")
    screen.fill(white)
        # Rotate the robot image

    rotated_robot_img = pygame.transform.rotate(original_robot_img, int(np.squeeze(robot_pose[2])/np.pi*180))
    rotated_rect = rotated_robot_img.get_rect()
    scaled_x, scaled_y = world_to_gui(robot_pose[1]), world_to_gui(robot_pose[0])
    rotated_rect.center = (scaled_x+int(18*np.sin(robot_pose[2])), scaled_y+int(18*np.cos(robot_pose[2])))
    # Draw the rotated robot image
    waypoints = [(1500,1500)]
    for fruit in search_list:
        for i in range(len(fruits_list)): # there are 5 targets amongst 10 objects
            if fruit == fruits_list[i]:
                waypoints.append((int(1500-fruits_true_pos[i][0]*1000), int(1500-fruits_true_pos[i][1]*1000)))
                #true_waypoints.append((int(1500-fruits_true_pos[i][0]*1000), int(1500-fruits_true_pos[i][1]*1000)))
    for i in range(1,len(waypoints)):
        #print((world_to_gui(true_waypoints[i][1]), world_to_gui(true_waypoints[i][0])))
        
        pygame.draw.circle(screen, yellow, (int(waypoints[i][1]/30*8), int(waypoints[i][0]/30*8)), 133)
        text_surface = font.render(str(i), True, red)
        screen.blit(text_surface, (int(waypoints[i][1]/30*8), int(waypoints[i][0]/30*8)-100))
    for i in range(len(fruits_true_pos)):
        #print(aruco_true_pos[i,:][0])
        #pygame.draw.circle(screen, yellow, (world_to_gui(fruits_true_pos[i,:][1]), world_to_gui(fruits_true_pos[j,:][0])), 133)
        screen.blit(durian_img, (world_to_gui(fruits_true_pos[i,:][1])-11, world_to_gui(fruits_true_pos[i,:][0])-11))
    for i in range(len(aruco_true_pos)):
        #print(aruco_true_pos[i,:][0])
        pygame.draw.rect(screen, black, (world_to_gui(aruco_true_pos[i,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[i,:][0])- ARUCO_SCREEN_SIZE//2, ARUCO_SCREEN_SIZE,  ARUCO_SCREEN_SIZE))
        text_content = str(i+1)
        text_surface = font.render(text_content, True, white)
        screen.blit(text_surface, (world_to_gui(aruco_true_pos[i,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[i,:][0])- ARUCO_SCREEN_SIZE//2))
    screen.blit(rotated_robot_img, rotated_rect)
    pygame.display.flip()
    
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                gui_y, gui_x = pygame.mouse.get_pos()
                print(gui_x)
                print(gui_y)
                x = 1.5-gui_x*3/800
                y = 1.5-gui_y*3/800
                print(x)
                print(y)
                scaled_x, scaled_y = world_to_gui(robot_pose[1]), world_to_gui(robot_pose[0])
                screen.fill(white)
                #pygame.draw.circle(screen, yellow, (int(waypoints[i+1][1]/30*8), int(waypoints[i+1][0]/30*8)), 133)
                for i in range(1,len(waypoints)):
                    #print((world_to_gui(true_waypoints[i][1]), world_to_gui(true_waypoints[i][0])))
                    pygame.draw.circle(screen, yellow, (int(waypoints[i][1]/30*8), int(waypoints[i][0]/30*8)), 133)
                    text_surface = font.render(str(i), True, red)
                    screen.blit(text_surface, (int(waypoints[i][1]/30*8), int(waypoints[i][0]/30*8)-100))
                for j in range(len(fruits_true_pos)):
                #print(aruco_true_pos[i,:][0])
                    screen.blit(durian_img, (world_to_gui(fruits_true_pos[j,:][1])-11, world_to_gui(fruits_true_pos[j,:][0])-11))
                
                for j in range(len(aruco_true_pos)):
                    pygame.draw.rect(screen, black, (world_to_gui(aruco_true_pos[j,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[j,:][0])- ARUCO_SCREEN_SIZE//2,  ARUCO_SCREEN_SIZE,  ARUCO_SCREEN_SIZE))
                    text_content = str(j+1)
                    text_surface = font.render(text_content, True, white)
                    screen.blit(text_surface, (world_to_gui(aruco_true_pos[j,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[j,:][0])- ARUCO_SCREEN_SIZE//2))
                    #pygame.draw.rect(screen, purple, (world_to_gui(fruits_true_pos[j,:][1])-24, world_to_gui(fruits_true_pos[j,:][0])-12, 24, 24))
                screen.blit(rotated_robot_img, rotated_rect)
                pygame.display.flip()
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
                        
                
                waypoint = (x,y)
                print(waypoint)
                
                screen.fill(white)
                # Display Robot after each waypoint
                rotated_robot_img = pygame.transform.rotate(original_robot_img, robot_pose[2]/np.pi*180)
                rotated_rect = rotated_robot_img.get_rect()
                scaled_x, scaled_y = world_to_gui(robot_pose[1]), world_to_gui(robot_pose[0])
                rotated_rect.center = (scaled_x+int(18*np.sin(robot_pose[2])), scaled_y+int(18*np.cos(robot_pose[2])))
                # Draw the rotated robot image

                for i in range(1,len(waypoints)):
                    #print((world_to_gui(true_waypoints[i][1]), world_to_gui(true_waypoints[i][0])))
                    pygame.draw.circle(screen, yellow, (int(waypoints[i][1]/30*8), int(waypoints[i][0]/30*8)), 133)
                    text_surface = font.render(str(i), True, red)
                    screen.blit(text_surface, (int(waypoints[i][1]/30*8), int(waypoints[i][0]/30*8)-100))
                
                for k in range(len(fruits_true_pos)):
                #print(aruco_true_pos[i,:][0])
                    #pygame.draw.circle(screen, yellow, (world_to_gui(fruits_true_pos[k,:][1]), world_to_gui(fruits_true_pos[k,:][0])), 133)
                    screen.blit(durian_img, (world_to_gui(fruits_true_pos[k,:][1])-11, world_to_gui(fruits_true_pos[k,:][0])-11))
                for k in range(len(aruco_true_pos)):
                    #print(aruco_true_pos[i,:][0])
                    pygame.draw.rect(screen, black, (world_to_gui(aruco_true_pos[k,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[k,:][0])- ARUCO_SCREEN_SIZE//2,  ARUCO_SCREEN_SIZE,  ARUCO_SCREEN_SIZE))
                    #screen.blit(text_surface, (int(waypoints[i][1]/30*8), int(waypoints[i][0]/30*8)-300))
                screen.blit(rotated_robot_img, rotated_rect)
                pygame.draw.line(screen, red, (scaled_x, scaled_y), (gui_y,gui_x), 5)
                pygame.display.flip()
                drive_to_point(waypoint,robot_pose,ekfvar)
                robot_pose = get_robot_pose()
                print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
                screen.fill(white)
                    # Rotate the robot image
                rotated_robot_img = pygame.transform.rotate(original_robot_img, robot_pose[2]/np.pi*180)
                rotated_rect = rotated_robot_img.get_rect()
                scaled_x, scaled_y = world_to_gui(robot_pose[1]), world_to_gui(robot_pose[0])
                rotated_rect.center = (scaled_x+int(18*np.sin(robot_pose[2])), scaled_y+int(18*np.cos(robot_pose[2])))
                # Draw the rotated robot image
                #pygame.draw.circle(screen, yellow, (int(waypoints[i+1][1]/30*8), int(waypoints[i+1][0]/30*8)), 133)
                for i in range(1,len(waypoints)):
                    #print((world_to_gui(true_waypoints[i][1]), world_to_gui(true_waypoints[i][0])))
                    pygame.draw.circle(screen, yellow, (int(waypoints[i][1]/30*8), int(waypoints[i][0]/30*8)), 133)
                    text_surface = font.render(str(i), True, red)
                    screen.blit(text_surface, (int(waypoints[i][1]/30*8), int(waypoints[i][0]/30*8)-100))
                for j in range(len(fruits_true_pos)):
                #print(aruco_true_pos[i,:][0])
                    #pygame.draw.circle(screen, yellow, (world_to_gui(fruits_true_pos[j,:][1]), world_to_gui(fruits_true_pos[j,:][0])), 133)
                    screen.blit(durian_img, (world_to_gui(fruits_true_pos[j,:][1])-11, world_to_gui(fruits_true_pos[j,:][0])-11))
                for j in range(len(aruco_true_pos)):
                    #print(aruco_true_pos[i,:][0])
                    pygame.draw.rect(screen, black, (world_to_gui(aruco_true_pos[j,:][1])- ARUCO_SCREEN_SIZE//2, world_to_gui(aruco_true_pos[j,:][0])- ARUCO_SCREEN_SIZE//2, ARUCO_SCREEN_SIZE, ARUCO_SCREEN_SIZE))
                screen.blit(rotated_robot_img, rotated_rect)
                pygame.display.flip()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
            time.sleep(0.2)
                    
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        #print(theta_pygame)
        

        #print(waypoints)




        #if distance_waypoint_to_waypoint(smoothed_path[-1], waypoints[i+1]) < 410:
            #smoothed_path[-1] = calc_new_waypoint2(smoothed_path[-2], smoothed_path[-1])
        #print(smoothed_path)

        
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
