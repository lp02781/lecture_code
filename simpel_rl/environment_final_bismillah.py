#!/usr/bin/env python3
import os
import numpy as np
import math
from math import pi
import random
import cv2


diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
istrain = False

class Env():
    def __init__(self, is_training):
        # self.position = Pose()
        # self.targetPoint = Pose()
        self.isTargetReached = True
        self.positionX = 0.
        self.positionY = 0.
        self.targetPointX = 5.
        self.targetPointY = 5.
        self.targetPointX = 0.
        self.targetPointY = 0.
        self.dt = 0.2
        self.minCrashRange = 0.2
        self.agentController = AgentPosController()
        self.goalCont = GoalController()
        self.psi = 0
        self.v = 0
        self.past_distance = 0.
        self.L = 0.35
        self.obs_r = 0.2
        self.robotstate = np.array([[self.positionX, self.positionY, self.v, self.psi]]).T
        self.max_speed = 0.2
        
        ########################Define Obstacles##################################
        obs1 = [2.2, 3.4]#[30, 30]
        obs2 = [4.2, 3.4]#[30, 80] 
        obs3 = [2.2, 5.4]#[80, 30]
        obs4 = [4.2, 5.4]#[80, 80]
        self.obs = np.vstack([obs1, obs2, obs3,obs4])

        #######################Define Walls#######################################
        max_wall_x_init =6.4
        max_wall_y_init =9.2
        min_wall_x_init =0
        min_wall_y_init =0
        wall_rad = 0.15
        self.wall_max_x = max_wall_x_init - wall_rad
        self.wall_max_y = max_wall_y_init - wall_rad
        self.wall_min_x = min_wall_x_init + wall_rad
        self.wall_min_y = min_wall_y_init + wall_rad

        self.past_distance = 0.
        if is_training:
            self.threshold_arrive = 0.4
        else:
            self.threshold_arrive = 0.4

    def calcDistance(self, x1, y1, x2, y2):
        '''
        Calculate euler distance of given two points

        return distance in float
        '''
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def getGoalDistace(self):
        goal_distance = math.hypot(self.targetPointX - self.positionX, self.targetPointY - self.positionY)
        self.past_distance = goal_distance

        return goal_distance
    
    def move(self, accelerate, delta):
        x_dot = self.v*np.cos(self.psi)
        y_dot = self.v*np.sin(self.psi)
        v_dot = accelerate
        psi_dot = self.v*np.tan(delta)/self.L
        return np.array([[x_dot, y_dot, v_dot, psi_dot]]).T

    def getState(self, state_dot):
        done = False
        arrive = False
        self.robotstate = self.robotstate + self.dt*state_dot
        if self.robotstate[2] > self.max_speed:
            self.robotstate[2] = self.max_speed
        self.positionX = self.robotstate[0,0]
        self.positionY = self.robotstate[1,0]
        self.v = self.robotstate[2,0]
        self.psi = self.robotstate[3,0]
        yaw = round(math.degrees(self.psi))
        position = np.array([self.positionX,self.positionY])
        heading = self.calcHeadingAngle(self.targetPointX, self.targetPointY, self.psi, self.positionX, self.positionY)
        for i in range(self.obs.shape[0]):
            obstacle_x = self.obs[i,0]
            obstacle_y = self.obs[i,1]
            obstacle = np.array([obstacle_x, obstacle_y])
            if np.linalg.norm(position - obstacle) > self.obs_r:
                done = False
                #print('path are safe')
            elif np.linalg.norm(position - obstacle) <= self.obs_r:
                done = True
                print('colliding with the obstacle')
                break
        if position[0] > self.wall_max_x or position[0] < self.wall_min_x:
            done = True
            print('collide with wall')
        if position[1] > self.wall_max_y or position[1] < self.wall_min_y:
            done = True
            print('colliding with the wall')

        current_distance = math.hypot(self.targetPointX - self.positionX, self.targetPointY - self.positionY)
        if current_distance <= self.threshold_arrive:
            # done = True
            arrive = True
        state = np.reshape(self.robotstate,(4,))
        target = np.array([self.targetPointX, self.targetPointY])
        #print('state shape in getState',state.shape)
        return state, current_distance, heading, target, done, arrive
    
    def calcHeadingAngle(self, targetPointX, targetPointY, yaw, robotX, robotY):
        '''
        Calculate heading angle from robot to target

        return angle in float
        '''
        targetAngle = math.atan2(targetPointY - robotY, targetPointX - robotX)

        heading = targetAngle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        return round(heading, 2)

    def setReward(self, done, arrive):
        current_distance = math.hypot(self.targetPointX - self.positionX, self.targetPointY - self.positionY)
        distance_rate = (self.past_distance - current_distance)

        reward = 500.*distance_rate
        self.past_distance = current_distance

        if done:
            reward = -100.
            self.reset()

        if arrive:
            reward = 120.
            self.targetPointX = random.uniform(0, 6.4)
            self.targetPointY = random.uniform(0, 9.2)
            
            self.goal_distance = self.getGoalDistace()
            arrive = False

        return reward
    
    def map_function(self, x, in_min, in_max, out_min, out_max):
        output = int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
        return output

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]
        
        acc = linear_vel#self.map_function(linear_vel, 0, 1, 0, self.max_speed)
        delta = np.deg2rad(self.map_function(ang_vel, -1, 1, -30, 30))
        state, current_dis, heading, target, done, arrive = self.getState(self.move(acc,delta))
        
        state = np.array([state[0], state[1], state[2], state[3], past_action[0], past_action[1], current_dis, heading, target[0], target[1]])
        reward = self.setReward(done, arrive)
        #print('current target :', target)
        return state, reward, done, arrive

    def reset(self):
        '''
        Reset the envrionment
        Reset bot position

        returns state as np.array

        State contains:
        laserData, heading, distance, obstacleMinRange, obstacleAngle
        '''

        while True:
            # Teleport bot to a random point
            agentX, agentY = self.agentController.teleportRandom()
            new_pos = np.array([agentX, agentY])
            print('Teleporting the robot to:', new_pos)
            if self.calcDistance(self.targetPointX, self.targetPointY, agentX, agentY) > self.minCrashRange:
                break
            else:
                print('Reteleporting the bot!')
                #time.sleep(1)

        if self.isTargetReached:
            while True:
                self.targetPointX, self.targetPointY = self.goalCont.calcTargetPoint()
                if self.calcDistance(self.targetPointX, self.targetPointY, agentX, agentY) > self.minCrashRange:
                    self.isTargetReached = False
                    break
                else:
                    print('Recalculating the target point!')
                    #time.sleep(1)

        # Unpause simulation to make observation
        speed = 0
        delta = 0
        self.robotstate[0] = agentX
        self.robotstate[1] = agentY
        self.robotstate[2] = speed
        self.robotstate[3] = np.deg2rad(np.random.uniform(0,360))
        state, current_dis, heading, target, done, arrive = self.getState(self.move(speed, delta))
        #print('state before: ', state)
        state = np.array([state[0], state[1], state[2], state[3], 0, 0, current_dis, heading, target[0], target[1]])
       
        return state # Return state

    

class AgentPosController():
    '''
    This class control robot position
    We teleport our agent when environment reset
    So agent start from different position in every episode
    '''
    def __init__(self):
        global istrain

    def teleportRandom(self):
        '''
        Teleport agent return new x and y point

        return agent posX, posY in list
        '''

        
        xy_list = [
                [0.95, 0.95], [2.2, 0.2], [3.2, 1.5], [5.4, 1.5],
                [1, 4.4], [3.5, 4.4], [5.4, 4.4], [1, 7.3],
                [3.5, 7.3], [5.4, 7.3], [3.2, 0.2], [1.1, 3.4]
            ]
    
    
        # xy_list = [
        #         [2.2, 0.2]
        #     ]
            
        
        # Get random position for agent
        robotX,robotY = random.choice(xy_list)
    
        return robotX, robotY

class GoalController():
    """
    This class controls target model and position
    """
    def __init__(self):
        global istrain
        self.targetPointX = None  # Initial positions
        self.targetPointY = None
        self.last_targetPointX = self.targetPointX 
        self.last_targetPointY = self.targetPointY


    def calcTargetPoint(self):
        """
        This function return a target point randomly for robot
        """
        
        # Wait for deleting
        self.last_targetPointX = self.targetPointX
        self.last_targetPointY = self.targetPointY
        
        goal_xy_list = [
                [0.95, 0.9], [2.2, 0.2], [3.2, 1.5], [5.4, 1.5],
                [1.0, 4.4], [3.5, 4.4], [5.4, 4.4], [1.0, 7.3],
                [3.5, 7.3], [5.4, 7.3], [3.2, 0.2], [1.0, 3.4]
            ]
    
        # goal_xy_list = [
        #         [5.4, 7.3]
        #     ]
    
        # Check last goal position not same with new goal
        while True:
            self.targetPointX, self.targetPointY = random.choice(goal_xy_list)

            if self.last_targetPointX != self.targetPointX:
                if self.last_targetPointY != self.targetPointY:
                    break


        self.last_targetPointX = self.targetPointX
        self.last_targetPointY = self.targetPointY
        # Inform user
        print('New goal position x: ', self.targetPointX)
        print('New goal position y: ', self.targetPointY)

        return self.targetPointX, self.targetPointY

class Render():
    def __init__(self):
        obs1 = [2.2, 3.4]#[30, 30]
        obs2 = [4.2, 3.4]#[30, 80] 
        obs3 = [2.2, 5.4]#[80, 30]
        obs4 = [4.2, 5.4]
        self.obs = np.vstack([obs1, obs2, obs3,obs4]) 

        # square1 = self.make_square(20,20,20)
        # square2 = self.make_square(20,80,20)
        # square3 = self.make_square(80,20,20)
        # square4 = self.make_square(80,80,20)
        # self.obs = np.vstack([square1,square2,square3,square4])

        self.margin = 0
        #coordinates are in [x,y] format
        self.car_length = 50
        self.car_width = 27
        self.wheel_length = 10
        self.wheel_width = 4
        self.wheel_positions = np.array([[15,12],[15,-12],[-15,12],[-15,-12]])
            
        self.color = np.array([0,0,255])/255
        self.wheel_color = np.array([20,20,20])/255
        

        self.car_struct = np.array([[+self.car_length/2, +self.car_width/2],
                                    [+self.car_length/2, -self.car_width/2],  
                                    [-self.car_length/2, -self.car_width/2],
                                    [-self.car_length/2, +self.car_width/2]], 
                                    np.int32)
        
        self.wheel_struct = np.array([[+self.wheel_length/2, +self.wheel_width/2],
                                      [+self.wheel_length/2, -self.wheel_width/2],  
                                      [-self.wheel_length/2, -self.wheel_width/2],
                                      [-self.wheel_length/2, +self.wheel_width/2]], 
                                      np.int32)

        #height and width
        self.background = np.ones((1000+20*self.margin,1000+20*self.margin,3))
        self.background[10:1000+20*self.margin:10,:] = np.array([200,200,200])/255
        self.background[:,10:1000+20*self.margin:10] = np.array([200,200,200])/255
    

    def rotate_car(self, pts, angle=0):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        return ((R @ pts.T).T).astype(int)

    def render(self, x, y, psi, delta, cur_goal_x, cur_goal_y):
        # x,y in 100 coordinates
        x = int(100*x)
        y = int(100*y)
        cur_goal_x = int(100*cur_goal_x)
        cur_goal_y = int(100*cur_goal_y)
       
        
        
        # x,y in 1000 coordinates
        # adding car body
        rotated_struct = self.rotate_car(self.car_struct, angle=psi)
        rotated_struct += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        rendered = cv2.fillPoly(self.background.copy(), [rotated_struct], self.color)

        # adding wheel
        rotated_wheel_center = self.rotate_car(self.wheel_positions, angle=psi)
        for i,wheel in enumerate(rotated_wheel_center):
            
            if i <2:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=delta+psi)
            else:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=psi)
            rotated_wheel += np.array([x,y]) + wheel + np.array([10*self.margin,10*self.margin])
            rendered = cv2.fillPoly(rendered, [rotated_wheel], self.wheel_color)

        # gel
        gel = np.vstack([np.random.randint(-50,-30,16),np.hstack([np.random.randint(-20,-10,8),np.random.randint(10,20,8)])]).T
        gel = self.rotate_car(gel, angle=psi)
        gel += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        gel = np.vstack([gel,gel+[1,0],gel+[0,1],gel+[1,1]])
        rendered[gel[:,1],gel[:,0]] = np.array([60,60,135])/255
        #cv2.circle(rendered,(cur_goal_x,cur_goal_y), 10, (255,0,0), -5)
        cv2.circle(rendered,(cur_goal_x,cur_goal_y), 10,(200,0,0),-10)
        for i in range (self.obs.shape[0]):
            obs_x = int(self.obs[i,0]*100)
            obs_y = int(self.obs[i,1]*100)
            cv2.circle(rendered, (obs_x,obs_y), 20, (255,255,0), 10)
        cv2.line(rendered,(0,0),(0,920),(0,0,0),3)
        cv2.line(rendered,(0,0),(640,0),(0,0,0),3)
        cv2.line(rendered,(0,920),(640,920),(0,0,0),3)
        cv2.line(rendered,(640,0),(640,920),(0,0,0),3)
        new_center = np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        self.background = cv2.circle(self.background, (new_center[0],new_center[1]), 2, [255/255, 150/255, 100/255], -1)
        #cv2.circle(rendered,(100, 100), 100, (255,255,0), 2)
        rendered = cv2.resize(np.flip(rendered, axis=0), (700,700))
        return rendered
    
    def make_square(self, x, y, width):
        square = np.array([[x-int(width/2),i] for i in range(y-int(width/2),y+int(width/2))] +\
                      [[x+int(width/2),i] for i in range(y-int(width/2),y+int(width/2))] +\
                      [[i,y-int(width/2)] for i in range(x-int(width/2),x+int(width/2))] +\
                      [[i,y+int(width/2)] for i in range(x-int(width/2),x+int(width/2))]) 
        return square