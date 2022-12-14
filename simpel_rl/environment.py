#!/usr/bin/env python3
import os
import numpy as np
import math
from math import pi
import random

# from geometry_msgs.msg import Twist, Point, Pose
# from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry
# from std_srvs.srv import Empty
# from gazebo_msgs.srv import SpawnModel, DeleteModel

diagonal_dis = math.sqrt(2) * (3.6 + 3.8)


class Env():
    def __init__(self, is_training):
        # self.position = Pose()
        # self.goal_position = Pose()
        self.isTargetReached = True
        self.positionX = 0.
        self.positionY = 0.
        self.goal_positionX = 0.
        self.goal_positionY = 0.
        self.targetPointX = 0.
        self.targetPointY = 0.
        self.dt = 0.2
        self.minCrashRange = 0.2
        self.agentController = AgentPosController()
        self.goalCont = GoalController()
        self.psi = 0
        self.v = 0
        self.state = 
        self.L = 0.35
        # self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        # self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        # self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        # self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        # self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        # self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        # self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

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
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4

    def calcDistance(self, x1, y1, x2, y2):
        '''
        Calculate euler distance of given two points

        return distance in float
        '''
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_positionX - self.positionX, self.goal_positionY - self.positionY)
        self.past_distance = goal_distance

        return goal_distance
    
    def move(self, v, delta):
        x_dot = v*np.cos(self.psi)
        y_dot = v*np.sin(self.psi)
        psi_dot = self.v*np.tan(delta)/self.L
        return np.array([[x_dot, y_dot, v, psi_dot]]).T

    def getState(self, state_dot):
        # scan_range = []
        # yaw = self.yaw
        # rel_theta = self.rel_theta
        # diff_angle = self.diff_angle
        # min_range = 0.2
        done = False
        arrive = False
        self.state = self.state + self.dt*state_dot
        self.positionX = self.state[0,0]
        self.positionY = self.state[1,0]
        self.v = self.state[2,0]
        self.psi = self.state[3,0]
        yaw = round(math.degrees(self.psi))
        position = np.array([self.positionX,self.positionY])
        heading = self.calcHeadingAngle(self.goal_positionX, self.goal_positionY, self.psi, self.positionX, self.positionY)
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

        current_distance = math.hypot(self.goal_positionX - self.positionX, self.goal_positionY - self.positionY)
        if current_distance <= self.threshold_arrive:
            # done = True
            arrive = True

        return self.positionX, self.positionY, current_distance, yaw, heading, done, arrive
    
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
        current_distance = math.hypot(self.goal_positionX - self.positionX, self.goal_positionY - self.positionY)
        distance_rate = (self.past_distance - current_distance)

        reward = 500.*distance_rate
        self.past_distance = current_distance

        if done:
            reward = -100.
            self.reset()

        if arrive:
            reward = 120.
            self.goal_positionX = random.uniform(0, 6.4)
            self.goal_positionY = random.uniform(0, 9.2)
            
            self.goal_distance = self.getGoalDistace()
            arrive = False

        return reward
    
    def map_function(x, in_min, in_max, out_min, out_max):
        output = int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
        return output

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]
        
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        acc = linear_vel
        delta = self.map_function(ang_vel, -1, 1, -30, 30)
        stateX, stateY, rel_dis, yaw, heading, done, arrive = self.update_state(self.move(acc,delta))
        state = np.append([stateX, stateY])
        for pa in past_action:
            state.append(pa)
        self.positionX, self.positionY, rel_dis, yaw, heading, done, arrive
        state = np.append[stateX, stateY, rel_dis / diagonal_dis, yaw / 360, heading / 360]
        reward = self.setReward(done, arrive)

        return np.asarray(state), reward, done, arrive

    # def update_state(self, state_dot):
    #     # self.u_k = command
    #     # self.z_k = state
    #     #self.state[0] = self.agentX
    #     max_speed = 0.1
    #     #max_angle_steer = np.deg2rad(30)
    #     self.state = self.state + self.dt*state_dot
    #     if self.state[2] > max_speed:
    #         self.state[2] = max_speed
    #     self.x = self.state[0,0]
    #     self.y = self.state[1,0]
    #     self.v = self.state[2,0]
    #     self.psi = self.state[3,0]
    #     heading = self.calcHeadingAngle(self.targetPointX, self.targetPointY, self.psi, self.x, self.y)
    #     distance = self.calcDistance(self.x, self.y, self.targetPointX, self.targetPointY)

    #     position = np.array([self.x,self.y])
    #     self.target = np.array([self.targetPointX, self.targetPointY])
    #     #print('current x,y is:',position,'current psi',self.psi,'current goal is', self.target)
    #     for i in range(self.obs.shape[0]):
    #         obstacle_x = self.obs[i,0]
    #         obstacle_y = self.obs[i,1]
    #         obstacle = np.array([obstacle_x, obstacle_y])
    #         if np.linalg.norm(position - obstacle) > self.obs_r:
    #             isCrash = False
    #             #print('path are safe')
    #         elif np.linalg.norm(position - obstacle) <= self.obs_r:
    #             isCrash = True
    #             print('colliding with the obstacle')
    #             break
    #     if position[0] > self.wall_max_x or position[0] < self.wall_min_x:
    #         isCrash = True
    #         print('collide with wall')
    #     if position[1] > self.wall_max_y or position[1] < self.wall_min_y:
    #         isCrash = True
    #         print('colliding with the wall')

    #     return np.array([self.x, self.y, self.v, self.psi, distance, heading]), isCrash

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
        state,_ = self.getState(self.move(speed, delta))
        
        self.state[0] = agentX
        self.state[1] = agentY
        #self.state[3] = np.deg2rad(90)
        self.state[3] = np.deg2rad(np.random.uniform(0,360))
        self.targetDistance = state[4]
        self.stateSize = len(state)
        #time.sleep(1)

        return state  # Return state

    # def reset(self):
    #     # Reset the env #
    #     rospy.wait_for_service('/gazebo/delete_model')
    #     self.del_model('target')

    #     rospy.wait_for_service('gazebo/reset_simulation')
    #     try:
    #         self.reset_proxy()
    #     except (rospy.ServiceException) as e:
    #         print("gazebo/reset_simulation service call failed")

        
    #         self.goal_positionX = random.uniform(0, 6.4)
    #         self.goal_positionY = random.uniform(0, 9.2)



    #         self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
    #     except (rospy.ServiceException) as e:
    #         print("/gazebo/failed to build the target")
    #     rospy.wait_for_service('/gazebo/unpause_physics')
    #     data = None
    #     while data is None:
    #         try:
    #             data = rospy.wait_for_message('scan', LaserScan, timeout=5)
    #         except:
    #             pass

    #     self.goal_distance = self.getGoalDistace()
    #     state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
    #     state = [i / 3.5 for i in state]
        
    #     state.append(0)
    #     state.append(0)

    #     state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

    #     return np.asarray(state)

class AgentPosController():
    '''
    This class control robot position
    We teleport our agent when environment reset
    So agent start from different position in every episode
    '''
    def __init__(self):
        self.agent_model_name = "turtlebot3_waffle"

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
        
        # Get random position for agent
        robotX,robotY = random.choice(xy_list)
    
        return robotX, robotY

class GoalController():
    """
    This class controls target model and position
    """
    def __init__(self):
        
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

class GoalController():
    """
    This class controls target model and position
    """
    def __init__(self):
        
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