import gymnasium as gym
from gymnasium import error, spaces, utils
import numpy as np
import math

import time
import serial
import sys
from ardSerial import *

from scipy.spatial.transform import Rotation as R


MAX_EPISODE_LEN = 10000  # Number of steps for one training episode
BOUND_ANGLE = 110
STEP_ANGLE = 11.0 #15 # Maximum angle delta per step
SPEED_FACTOR = 0.18 # float([0, 1]) where 1 = full speed
ANG_FACTOR = 0.012 # Adjust gyro sensitivity.



class OpenCatGymEnv(gym.Env):
    """ 
    Gym environment (stable baselines 3) for OpenCat robots.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        self.state_robot = np.array([])
        self.state_robot_history = np.array([])
        self.jointAngles_history = np.array([])
        self.jointAnglesPrev = np.array([])
        self.boundAngles = np.deg2rad(BOUND_ANGLE)
              
        try:
            self.serialObject = connectPort()
            if self.serialObject!=-1:       
                time.sleep(5)

                print("Initialization successful.")

                # The action space are the 8 joint angles        
                self.action_space = spaces.Box(np.array([-1]*8), np.array([1]*8))

                # The observation space are the torso roll, pitch and the angular velocities and a history of the last 20 joint angles
                self.observation_space = spaces.Box(np.array([-1]*246), np.array([1]*246))
        

        except Exception as e:
            logger.info("Exception")
            closeSerialBehavior()
            raise e


    def step(self, action):
        self.timer = 0
        start_time = time.time()
        jointAngles = self.jointAnglesPrev
        jointAngles = self.read_robot_joints()
        #time.sleep(1/240/2)
        ds = np.deg2rad(STEP_ANGLE*SPEED_FACTOR)  # Maximum joint angle derivative (maximum change per step)
        jointAngles += action * ds   # Change per step including agent action
  
        # Apply joint boundaries       
        jointAngles[0] = np.clip(jointAngles[0], -self.boundAngles, self.boundAngles)       # shoulder_left
        jointAngles[1] = np.clip(jointAngles[1], -self.boundAngles, self.boundAngles)       # elbow_left
        jointAngles[2] = np.clip(jointAngles[2], -self.boundAngles, self.boundAngles)       # shoulder_right
        jointAngles[3] = np.clip(jointAngles[3], -self.boundAngles, self.boundAngles)       # elbow_right
        jointAngles[4] = np.clip(jointAngles[4], -self.boundAngles, self.boundAngles)       # hip_right
        jointAngles[5] = np.clip(jointAngles[5], -self.boundAngles, self.boundAngles)       # knee_right
        jointAngles[6] = np.clip(jointAngles[6], -self.boundAngles, self.boundAngles)       # hip_left
        jointAngles[7] = np.clip(jointAngles[7], -self.boundAngles, self.boundAngles)       # knee_left
        

        # Emulate one degree discretization of motors 
        jointAnglesDeg = np.rad2deg(jointAngles) 
        jointAnglesDegRounded = np.round(jointAnglesDeg)  
        jointAngles = np.deg2rad(jointAnglesDegRounded)
        
        self.control_servos(jointAngles)  
        self.jointAnglesPrev = jointAngles

        # Normalize jointAngles
        jointAngles[0] /= self.boundAngles
        jointAngles[1] /= self.boundAngles
        jointAngles[2] /= self.boundAngles
        jointAngles[3] /= self.boundAngles                   
        jointAngles[4] /= self.boundAngles
        jointAngles[5] /= self.boundAngles                   
        jointAngles[6] /= self.boundAngles          
        jointAngles[7] /= self.boundAngles     
    
        # Every 2nd iteration will be added to the joint history
        if(self.step_counter % 2 == 0): 
            self.jointAngles_history = np.append(self.jointAngles_history, jointAngles)
            self.jointAngles_history = np.delete(self.jointAngles_history, np.s_[0:8])
                   
        # Read robot state (pitch, roll and their derivatives of the torso-link)
        self.state_robot = self.read_robot_state()
        info = {}
        truncated = False
        terminated = False
        reward = 0
        done = False
        
        # Stop criteria of current learning episode: Number of steps or robot fell
        self.step_counter += 1
        if self.step_counter > MAX_EPISODE_LEN:
            self.step_counter_session += self.step_counter
            info["TimeLimit.truncated"] = True # handle termination due to maximum number of steps correctly                  
            truncated = True
            terminated = True
            done = terminated
        elif self.is_fallen(self.state_robot):
            serialWriteByte('d')
            time.sleep(5)
            truncated = False
            terminated = True

        self.observation = np.hstack((self.state_robot, self.jointAngles_history))

        #self.timer =  
        info = {"Timer" : time.time() - start_time}

        time_per_step = time.time() - start_time
        if time_per_step < 25e-3:
            time.sleep(25e-3-time_per_step)
        else:
            pass
                
        print("--- %s seconds ---" % (time.time() - start_time))

        return np.array(self.observation).astype(np.float32), reward, terminated, truncated, info
            
 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
      
        # Set start pose (vector is for simulation model)
        #jointAngles = np.deg2rad(np.array([1, 0, 1, 0, 1, 0, 1, 0])*30)
        jointAngles = np.deg2rad(np.array([1, -1, 1, -1, 1, -1, 1, -1])*45) 
        self.control_servos(jointAngles)
        #time.sleep(1/240)
        self.jointAnglesPrev = jointAngles
        
        # Normalize jointAngles
        jointAngles[0] /= self.boundAngles
        jointAngles[1] /= self.boundAngles
        jointAngles[2] /= self.boundAngles
        jointAngles[3] /= self.boundAngles                   
        jointAngles[4] /= self.boundAngles
        jointAngles[5] /= self.boundAngles                   
        jointAngles[6] /= self.boundAngles          
        jointAngles[7] /= self.boundAngles                 
        

        # Read robot state (pitch, roll and their derivatives of the torso-link)
        self.state_robot = self.read_robot_state()
        time.sleep(2)

        # Initialize robot state history with reset position
        self.jointAngles_history = np.tile(jointAngles, 30)
        self.observation = np.concatenate((self.state_robot, self.jointAngles_history))

        info = {}
        return np.array(self.observation).astype(np.float64), info


    def render(self, mode='human'):
        pass


    def close(self):
        p.disconnect()


    def is_fallen(self, robot_state):
        """ 
        Check if robot is fallen. It becomes "True", when pitch or roll is more than 0.9 rad.
        """
        
        # Transform quaternion to euler angles in radians
        try:
            r = R.from_quat(robot_state[0:4])   
            orientation = r.as_euler('zyx', degrees=False)

            is_fallen = np.fabs(orientation[1]) > 1.57 or np.fabs(orientation[2]) > 1.57
        except:
            is_fallen = False
        
        return is_fallen


    def read_robot_joints(self):
        """ Request joint angles from robot and remap to simulation model.
        """

        joint_angles = []
        serialWriteByte('j')
        raw_data = self.serialObject.Read_Line().decode("utf-8")#np.fromstring(serialObject.data, sep='\t')[-6:]
       
        joint_angles = np.fromstring(raw_data, sep='\t')
        joint_angles = np.deg2rad(joint_angles)

        joint_angles = joint_angles[[0, 4, 1, 5, 2, 6, 3, 7]]
        
        time.sleep(1/240/2)
        return joint_angles


    def read_robot_state(self):
        """ 
        Read robot state: Quaternion, angular velocities and motor angles from serial connection.
        """

        # Request gyro data
        state_robot = []
        serialWriteByte('v')
        raw_data = self.serialObject.Read_Line().decode("utf-8")
        state_robot = np.deg2rad(np.fromstring(raw_data, sep='\t'))
        state_robot = np.concatenate((state_robot[0:4]/1000,  np.clip(state_robot[4:]*ANG_FACTOR, -1, 1)))

        time.sleep(1/240/2) # Sending via serial port and moving the motors needs some time.

        return state_robot


    def control_servos(self, jointAngles): 
        """
        The  vector jointAngles is defined for the simulation model. The mapping between 
        hardware and simulation is described below.


                                HARDWARE    SIMULATION
        SHOULDER_LEFT               0           0             
        SHOULDER_RIGHT              1           2
        HIP_RIGHT                   2           4
        HIP_LEFT                    3           6
        ELBOW_LEFT                  4           1
        ELBOW_RIGHT                 5           3
        KNEE_RIGHT                  6           5
        KNEE_LEFT                   7           7

        SIMULATION --> HARWARE: SIMVECTOR[0, 2, 4, 6, 1, 3, 5, 7]

                                HARDWARE    SIMULATION
        SHOULDER_LEFT               0           0 
        ELBOW_LEFT                  4           1
        SHOULDER_RIGHT              1           2
        ELBOW_RIGHT                 5           3 
        HIP_RIGHT                   2           4
        KNEE_RIGHT                  6           5
        HIP_LEFT                    3           6
        KNEE_LEFT                   7           7  

        HARDWARE --> SIMULATION: HARWAREVECTOR[0, 4, 1, 5, 2, 6, 3, 7]
        """

        empty_servos = [0, 0, 0, 0, 0, 0, 0, 0]
        remapped_angles = list(np.rad2deg(np.array(jointAngles)[[0,2,4,6,1,3,5,7]]))    
        serialWriteNumToByte('L', empty_servos + remapped_angles)
        time.sleep(1/240) # Sending via serial port and moving the motors needs some time.
