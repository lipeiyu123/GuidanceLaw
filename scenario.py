# -*- coding: utf-8 -*-
import numpy as np
from core import World, Agent ,Landmark
import math
import random

AD_END_DISTANCE = 1

class Scenario(object):

        def __init__(self):
                self.random_r = 2500 #+ random.randint(-200 , 200)
        def make_world(self):
            world = World()

            # communication channel dimensionality
            world.dim_c = 2

            num_target_agents = 1
            num_defenfer_agents = 1
            num_agents = num_target_agents + num_defenfer_agents
            num_attackers = 1

            # add agents
            world.agents = [Agent() for i in range(num_agents)]

            #set param for agents
            for i, agent in enumerate(world.agents):
                agent.name = 'target'   if i == 0          else 'defender'
                agent.survived = True
                agent.leader = True if i == 0 else False
                agent.target  = True if i == 0 else False
                agent.silent = True
                agent.accel = 0.5   if i == 0       else    1
                agent.max_speed = 2  if i == 0       else    3
            
            world.landmarks = [Landmark() for i in range(num_attackers)]
            for i, landmarks in enumerate(world.landmarks):
                landmarks.name = 'attacker'
                landmarks.survived = True
                landmarks.accel = 1
                landmarks.max_speed = 3

            self.reset_world(world)
            return world


        def reset_world(self, world):

            self.dist_flag_1250 = False
            self.dist_flag_1000 = False
            self.dist_flag_750 = False
            self.dist_flag_500 = False
            self.dist_flag_250 = False
            self.dist_flag_100 = False
            self.dist_flag_50 = False
            self.dist_flag_10 = False

            for i, agent in enumerate(world.agents):
                if i ==0:
                    agent.state.p_pos = np.array([0, 0],dtype=np.float32)
                    agent.angle = np.array([np.pi / 2], dtype=np.float32)
                    agent.state.p_vel = np.array([150])
                    agent.cmd = 0
                    agent.last_cmd = 0

            for i, landmarks in enumerate(world.landmarks):
                seed = random.uniform(     (np.pi / 4) , (np.pi * 2 / 4)    )
                #print(seed , "seed\n")
                x = math.cos(seed) * self.random_r
                y = math.sin(seed) * self.random_r

                landmarks.state.p_pos = np.array([x, y],dtype=np.float32)

                rp = np.zeros(2) - landmarks.state.p_pos
                random_angle = math.atan2(  rp[1] , rp[0])
                
                #print("seed = " , seed * 180 / np.pi, "      pos = " , landmarks.state.p_pos , "     angle = " , random_angle *180 / np.pi , "\n")

                landmarks.angle = np.array([random_angle], dtype=np.float32)
                landmarks.state.p_vel = np.array([250])

                landmarks.state.p_vel_x = np.array(landmarks.state.p_vel[0] * math.cos(landmarks.angle ))
                landmarks.state.p_vel_y = np.array(landmarks.state.p_vel[0] * math.sin(landmarks.angle  ))
                landmarks.last_relative_position = world.agents[0].state.p_pos  - landmarks.state.p_pos 
            
            for i, agent in enumerate(world.agents):
                if i ==1:

                    agent.state.p_pos = np.array([0, 0],dtype=np.float32)
                    agent.state.p_vel = np.array([250])

                    vax = agent.state.p_vel[0] * math.cos(world.landmarks[0].angle)
                    vay = agent.state.p_vel[0] * math.sin(world.landmarks[0].angle)

                    v_a_array = np.array([ vax , vay ])
                    ad_array = agent.state.p_pos   -   world.landmarks[0].state.p_pos 
                    x = (np.dot(v_a_array , ad_array)) / (np.linalg.norm(v_a_array) * np.linalg.norm(ad_array))                        
                    
                    if x >= 1.0 :
                        x = 1.000000
                    elif    x <= -1.0 :
                        x = -1.00000
                    beta = math.acos(  x  )

                    alpha = math.asin( (world.landmarks[0].state.p_vel * math.sin(beta) ) / agent.state.p_vel)

                    da_array = world.landmarks[0].state.p_pos  - agent.state.p_pos 
                    alpha_0 = math.atan2( da_array[1] , da_array[0])

                    if alpha_0 >= np.pi / 2 :
                        bastangle = alpha_0 - alpha
                    elif alpha_0 < np.pi / 2:
                        bastangle = alpha_0 + alpha

                    agent.angle = np.array([bastangle], dtype=np.float32)
                    agent.cmd = 0
                    agent.last_cmd = 0
                    agent.time = 0

        def has_winner(self,agent, world):
            state_info = [ False for i in range(2)]
            for i, landmarks in enumerate(world.landmarks):
                landmark = landmarks

            for i , temp_agent in enumerate(agent):
                #target 结束条件
                if i == 0 :
                    distance = np.linalg.norm(landmark.state.p_pos - temp_agent.state.p_pos)
                    if distance <10 :
                        print("目标G了!         失败                ")
                        #print(distance)
                        state_info[i] = True
                        return state_info
                #defender 结束条件
                elif i ==1 :

                    distance = np.linalg.norm(landmark.state.p_pos - temp_agent.state.p_pos)
                    '''
                    ad_array = temp_agent.state.p_pos   -   world.landmarks[0].state.p_pos
                    vel_angle = np.array([temp_agent.state.p_vel[0] * math.cos(temp_agent.angle) , temp_agent.state.p_vel[0] * math.sin(temp_agent.angle)])
                    x = (np.dot(vel_angle , -ad_array)) / (np.linalg.norm(vel_angle) * np.linalg.norm(-ad_array))
                    if x >= 1.0 :
                        x = 1.000000
                    elif    x <= -1.0 :
                        x = -1.00000
                    fov = math.acos( x )* 180 / np.pi 
                    if fov > 40:
                        print("跟丢了!            失败!       ")
                        state_info[i] = True                                       
                    '''
                    if distance < AD_END_DISTANCE :
                        print("刺客G了!         成功!       distance =" , distance ,"       NBNBNBNB yes                ")
                        state_info[i] = True
                        return state_info

            return  state_info


        def reward(self, agent, world):
            # Agents are rewarded based on minimum agent distance to each landmark
            #boundary_reward = -10 if self.outside_boundary(agent) else 0
            main_reward = self.target_reward(agent, world) if agent.target else self.defender_reward(agent, world)          
            return main_reward

        def defender_reward(self, agent, world):
            # Agents are rewarded based on minimum agent distance to each landmark
            rew = 0
            adversaries = world.landmarks
            for adv in adversaries:
                vax = world.landmarks[0].state.p_vel[0] * math.cos(world.landmarks[0].angle)
                vay = world.landmarks[0].state.p_vel[0] * math.sin(world.landmarks[0].angle)
                v_a_array = np.array([ vax , vay ])         
                ad_array = agent.state.p_pos   -   world.landmarks[0].state.p_pos
                
                x = (np.dot(v_a_array , ad_array)) / (np.linalg.norm(v_a_array) * np.linalg.norm(ad_array))                        

                if x >= 1.0 :
                    x = 1.000000
                elif    x <= -1.0 :
                    x = -1.00000
                beta = math.acos(  x  )

                x = (world.landmarks[0].state.p_vel * math.sin(beta) ) / agent.state.p_vel
                if x > 1.0 or x < -1.0:
                    rew = 0
                    return rew 
                alpha = math.asin(x )

                da_array = world.landmarks[0].state.p_pos  - agent.state.p_pos 
                alpha_0 = math.atan2( da_array[1] , da_array[0])

                if alpha_0 >= np.pi / 2 :
                    bastangle = alpha_0 - alpha
                elif alpha_0 < np.pi / 2:
                    bastangle = alpha_0 + alpha
                
                vel_angle = np.array([agent.state.p_vel[0] * math.cos(agent.angle) , agent.state.p_vel[0] * math.sin(agent.angle)])
                
                bast_angle = np.array([agent.state.p_vel[0] * math.cos(bastangle ) , agent.state.p_vel[0] * math.sin( bastangle)])
                x = (np.dot(vel_angle , bast_angle)) / (np.linalg.norm(vel_angle) * np.linalg.norm(bast_angle))
                if x >= 1.0 :
                    x = 1.000000
                elif    x <= -1.0 :
                    x = -1.00000
                theat = math.acos( x )
                
                x = (np.dot(vel_angle , -ad_array)) / (np.linalg.norm(vel_angle) * np.linalg.norm(-ad_array))
                if x >= 1.0 :
                    x = 1.000000
                elif    x <= -1.0 :
                    x = -1.00000
                fov = math.acos( x )

                distance = np.linalg.norm(agent.state.p_pos - adv.state.p_pos)
            
                #rew +=  0.5* ( math.exp(-1.46 *abs( theat) ) ) + \
                #                2.5*( math.exp(-0.012 *abs(agent.cmd  ) ) )
                rew += 3.2 * ( ((9.8 * 15) - abs(agent.cmd) ) / ( 9.8 * 15 ) )
                                # * ( (np.pi - abs(fov) ) / np.pi  ) + \
                               #3 * ( ((9.8 * 40) - abs(agent.cmd) ) / ( 9.8 * 40 ) )
                                #( math.exp(-1 *abs((agent.cmd  - agent.last_cmd) / (9.8 * 80))) )
                                #( math.exp(-0.8 *abs( fov) ) ) + \
                #print("rew" , fov)  
                if fov > 40:
                    rew -=1
                rew = rew / 3000
                    #return rew

                if distance < AD_END_DISTANCE * 1250      and     not self.dist_flag_1250:
                    rew += 0.0625
                    self.dist_flag_1250 = True
                if distance < AD_END_DISTANCE * 1000      and     not self.dist_flag_1000:
                    rew += 0.0625
                    self.dist_flag_1000 = True
                if distance < AD_END_DISTANCE * 750      and     not self.dist_flag_750:
                    rew += 0.0625
                    self.dist_flag_750 = True
                if distance < AD_END_DISTANCE * 500      and     not self.dist_flag_500:
                    rew += 0.0625
                    self.dist_flag_500 = True
                if distance < AD_END_DISTANCE * 250      and     not self.dist_flag_250:
                    rew += 0.0625
                    self.dist_flag_250 = True
                if distance < AD_END_DISTANCE * 100      and     not self.dist_flag_100:
                    rew += 0.0625
                    self.dist_flag_100 = True
                if distance < AD_END_DISTANCE * 50      and     not self.dist_flag_50:
                    rew += 0.0625  
                    self.dist_flag_50 = True   
                if distance < AD_END_DISTANCE * 10      and     not self.dist_flag_10:
                    rew += 0.0625   
                    self.dist_flag_10 = True
                if distance < AD_END_DISTANCE:
                    rew += 0.5          

            return rew


        def target_reward(self, agent, world):
            # Agents are rewarded based on minimum agent distance to each landmark
            rew = 0
            adversaries = world.landmarks
            for adv in adversaries:
                vel_angle = np.array([np.array(agent.state.p_vel[0] * math.cos(agent.angle )) , np.array(agent.state.p_vel[0] * math.sin(agent.angle )) ])
                los = adv.state.p_pos - agent.state.p_pos
                x = (np.dot(vel_angle , los)) / (np.linalg.norm(vel_angle) * np.linalg.norm(los))
                if x >= 1.0 :
                    x = 1.000000
                elif    x <= -1.0 :
                    x = -1.00000
                theat = math.acos( x ) 
                #rew = math.exp( -1.46* ( ( abs( theat )) ) ) + \
                #            ( math.exp(-0.052 *abs(agent.cmd)) ) 
                rew += 1 * ( abs(theat)  / np.pi  ) + \
                               1 * ( ((9.8 * 9) - abs(agent.cmd) ) / ( 9.8 * 9 ) )

                distance = np.linalg.norm(agent.state.p_pos - adv.state.p_pos)

            rew = rew / 2000
            if distance < 10 :
                rew -= 1
            return rew

        def get_radir_detect(self , self_p , self_angle , enemy_p , normalization_param =1):

            theat = self_angle - np.array([np.pi/2])
            p = enemy_p - self_p 
            p_0 = math.cos(theat) * p[0] + math.sin(theat) * p[1]
            p_1 = math.cos(theat) * p[1] - math.sin(theat) * p[0]
            dist_angle = np.array([np.linalg.norm(enemy_p - self_p) / normalization_param , math.atan2(p_1 , p_0) / (np.pi / 2) ] ,dtype=np.float32)
            return  dist_angle 


        def observation(self, agent, world):

            firend_obs = []
            enemy_obs = []
            bool_hit = []
            normalization_p = 2500
            normalization_v = 250
            normalization_angle = 2* np.pi

            pa = world.landmarks[0].state.p_pos
            if agent.target :
                pt = agent.state.p_pos
                angle_t = agent.angle
                for other_agent in world.agents:
                    if other_agent == agent:continue
                    pd = other_agent.state.p_pos
                firend_obs.append( self.get_radir_detect(pt , angle_t , pd , normalization_param = normalization_p) )
                enemy_obs.append( self.get_radir_detect(pt , angle_t , pa , normalization_param = normalization_p) )
                distance = np.linalg.norm(pt  - pa)
                if  distance    <   10 :
                    bool_hit.append([-1])
                else:
                    bool_hit.append([1])

            if  not agent.target :
                pd = agent.state.p_pos
                angle_d = agent.angle
                for other_agent in world.agents:
                    if other_agent == agent:continue
                    pt = other_agent.state.p_pos
                firend_obs.append( self.get_radir_detect(pd , angle_d , pt , normalization_param = normalization_p) )
                enemy_obs.append( self.get_radir_detect(pd , angle_d , pa , normalization_param = normalization_p) )
                distance = np.linalg.norm(pd  - pa)
                if  distance    <   AD_END_DISTANCE :
                    bool_hit.append([-1])
                else:
                    bool_hit.append([1])
            return np.concatenate([agent.state.p_vel / normalization_v] + [agent.angle / normalization_angle] + firend_obs + enemy_obs + bool_hit)

