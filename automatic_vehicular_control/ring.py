from automatic_vehicular_control.exp import *
from automatic_vehicular_control.env import *
from automatic_vehicular_control.u import *
import copy 

class buffer() : 

    def __init__(self,obs_size=4,batch_size=32,replay_size=2000000,prediction_horizon=10,num_relations=1,action_dim=1) : 
        # This buffer will only handle fixed number of object relations (currently 1), if the number of relations are not fixed, then a different
        # setup for the buffer will probably be needed 
        self.num_relations = num_relations
        self.fr_u = np.zeros((replay_size,self.num_relations,obs_size+action_dim)) 
        self.fr_v = np.zeros((replay_size,self.num_relations,obs_size+action_dim)) 
        self.fo_in = np.zeros((replay_size,obs_size+action_dim)) 
        self.actions =  np.zeros((replay_size,action_dim)) 
        self.y     = np.zeros((replay_size,obs_size)) 
        self.ptr, self.size, self.max_size = 0, 0,replay_size
        self.batch_size = batch_size 
        self.trajectory = {} 
        self.ground_truth = {} 
        self.o_size = obs_size
        self.horizon = prediction_horizon
        self.multi_step_struct =[] 
        self.relation_map = [] 
        self.ep_len = 0
        self.action_dim = action_dim 
        self.max_struct_size, self.struct_ptr, self.reached_max = 30000, 0 , 0  

    def step_update(self,o,r,a) : 
        self.relation_map.append(r) 
        for o_key,obs2 in o.items() :
            if o_key not in self.trajectory : 
                self.trajectory[o_key] = [ ] 
                self.ground_truth[o_key] = [ ] 
            obs = copy.deepcopy(obs2)
            obs.append(a[o_key]) 
            #Check number of relations  
            relations = r[o_key]
            num_r = len(relations) 
            # Make appropriate dimension input for given object 
            u =  np.repeat(np.expand_dims(obs,axis=0), repeats=num_r,axis=0) 
            # Make appropriate dimension input for relations 
            v = np.zeros((num_r,self.o_size+self.action_dim)) 
            # Make appropriate dimension input for actons 
            w = np.zeros((1,self.action_dim))
            w[0] = a[o_key]
            
            for i in range(num_r) :
                # Id of vehicle 
                r_key= relations[i] 
                # Corresponding obs for that vehicle 
                r_obs = copy.deepcopy(o[r_key]) 
                r_obs.append(a[r_key])
                # Add this relation object to v  
                v[i] = np.array(r_obs)   
            
            # To trajectory append (my obs, n copies of my obs where n = number of relations, n obs of related vehicles)
            self.trajectory[o_key].append([obs,u,v,w,obs2])  
            # If this is not the initial step, then append ground truth observation as well. By skipping the first step, we automatically ensure that ground 
            # truths (y(t)) correspond to (x(t-1)) 
            if len(self.trajectory[o_key]) != 1:      
                self.ground_truth[o_key].append(obs2) 
        self.ep_len +=1 

    def push_to_buffer(self) : 
        # For each object (vehicle) in the trajectory 
        for k in self.trajectory :
            # Remove last element, since we will not have the ground truth for the last element 
            self.trajectory[k].pop() 
            # Sanity check assertion, 
            assert(len(self.trajectory[k])) == len(self.ground_truth[k]) 
            # Build up the buffer 
            for item,y in zip(self.trajectory[k],self.ground_truth[k]) :
                if k in [0,'0'] : 
                    pass 
                else : 
                    if np.random.rand()>0.2 : 
                        break  
                # my observation 
                self.fo_in[self.ptr] = item[0] 
                # my observation repated n times
                self.fr_u[self.ptr] = item[1] 
                # related vehicles obs repeated n times (n= number of relations)
                self.fr_v[self.ptr] = item[2]  
                self.actions[self.ptr] = item[3] 
                # ground truth 
                self.y[self.ptr]   = y 
                # 2 updates to maintain insertion ptr for entry into buffer
                self.ptr = (self.ptr+1)% self.max_size
                self.size = min(self.size+1,self.max_size)

    def one_step_push(self,t) : 
        # This currently only supports fixed number of relations for the system i.e., each object should have r relations throughout training
        num_objects = len(self.trajectory) 
        new_struct = {'fr_u':np.zeros((num_objects,self.num_relations,self.o_size+self.action_dim)),'fr_v':np.zeros((num_objects,self.num_relations,self.o_size+self.action_dim)),
        'fo_in':np.zeros((num_objects,self.o_size+self.action_dim))} 
        # Building up a data structure for multi-step prediction
        for k in range(1,self.horizon) : 
            # Ground truths for next self.horizon steps 
            new_struct['y_{}'.format(k)] = np.zeros((num_objects,self.o_size)) 
            new_struct['a_{}'.format(k)] = np.zeros((num_objects,self.action_dim)) 

        # Current relation map (this is fixed for the ring env but I assume dynamic relations) 
        relations = self.relation_map[t] 
        # We have to rebuild the relation map because we lost the order through indexing from 0 to num_objects (done below)
        r_map = np.zeros((num_objects)) 
        veh_order =[] 
        for i,k in enumerate(self.trajectory) : 
                # For each object, store the relevant observations at timestep t 
                # My own obs 
                new_struct['fo_in'][i] = self.trajectory[k][t][0] 
                # My own obs, repeated num_relations times? 
                new_struct['fr_u'][i] = self.trajectory[k][t][1]
                # Observations of related vehicles  
                new_struct['fr_v'][i] = self.trajectory[k][t][2] 
                for j in range(1,self.horizon) : 
                    # Appropriate ground truth j steps ahead, j up to horizon
                    new_struct['y_{}'.format(j)][i] = self.trajectory[k][t+j-1][4] 
                    new_struct['a_{}'.format(j)][i] = self.trajectory[k][t+j-1][3] 
            
                # Keeping track of the vehicle id at ith index
                veh_order.append(int(k))

        # For each object 
        for i,k in enumerate(self.trajectory) : 
            # Find the vehicle id of my relation
            leader_id = relations[k][0]  
            #Find the index that vehicle id was inserted at 
            leader_ind = veh_order.index(int(leader_id)) 
            # Mark the relation map to point to that related vehicle 
            r_map[i] = leader_ind
        
        # Store the new relation map (which is equivalent to the original relations but just maps to indexes instead of ids)
        new_struct['r'] = r_map 
        new_struct['num_objects'] = num_objects
        if self.reached_max :
            self.multi_step_struct[self.struct_ptr] = new_struct 
            self.struct_ptr = (self.struct_ptr+1) % self.max_struct_size 
        else :
            self.multi_step_struct.append(new_struct)
            self.reached_max = len(self.multi_step_struct) == self.max_struct_size 

  
    def sample_batch(self) :
        # Sample random idxs from buffer 
        # fr_u,fr_v are input to the relation model, effects (obtained using relation model) and fo_u are input to the object model, y is the ground truth
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return dict(fr_u=self.fr_u[idxs],
                    fr_v=self.fr_v[idxs],
                    fo_u=self.fo_in[idxs],
                    y=self.y[idxs])

    def sample_multi_step(self) : 
        # To test multi-step prediction for all vehicles at a given timestep, a single element from the multi_step_struct is sampled (This is not used for training
        # but is used for evaluating the performance of the model on long horizon predictions)
        ind = np.random.randint(0,len(self.multi_step_struct)) 
        return self.multi_step_struct[ind] 
    
    def end_episode(self): 
        # Pushing to buffer just builds up the training buffer 
        self.push_to_buffer()
        for _ in range(500) :
            t= np.random.randint(0,self.ep_len-self.horizon)
            #for t in range(self.ep_len - self.horizon) :
            try :
                self.one_step_push(t)
            except :
                print('Error building up multi-step struct, breaking')
                break  
        # Reset quantities which just need to be stored for the duration of a trajectory  
        self.ep_len = 0 
        self.trajectory = {} 
        self.ground_truth = {} 
        self.relation_map = [] 

class RingEnv(Env):
    def def_sumo(self):
        c = self.c
        r = c.circumference / (2 * np.pi)
        nodes = E('nodes',
            E('node', id='bottom', x=0, y=-r),
            E('node', id='top', x=0, y=r),
        )

        get_shape = lambda start_angle, end_angle: ' '.join('%.5f,%.5f' % (r * np.cos(i), r * np.sin(i)) for i in np.linspace(start_angle, end_angle, 80))
        edges = E('edges',
            E('edge', **{'id': 'right', 'from': 'bottom', 'to': 'top', 'length': c.circumference / 2, 'shape': get_shape(-np.pi / 2, np.pi / 2), 'numLanes': c.n_lanes}),
            E('edge', **{'id': 'left', 'from': 'top', 'to': 'bottom', 'length': c.circumference / 2, 'shape': get_shape(np.pi / 2, np.pi * 3 / 2), 'numLanes': c.n_lanes}),
        )

        connections = E('connections',
            *[E('connection', **{'from': 'left', 'to': 'right', 'fromLane': i, 'toLane': i}) for i in range(c.n_lanes)],
            *[E('connection', **{'from': 'right', 'to': 'left', 'fromLane': i, 'toLane': i}) for i in range(c.n_lanes)],
        )

        additional = E('additional',
            E('vType', id='human', **{**IDM, **LC2013, **dict(accel=1, decel=1.5, minGap=2, sigma=c.sigma)}),
            E('vType', id='rl', **{**IDM, **LC2013, **dict(accel=1, decel=1.5, minGap=2, sigma=0)}),
            *build_closed_route(edges, c.n_veh, c.av, space=c.initial_space)
        )
        return super().def_sumo(nodes, edges, connections, additional)

    def reset_sumo(self):
        c = self.c
        if c.circumference_range:
            c.circumference = np.random.randint(*c.circumference_range)
        return super().reset_sumo()

    @property
    def stats(self):
        c = self.c
        stats = {k: v for k, v in super().stats.items() if 'flow' not in k}
        stats['circumference'] = c.circumference
        return stats

    def step(self, action=None):
        c = self.c
        ts = self.ts
        max_speed = c.max_speed
        circ_max = max_dist = c.circumference_max
        circ_min = c.circumference_min
        rl_type = ts.types.rl

        if c.no_rl : 
            action = None 
        
        if not rl_type.vehicles:
            super().step()
            return c.observation_space.low, 0, False, 0

        rl = nexti(rl_type.vehicles)
        if action is not None: # action is None only right after reset
            ts.tc.vehicle.setMinGap(rl.id, 0) # Set the minGap to 0 after the warmup period so the vehicle doesn't crash during warmup
            accel, lc = (action, None) if not c.lc_av else action if c.lc_act_type == 'continuous' else (action['accel'], action['lc'])
            if isinstance(accel, np.ndarray): accel = accel.item()
            if isinstance(lc, np.ndarray): lc = lc.item()
            if c.norm_action and isinstance(accel, (float, np.floating)):
                accel = (accel - c.low) / (c.high - c.low)
            if c.norm_action and isinstance(lc, (float, np.floating)):
                lc = bool(np.round((lc - c.low) / (c.high - c.low)))

            if c.get('handcraft'):
                accel = (0.75 * np.sign(c.handcraft - rl.speed) + 1) / 2
                lc = True
                if c.get('handcraft_lc'):
                    if c.handcraft_lc == 'off':
                        lc = False
                    elif c.handcraft_lc == 'stabilize':
                        other_lane = rl.lane.left or rl.lane.right
                        oleader, odist = other_lane.next_vehicle(rl.laneposition, route=rl.route)
                        ofollower, ofdist = other_lane.prev_vehicle(rl.laneposition, route=rl.route)
                        if odist + ofdist < 7 and odist > 3:
                            lc = True
                        else:
                            lc = False

            if c.act_type == 'accel_discrete':
                ts.accel(rl, accel / (c.n_actions - 1))
            elif c.act_type == 'accel':
                if c.norm_action:
                    accel = (level * 2 - 1) * (c.max_accel if accel > 0.5 else c.max_decel)
                ts.accel(rl, accel)
            else:
                if c.act_type == 'continuous':
                    level = accel
                elif c.act_type == 'discretize':
                    level = min(int(accel * c.n_actions), c.n_actions - 1) / (c.n_actions - 1)
                elif c.act_type == 'discrete':
                    level = accel / (c.n_actions - 1)
                ts.set_max_speed(rl, max_speed * level)
            if c.n_lanes > 1:
                if c.symmetric_action if c.symmetric_action is not None else c.symmetric:
                    if lc:
                        ts.lane_change(rl, -1 if rl.lane_index % 2 else +1)
                else:
                    ts.lane_change_to(rl, lc)

        obs_dict = {} 
        relations_dict = {} 
        actions_dict = {} 

        for veh in ts.types.human.vehicles:
            leader, dist = veh.leader() 
            obs = [veh.speed / max_speed, leader.speed / max_speed, dist / max_dist,0]
            obs_dict[veh.id] = obs 
            relations_dict[veh.id] = [leader.id]   
            actions_dict[veh.id]  = 0 
        
        if action is None : 
            a = -1 
        else :
            a= action[0]

        for veh in ts.types.rl.vehicles : 
            leader, dist = veh.leader() 
            obs = [veh.speed / max_speed, leader.speed / max_speed, dist / max_dist,1]
            obs_dict[veh.id] = obs 
            relations_dict[veh.id] = [leader.id]  
            actions_dict[veh.id]  = a  
  
        self.c.buffer.step_update(obs_dict,relations_dict,actions_dict)
        
        super().step()

        if len(ts.new_arrived | ts.new_collided):
            print('Detected collision')
            return c.observation_space.low, -c.collision_penalty, True, None
        elif len(ts.vehicles) < c.n_veh:
            print('Bad initialization occurred, fix the initialization function')
            return c.observation_space.low, 0, True, None

        leader, dist = rl.leader()
        if c.n_lanes == 1:
            obs = [rl.speed / max_speed, leader.speed / max_speed, dist / max_dist]
            if c.circ_feature:
                obs.append((c.circumference - circ_min) / (circ_max - circ_min))
            if c.accel_feature:
                obs.append(0 if leader.prev_speed is None else (leader.speed - leader.speed) / max_speed)
        elif c.n_lanes == 2:
            lane = rl.lane
            follower, fdist = rl.follower()
            if c.symmetric:
                other_lane = rl.lane.left or rl.lane.right
                oleader, odist = other_lane.next_vehicle(rl.laneposition, route=rl.route)
                ofollower, ofdist = other_lane.prev_vehicle(rl.laneposition, route=rl.route)
                obs = np.concatenate([
                    np.array([rl.speed, leader.speed, oleader.speed, follower.speed, ofollower.speed]) / max_speed,
                    np.array([dist, odist, fdist, ofdist]) / max_dist
                ])
            else:
                obs = [rl.speed]
                for lane in rl.edge.lanes:
                    is_rl_lane = lane == rl.lane
                    if is_rl_lane:
                        obs.extend([is_rl_lane, dist, leader.speed, fdist, follower.speed])
                    else:
                        oleader, odist = lane.next_vehicle(rl.laneposition, route=rl.route)
                        ofollower, ofdist = lane.prev_vehicle(rl.laneposition, route=rl.route)
                        obs.extend([is_rl_lane, odist, oleader.speed, ofdist, ofollower.speed])
                obs = np.array(obs) / [max_speed, *([1, max_dist, max_speed, max_dist, max_speed] * 2)]
        else:
            obs = [rl.speed]
            follower, fdist = rl.follower()
            for lane in rl.edge.lanes:
                is_rl_lane = lane == rl.lane
                if is_rl_lane:
                    obs.extend([is_rl_lane, dist, leader.speed, fdist, follower.speed])
                else:
                    oleader, odist = lane.next_vehicle(rl.laneposition, route=rl.route)
                    ofollower, ofdist = lane.prev_vehicle(rl.laneposition, route=rl.route)
                    obs.extend([is_rl_lane, odist, oleader.speed, ofdist, ofollower.speed])
            obs = np.array(obs) / [max_speed, *([1, max_dist, max_speed, max_dist, max_speed] * 3)]
        obs = np.clip(obs, 0, 1) * (1 - c.low) + c.low
        reward = np.mean([v.speed for v in (ts.vehicles if c.global_reward else rl_type.vehicles)])
        if c.accel_penalty and hasattr(self, 'last_speed'):
            reward -= c.accel_penalty * np.abs(rl.speed - self.last_speed) / c.sim_step
        self.last_speed = rl.speed

        return obs.astype(np.float32), reward, False, None

class Ring(Main):
    def create_env(c):
        return NormEnv(c, RingEnv(c))

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(c._n_obs,), dtype=np.float32)

    @property
    def action_space(c):
        c.setdefaults(lc_av=False)
        assert c.act_type in ['discretize', 'discrete', 'continuous', 'accel', 'accel_discrete']
        if c.act_type in ['discretize', 'continuous', 'accel']:
            if not c.lc_av or c.lc_act_type == 'continuous':
                return Box(low=c.low, high=c.high, shape=(1 + bool(c.lc_av),), dtype=np.float32)
            elif c.lc_act_type == 'discrete':
                return Namespace(accel=Box(low=c.low, high=c.high, shape=(1,), dtype=np.float32), lc=Discrete(c.lc_av))
        elif c.act_type in ['discrete', 'accel_discrete']:
            if c.lc_av:
                return Namespace(accel=Discrete(c.n_actions), lc=Discrete(c.lc_av))
            return Discrete(c.n_actions)

    def on_train_start(c):
        super().on_train_start()
        if c.get('last_unbiased'):
            c._model.p_head[-1].bias.data[c.lc_av:] = 0
        c.buffer = buffer(prediction_horizon=c.prediction_horizon)

    def on_step_end(c, gd_stats):
        super().on_step_end(gd_stats)
        if c.get('last_unbiased'):
            c._model.p_head[-1].bias.data[c.lc_av:] = 0

if __name__ == '__main__':
    c = Ring.from_args(globals(), locals()).setdefaults(
        n_lanes=1,
        horizon=1000,
        warmup_steps=1000,
        sim_step=0.1,
        av=1,
        max_speed=10,
        max_accel=0.5,
        max_decel=0.5,
        circumference=250,
        circumference_max=300,
        circumference_min=200,
        circumference_range=None,
        initial_space='random_free',
        sigma=0.2,
        n_workers=1, 

        circ_feature=False,
        accel_feature=False,
        act_type='continuous',
        lc_act_type='discrete',
        low=-1,
        high=1,
        norm_action=True,
        global_reward=False,
        accel_penalty=0,
        collision_penalty=100,

        n_steps=100,
        gamma=0.999,
        alg='TRPO',
        norm_reward=True,
        center_reward=True,
        adv_norm=False,
        step_save=None,
        wb=True, tb = False, 
        tag='trial',
        

        render=False,
    )
    if c.n_lanes == 1:
        c.setdefaults(n_veh=22, _n_obs=3 + c.circ_feature + c.accel_feature)
    elif c.n_lanes == 2:
        c.setdefaults(n_veh=44, lc_mode=LC_MODE.no_lat_collide, symmetric=False, symmetric_action=None, lc_av=2)
        c._n_obs = (1 + 2 * 2 * 2) if c.symmetric else (1 + 2 * 5)
    elif c.n_lanes == 3:
        c.setdefaults(n_veh=66, lc_mode=LC_MODE.no_lat_collide, symmetric=False, symmetric_action=None, lc_av=3, _n_obs=1 + 3 * (1 + 2 * 2))
    c.step_save = c.step_save or min(5, c.n_steps // 10)
    c.redef_sumo = bool(c.circumference_range)
    c.run()





#class graph() :
#    def __init__(self,o_size,objects,relations) : 
#        self.__init__ 
#        self.o_size= int(o_size)
#        self.objects = objects
#        self.relations = relations 
        
#    def relation_inputs(self,o,r) :
#        r_inputs = {} 
#        for o_key,obs in o.items() :
#            num_r = len(r[o_key]) 
#            r_input = np.zeros((num_r,2*self.o_size))
#            for i in range(num_r) :
#                r_key= r[o_key][i]
#                r_input[i] = np.concatenate((np.array(obs),np.array(o[r_key]))) 
#            r_inputs[o_key] = r_input 
#        return r_inputs 

#    def multi_step_prediction(self,o,r,k) : 
#        predictions = {} 
#        for t in range(1,k+1) :
#            r_inputs = self.relation_inputs(o,r)
#            o_next,r_next = self.one_step_prediction(o,r_inputs)   
#            predictions[t] = o_next  
#            o,r = o_next,r_next 
#        return predictions 

#    def one_step_prediction(self,o,r_inputs): 
#        one_step_pred = {} 
#        for key,obs in o.items() : 
#            errors = self.rel_f(r_inputs) 
#            agg_errors = np.sum(errors,axis=1)  
#            next_obs = self.obj_f(obs,agg_errors) 
#            one_step_pred[key] = next_obs
#         return one_step_pred 