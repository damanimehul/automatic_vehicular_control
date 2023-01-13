from automatic_vehicular_control.exp import *
from automatic_vehicular_control.env import *
from automatic_vehicular_control.u import *

class new_buffer() : 

    def __init__(self,obs_size=4,batch_size=1,replay_size=2000000,prediction_horizon=10,action_dim=1) : 
        self.obs_dim = obs_size 
        self.batch_size = batch_size 
        self.replay_size = replay_size 
        self.horizon = prediction_horizon 
        self.action_dim = action_dim 
        self.trajectory, self.multi_step_struct= [], [] 
        self.ptr, self.size, self.max_size = 0, 0,replay_size
        self.max_struct_size, self.struct_ptr, self.reached_max = 30000, 0 , 0 
        self.ep_len = 0 
        self.fo_in = np.zeros((replay_size,obs_size+action_dim)) 
        self.fo_r = [None for _ in range(self.replay_size)]
        self.y     = np.zeros((replay_size,obs_size)) 

    def step_update(self,o,r,a) : 
        keys = sorted(o.keys()) 
        num_objects = len(keys) 
        obs = np.zeros((num_objects,self.obs_dim))  
        act = np.zeros((num_objects,self.action_dim)) 
        rel = [ None for _ in range(num_objects)]
        for i,elem in enumerate(keys) : 
            obs[i] = o[elem]
            act[i] = a[elem]
            rel[i] = r[elem] 

        self.trajectory.append([obs,act,rel]) 
        self.ep_len +=1 

    def get_obs(self,o,r,a) : 
        keys = sorted(o.keys()) 
        num_objects = len(keys) 
        obs = np.zeros((num_objects,self.obs_dim))  
        for i,elem in enumerate(keys) : 
            obs[i] = o[elem]
        return obs 

    def push_to_buffer(self) : 
        for t in range(len(self.trajectory)-1) : 
            o,a,r,y = self.trajectory[t][0],self.trajectory[t][1],self.trajectory[t][2],self.trajectory[t+1][0] 
            num_objects = o.shape[0] 
            o_comb = np.concatenate((o,a),axis=1)   
            for i in range(num_objects) : 
                self.fo_in[self.ptr] = o_comb[i] 
                rels = r[i] 
                v = np.zeros((len(rels),self.obs_dim+self.action_dim)) 
                for j in range(len(rels)) : 
                    v[j] = o_comb[int(rels[j])] 
                self.fo_r[self.ptr] = v 
                self.y[self.ptr] = y[i] 
                self.ptr = (self.ptr+1)% self.max_size
                self.size = min(self.size+1,self.max_size)

    def sample_batch(self) :
        # Sample random idxs from buffer 
        # fr_u,fr_v are input to the relation model, effects (obtained using relation model) and fo_u are input to the object model, y is the ground truth
        idx = int(np.random.randint(0, self.size, size=self.batch_size))
        return dict(fo_in=self.fo_in[idx],
                    fo_r=self.fo_r[idx],
                    y=self.y[idx])
                
    def multi_step_push(self,t) : 
        state = {} 
        o,a,r = self.trajectory[t][0],self.trajectory[t][1],self.trajectory[t][2]
        o = np.concatenate((o,a),axis=1)
        state['fo_in'] = o 
        state['fo_r'] = r 
        for i in range(self.horizon) :  
            state['y_{}'.format(i+1)] = self.trajectory[t+i+1][0]
            state['a_{}'.format(i)] = self.trajectory[t+i][1]

        if self.reached_max :
                self.multi_step_struct[self.struct_ptr] = state  
                self.struct_ptr = (self.struct_ptr+1) % self.max_struct_size 
        else :
                self.multi_step_struct.append(state)
                self.reached_max = len(self.multi_step_struct) == self.max_struct_size
                if self.reached_max : 
                    print('Reached Maximum for Struct, now samples will be removed from it') 
        
    def sample_multi_step(self) : 
        # To test multi-step prediction for all vehicles at a given timestep, a single element from the multi_step_struct is sampled (This is not used for training
        # but is used for evaluating the performance of the model on long horizon predictions)
        ind = np.random.randint(0,len(self.multi_step_struct)) 
        return self.multi_step_struct[ind] 
    
    def end_episode(self): 
        # Pushing to buffer just builds up the training buffer 
        self.push_to_buffer()
        for _ in range(500) :
            t= np.random.randint(0,self.ep_len-self.horizon-1)
            try :
                self.multi_step_push(t)
            except :
                print('Error building up multi-step struct, breaking')
                break  
        # Reset quantities which just need to be stored for the duration of a trajectory  
        self.ep_len = 0 
        self.trajectory = [] 


class FigureEightEnv(Env):
    def def_sumo(self):
        c = self.c

        r = c.radius
        ring_length = r * (3 * np.pi / 2)
        nodes = E('nodes',
            E('node', id='center', x=r, y=r), # center around (r, r) instead of (0, 0) so that SUMO creates all internal edges symmetrically
            E('node', id='right', x=2 * r, y=r),
            E('node', id='top', x=r, y=2 * r),
            E('node', id='left', x=0, y=r),
            E('node', id='bottom', x=r, y=0),
        )
        center, right, top, left, bottom = nodes
        builder = NetBuilder()
        builder.chain([left, bottom, center, top, right, center, left], edge_attrs=[
            dict(length=ring_length, shape=' '.join(f'{r * np.cos(i):.5f},{r * np.sin(i):.5f}' for i in np.linspace(np.pi / 2, 2 * np.pi, 40))),
            {}, {},
            dict(length=ring_length, shape=' '.join(f'{r * (2 - np.cos(i)):.5f},{r * (2 + np.sin(i)):.5f}' for i in np.linspace(0, 3 * np.pi / 2, 40))),
            {}, {},
        ])
        _, edges, connections, _ = builder.build()

        assert c.av == 0 or c.n_veh % c.av == 0
        v_params = {**IDM, **LC2013, **dict(accel=1, decel=1.5, minGap=c.get('min_gap', 2))}
        additional = E('additional',
            E('vType', id='human', **v_params),
            E('vType', id='rl', **v_params),
            *build_closed_route(edges, c.n_veh, space=c.initial_space, type_fn=lambda i: 'rl' if c.av != 0 and i % (c.n_veh // c.av) == 0 else 'human', depart_speed=c.get('depart_speed', 0), offset=c.get('offset', 0), init_length=c.get('init_length'))
        )
        return super().def_sumo(nodes, edges, connections, additional)

    @property
    def stats(self):
        return {k: v for k, v in super().stats.items() if 'flow' not in k}

    def step(self, action=[]):
        c = self.c
        ts = self.ts
        max_speed = c.max_speed
        
        if action is None : 
            action = [] 
        ids = np.arange(c.n_veh)
        rl_ids = ids[::c.n_veh // c.av] if c.av else [] # ith vehicle is RL
        if len(action):
            rls = [ts.vehicles[f'{id}'] for id in rl_ids]
            if not isinstance(action, int):
                action = (action - c.low) / (1 - c.low)
            for a, rl in zip(action, rls):
                if c.act_type.startswith('accel'):
                    level = a / (c.n_actions - 1) if c.act_type == 'accel_discrete' else a
                    if c.get('handcraft'):
                        level = (0.75 * np.sign(c.handcraft - rl.speed) + 1) / 2
                        n_followers = c.n_veh // c.av - 1
                        veh, dist = rl, 0
                        for _ in range(n_followers):
                            veh, f_dist = veh.follower()
                            dist += f_dist
                        if dist > c.get('follower_gap', 17) * n_followers:
                            level = (0.75 * np.sign(0.5 - rl.speed) + 1) / 2
                    ts.accel(rl, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))
                else:
                    if c.act_type == 'continuous':
                        level = a
                    elif c.act_type == 'discretize':
                        level = min(int(a * c.n_actions), c.n_actions - 1) / (c.n_actions - 1)
                    ts.set_max_speed(rl, max_speed * level)

        super().step()

        if ts.new_arrived | ts.new_collided: # Collision
            print(f'Collision between vehicles {[v.id for v in ts.new_arrived | ts.new_collided]} on step {self._step}')
            return dict(reward=np.full(c.av, -10), done=True)
        elif len(ts.vehicles) not in [0, c.n_veh]:
            print(f'Bad number of initial vehicles {len(ts.vehicles)}, probably due to collision')
            return dict(reward=0, done=True)


        obs_dict = {} 
        relations_dict = {} 
        actions_dict = {}

        rls = [ts.vehicles[f'{id}'] for id in rl_ids]
        vehs = [ts.vehicles[f'{id}'] for id in ids]

        route = nexti(ts.routes)
        max_dist = max(x.route_position[route] + x.length for x in route.edges)

        state = np.array([(v.edge.route_position[route] + v.laneposition, v.speed) for v in vehs])
        mina,minb,minc = 10000,10000,10000 
        id_a,id_b,id_c = None,None,None 
    
        for veh in ts.types.human.vehicles:
            if 'right' in str(veh.edge) :
                dis = veh.edge.route_position[route] + veh.laneposition 
                if dis< mina : 
                    mina = dis 
                    id_a = veh.id 
            elif 'left' in str(veh.edge) :
                dis = veh.edge.route_position[route] + veh.laneposition 
                if dis< minb : 
                    minb = dis 
                    id_b = veh.id 
            else :
                dis = veh.edge.route_position[route] + veh.laneposition 
                if dis< minc : 
                    minc = dis 
                    id_c = veh.id 

        edge_ids = [] 
        if id_a is not None : 
            edge_ids.append(int(id_a)) 
        if id_b is not None : 
            edge_ids.append(int(id_b)) 
        if id_c is not None :
            edge_ids.append(int(id_c)) 

        for veh in ts.types.human.vehicles:
            leader, dist = veh.leader() 
            dis = veh.edge.route_position[route] + veh.laneposition
            obs = [dis/500, veh.speed / max_speed, leader.speed / max_speed,0]
            obs_dict[veh.id] = obs 
            relations_dict[veh.id] = [int(leader.id)]
            for i in edge_ids : 
                if i!=int(veh.id) : 
                    relations_dict[veh.id].append(i) 
            actions_dict[veh.id]  = 0

        if action == []  : 
            a = -1 
        else :
            a= action[0]

        for veh in ts.types.rl.vehicles : 
            leader, dist = veh.leader() 
            dis = veh.edge.route_position[route] + veh.laneposition
            obs = [dis/500, veh.speed / max_speed, leader.speed / max_speed,0]
            obs_dict[veh.id] = obs 
            relations_dict[veh.id] = [int(leader.id)]
            for i in edge_ids : 
                if i!=int(veh.id) : 
                    relations_dict[veh.id].append(i)  
            actions_dict[veh.id]  = a 

        self.c.buffer.step_update(obs_dict,relations_dict,actions_dict)

        obs = np.array([np.roll(state, -i, axis=0) for i in rl_ids]).reshape(c.av, c.n_veh, 2) / [max_dist, max_speed] # (c.av, c.n_veh, 2)
        obs[:, :, 0] = obs[:, :, 0] - 0.5 * (obs[:, 0, 0] >= 0.5).reshape(c.av, 1) # Subtract the position by 0.5 as needed for symmetry
        obs = obs.reshape(c.av, c._n_obs)
        assert c.av == 0 or 0 <= np.abs(obs).max() < 1, f'Observation out of range: {obs}'
        rl_speeds = np.array([v.speed for v in rls])
        reward = np.mean([v.speed for v in vehs] if c.global_reward else rl_speeds)

        if c.accel_penalty and hasattr(self, 'last_speeds'):
            reward -= c.accel_penalty * np.abs(rl_speeds - self.last_speeds) / c.sim_step
        self.last_speeds = rl_speeds

        return dict(obs=obs.astype(np.float32), id=rl_ids, reward=np.full(c.av, reward))

class FigureEight(Modified):
    def create_env(c):
        return NormEnv(c, FigureEightEnv(c))

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(c._n_obs,), dtype=np.float32)

    def on_train_start(c):
        super().on_train_start()
        if c.get('last_unbiased'):
            c._model.p_head[-1].bias.data[c.lc_av:] = 0
        c.buffer = new_buffer(prediction_horizon=c.prediction_horizon)

    @property
    def action_space(c):
        assert c.act_type in ['discretize', 'continuous', 'accel', 'accel_discrete']
        if c.act_type == 'accel_discrete':
            return Discrete(c.n_actions)
        return Box(low=c.low, high=1, shape=(1,), dtype=np.float32) # Need to have a nonzero number of actions

if __name__ == '__main__':
    c = FigureEight.from_args(globals(), locals()).setdefaults(
        horizon=3000,
        warmup_steps=100,
        sim_step=0.1,
        n_veh=14,
        av=1,
        max_speed=30,
        max_accel=0.5,
        max_decel=0.5,
        radius=30,
        speed_mode=SPEED_MODE.obey_safe_speed,
        initial_space='random_free',

        act_type='accel_discrete',
        low=-1,
        n_actions=3,
        global_reward=True,
        accel_penalty=0,

        n_steps=100,
        gamma=0.99,
        alg=PG,
        norm_reward=True,
        center_reward=True,
        adv_norm=False,
        batch_concat=True,
        step_save=5,

        render=False,
    )
    c._n_obs = 2 * c.n_veh
    c.run()
