import numpy as np


class Collector():
    def __init__(self, init_coor=[1, 1]):
        self.pos = init_coor.copy()  # position
        self.save_start_pos = init_coor.copy()

    def move(self, action):
        # clockwise: [up,right,down,left]
        # 0=up, 1=right, 2=down, 3=left, 4=stay

        # up
        if action == 0:
            self.pos[0] -= 1
        # right
        elif action == 1:
            self.pos[1] += 1
        # down
        elif action == 2:
            self.pos[0] += 1
        # left
        elif action == 3:
            self.pos[1] -= 1
        # stay
        elif action == 4:
            pass

    def provisional_move(self, action):
        # clockwise: [up,right,down,left]
        # 0=up, 1=right, 2=down, 3=left, 4=stay
        provisional_pos = self.pos.copy()
        # up
        if action == 0:
            provisional_pos[0] -= 1
        # right
        elif action == 1:
            provisional_pos[1] += 1
        # down
        elif action == 2:
            provisional_pos[0] += 1
        # left
        elif action == 3:
            provisional_pos[1] -= 1
        # stay
        elif action == 4:
            pass
        return provisional_pos

    def reset(self):
        self.pos = self.save_start_pos.copy()


class Grid():
    AGENT_COLOR = np.array([255, 0, 0], dtype=np.uint8)
    SPACE_COLOR = np.array([0, 0, 0], dtype=np.uint8)  # background color
    FOOD_COLOR = np.array([1, 255, 0], dtype=np.uint8)
    WALL_COLOR = np.array([100, 100, 100], dtype=np.uint8)
    HOME_COLOR = np.array([3, 0, 255], dtype=np.uint8)

    # OBJECT_COLOR = np.array([3,0,0], dtype=np.uint8)

    def __init__(self, grid_size=[20, 20]):

        # get dimensions of grid
        self.grid_size = np.asarray(grid_size, dtype=np.int32)
        self.height = self.grid_size[0].copy()
        self.width = self.grid_size[1].copy()

        # Create grid and fill with background color
        self.grid = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.grid[:, :, :] = self.SPACE_COLOR

        # Create walls on the edges
        self.grid[:, 0, :] = self.WALL_COLOR
        self.grid[:, -1, :] = self.WALL_COLOR
        self.grid[0, :, :] = self.WALL_COLOR
        self.grid[-1, :, :] = self.WALL_COLOR

        # Create home
        home_x_middle = self.width // 2
        self.home_coors = [self.height - 3, self.height - 1, home_x_middle - 1, home_x_middle + 2]  # For drawing
        self.home_goal_coor = [self.height - 2, home_x_middle]  # For defining goal (x,y)
        self.grid[self.home_coors[0]:self.home_coors[1], self.home_coors[2]:self.home_coors[3], :] = self.HOME_COLOR

        # variable for storing food position
        self.food = list()

        # for reset
        self.save_grid = self.grid.copy()

    def reset(self):
        self.grid = self.save_grid.copy()
        self.place_food_random()

    def get_color(self, coor):
        # coor is as (x,y) but color is (r,g,b)
        return self.grid[coor[0], coor[1], :]

    def check_food_eaten(self, coor):
        # coor needs to be where agent WANTS to go
        # i.e. agent moved so coors of agent location are updated, but grid not yet
        square_color = self.get_color(coor)
        return np.array_equal(square_color, self.FOOD_COLOR)

    def check_home_reached(self, coor):
        square_color = self.get_color(coor)
        return np.array_equal(square_color, self.HOME_COLOR)

    def place_food_random(self, num_foods=1):
        for _ in range(num_foods):
            while True:
                new_x = np.random.randint(1, self.height - 2)
                new_y = np.random.randint(1, self.width - 1)

                if np.array_equal(self.grid[new_x, new_y, :], self.SPACE_COLOR):
                    # place food
                    self.grid[new_x, new_y, :] = self.FOOD_COLOR
                    self.food = [new_x, new_y]
                    break

    def erase_cell(self, coor):
        self.grid[coor[0], coor[1], :] = self.SPACE_COLOR

    def draw_agent(self, coor):
        self.grid[coor[0], coor[1], :] = self.AGENT_COLOR

    def draw_home(self):
        self.grid[self.home_coors[0]:self.home_coors[1], self.home_coors[2]:self.home_coors[3], :] = self.HOME_COLOR

    def check_legal_space(self, coor):
        square_color = self.get_color(coor)
        return not np.array_equal(square_color, self.WALL_COLOR)

    # Unused in the final version
    # def get_state(self):
    #     return self.grid.copy()

    def get_state_discrete(self, agent_pos_list):
        '''
        agent_pos_list -> [[x1,y1], [x2,y2], ...]
        '''

        # initialise state
        # end goal is [[all state vars for agent1], [all state vars for agent2], ...]
        state = [[] for _ in agent_pos_list]

        # DISTANCE TO FOOD
        for i in range(len(agent_pos_list)):
            agent_pos = agent_pos_list[i]
            x_dist_to_food = agent_pos[0] - self.food[0]
            y_dist_to_food = agent_pos[1] - self.food[1]
            state[i].extend([x_dist_to_food, y_dist_to_food])

        # DISTANCE TO HOME
        for i in range(len(agent_pos_list)):
            agent_pos = agent_pos_list[i]
            x_dist_to_home = agent_pos[0] - self.home_goal_coor[0]
            y_dist_to_home = agent_pos[1] - self.home_goal_coor[1]
            state[i].extend([x_dist_to_home, y_dist_to_home])

        # DISTANCE TO OTHER AGENTS
        # go through all agent combinations
        for i in range(len(agent_pos_list)):
            agent_pos = agent_pos_list[i]  # current agent

            for j in range(len(agent_pos_list)):
                if i == j: continue

                agent_2_pos = agent_pos_list[j]  # all other agents
                x_dist = agent_pos[0] - agent_2_pos[0]
                y_dist = agent_pos[1] - agent_2_pos[1]

                state[i].extend([x_dist, y_dist])

        return state

# main env class
class Env_control():
    
    def __init__(self, grid_size=[11,11], n_agents=2):
        self.n_agents = n_agents
        
        # Initialise grid
        self.grid = Grid(grid_size)
        
        # Create a static list for initial position of agents
        self.init_coors_agents = [[9,3], [9,7], [9,5], [9,1], [9,9]]
        if self.n_agents > len(self.init_coors_agents):
            raise ValueError('n_agents too large, max agents = 5')
        
        # Create agents
        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(Collector(init_coor=self.init_coors_agents[i]))

        self.done = False
        self.state = None
        self.food_eaten = 0  #0=not eaten, 1=eaten
    
    def step(self, action):

        rewards = [-0.1 for _ in range(self.n_agents)]
        info = [{} for _ in range(self.n_agents)]
        
        # Erase old positions
        for agent in self.agents:
            self.grid.erase_cell(agent.pos)
        
        # Move agents if actions are legal (don't move, just provisional move in case they bump into each other)
        provisional_positions = [ [] for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            if self.check_legal_move(self.agents[i].pos, action[i]):
                provisional_positions[i].extend(self.agents[i].provisional_move(action[i]))
            else:
                provisional_positions[i].extend(self.agents[i].pos)
        
        
        # ENVIRONMENT DYNAMICS
        
        # Check if agents bump into each other
        if len(set([tuple(x) for x in provisional_positions])) < len(provisional_positions):
            rewards = [x-5 for x in rewards]
            # agents stay in old pos
        else:
            #move agents
            for i in range(self.n_agents):
                if self.check_legal_move(self.agents[i].pos, action[i]):
                    self.agents[i].move(action[i])
                else:
                    rewards[i] -= 1
        
        # Check if food eaten
        for i in range(self.n_agents):
            if self.grid.check_food_eaten(self.agents[i].pos):
                self.food_eaten=True
                rewards[i] += 10
                
                # check if other agent is nearby:
                for j in range(self.n_agents):
                    if i==j: continue
                    if self.check_near_agents(self.agents[i].pos, self.agents[j].pos):
                        rewards[i] += 5
                        rewards[j] += 5
        
        
        # Check if home reached
        reached_home = 0
        for agent in self.agents:
            if self.grid.check_home_reached(agent.pos) and self.food_eaten:
                reached_home+=1
        if reached_home == self.n_agents:
            rewards = [x+20 for x in rewards]
            self.done = True
        
        # Draw new agent
        self.grid.draw_home()
        for agent in self.agents:
            self.grid.draw_agent(agent.pos)
        
        
        self.state = self.get_state()
        done = [self.done for _ in range(self.n_agents)]
        
        return self.state, rewards, done, info
    
    
    def get_state(self):
        state = self.grid.get_state_discrete([agent.pos for agent in self.agents])
        for s in state:
            s.append(self.food_eaten)
        state = [tuple(s) for s in state]
        return state
    
    def reset(self):
        self.done = False
        self.grid.reset()
        for agent in self.agents:
            agent.reset()
            self.grid.draw_agent(agent.pos)
        return self.get_state()
    
    
    def check_legal_move(self, agent_coor, action):
        # if not .copy() it will change the original value
        coor = agent_coor.copy()
        # fake move
        if   action == 0: coor[0] -= 1 #up
        elif action == 1: coor[1] += 1 #right
        elif action == 2: coor[0] += 1 #down
        elif action == 3: coor[1] -= 1 #left
        elif action == 4: pass         #stay
        
        return self.grid.check_legal_space(coor)
        
    def check_near_agents(self, agent1_pos, agent2_pos):
        if np.abs(agent1_pos[0] - agent2_pos[0]) <= 1:
            if np.abs(agent1_pos[1] - agent2_pos[1]) <= 1:
                return True
        else: return False
        