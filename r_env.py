# Code for the robot environment
import random

OBSTACLE_SPACES = [ (1, 5), (1, 11), (1, 15),
                    (2, 1), (2, 2), (2, 5), (2, 7), (2, 8), (2, 10), (2, 12),
                      (2, 14), (2, 15), (2, 16),
                    (3, 1), (3, 5), (3, 7), (3, 14), (3, 15),
                    (4, 3), (4, 7), (4, 12)]

class RobotEnv:
  def __init__(self, num_rows=4, num_cols=16, obstacle_sp_list=OBSTACLE_SPACES):
    self.num_rows = num_rows
    self.num_cols = num_cols
    self.obstacle_sp = set(obstacle_sp_list)
    
    self.map_empty_spaces()
    self.num_states = len(self.state_sp_mapper.keys())
    self.adj_list = self.__get_adj_list()

    self.reset()

  def reset(self):
    self.cur_state = self.get_start_state()
    self.time_step = 0
    self.all_states = [self.cur_state]

  def get_next_evidence(self, error_rate):
    self.cur_state = self.get_next_state()
    self.all_states.append(self.cur_state)
    self.time_step += 1
    true_neighbour_dirns = self.adj_list[self.cur_state].keys()
    all_dirns = ["N", "S", "E", "W"]

    act_sensor_reading = list(true_neighbour_dirns)
    for dirn in all_dirns:
      if random.random() < error_rate:
        # error in sensor reading
        if dirn in act_sensor_reading:
          act_sensor_reading.remove(dirn)
        else:
          act_sensor_reading.append(dirn)

    # print("TRUE SENSOR: ", true_neighbour_dirns)
    # print("ACTUAL VALS: ", act_sensor_reading)
    return act_sensor_reading

  def get_next_state(self):
    neighbours = list(self.adj_list[self.cur_state].values())
    rand_index = random.randrange(len(neighbours))
    return neighbours[rand_index]

  def is_valid_empty_space(self, space):
    if (1 <= space[0] <= self.num_rows and
        1 <= space[1] <= self.num_cols and
        space not in self.obstacle_sp):
      return True
    return False

  def get_empty_neighbours(self, space):
    """
    Gets a list of neighbours in NSWE directions in that order
    """
    neighbours = {}
    directions =  {(-1, 0) : "N", (1, 0) : "S", (0, 1) : "E", (0, -1) : "W"}

    for dirn in directions.keys():
      check_space = list(space)
      check_space[0] += dirn[0]; check_space[1] += dirn[1]
      check_space = tuple(check_space)
      if self.is_valid_empty_space(check_space):
        neighbours[directions[dirn]] = self.sp_state_mapper[check_space]

    return neighbours

  def __get_adj_list(self):
    adj_list = []
    for i in range(self.num_states):
      space = self.state_sp_mapper[i]
      adj_list.append(self.get_empty_neighbours(space))

    return adj_list

  def get_adj_list(self):
    return self.adj_list

  def map_empty_spaces(self):
    """
    Maps each empty space in the maze to a number i denoting the state S_i
    and the other way round
    """
    self.state_sp_mapper = {}
    self.sp_state_mapper = {}
    state =  0
    for i in range(1, self.num_rows+1):
      for j in range(1, self.num_cols+1):
        if (i, j) not in self.obstacle_sp:
          self.sp_state_mapper[(i, j)] = state
          self.state_sp_mapper[state] = (i, j)
          state += 1

  def get_start_state(self):
    st_state = 12
    while st_state == 12:
      st_state =  random.randrange(self.num_states)
    return st_state

  def get_num_states(self):
    return self.num_states

  def get_localization_err(self, agent_state):
    actual_space = self.state_sp_mapper[self.cur_state]
    agent_space = self.state_sp_mapper[agent_state]
    # print(actual_space, agent_space)
    ans = 0
    for i in range(2):
      ans += abs(actual_space[i] - agent_space[i])
    return ans

  def get_path_acc(self, agent_path):
    actual_path = self.all_states
    agent_path = agent_path
    # print("ACT: ", actual_path, "agent_path: ", agent_path)
    correct = 0
    for i in range(len(actual_path)):
      if actual_path[i] == agent_path[i]:
        correct += 1
    return correct/len(actual_path)



if __name__ == "__main__":
  re = RobotEnv(4, 16, OBSTACLE_SPACES)
  print(re.adj_list)
  for i in range(re.num_states):
    if len(re.adj_list[i]) == 0:
      print("ERRER: ", i)
