import numpy as np

def get_pow(a, b):
  if (b == 0):
    return 1
  return a**b

class RobotAgent:
  def __init__(self, rob_env, error_rate, num_observations):
    self.robot_env = rob_env

    self.num_states = rob_env.get_num_states()
    self.adj_list = rob_env.get_adj_list()

    self.trans_matrix = self.get_transition_matrix()

    self.error_rate = error_rate
    self.num_obs = num_observations

    self.reset()
    

  def reset(self):
    self.time_slice = 0
    # For forward algorithm
    # f_1_t - stores the distribution of the states at time slice t
    # Initially all states have equal probability
    self.f_1_t = np.array([1/self.num_states for i in range(self.num_states)])
    # stores the max probability to end at state i given e_1_t for each t
    self.viterbi_mat = [[(1/self.num_states, -1) for i in range(self.num_states)]]
    # Store the statistics
    self.localization_err = []
    self.path_acc = []

  def get_Viterbi_path(self, evidence_new, O_new):
    m_1_t = []
    t = self.time_slice
    tot_sum = 0

    for state in range(self.num_states):
      opt_prob, parent_state = max([
                        (self.trans_matrix[par_state][state] *
                            self.viterbi_mat[t-1][par_state][0],
                        par_state) for par_state in range(self.num_states)])
      
      opt_val = (opt_prob * O_new[state][state], parent_state)
      m_1_t.append(opt_val)

      tot_sum += opt_val[0]

    alpha = 1/tot_sum
    for state in range(self.num_states):
      m_1_t[state] = (m_1_t[state][0]*alpha, m_1_t[state][1])

    self.viterbi_mat.append(m_1_t)

    # Creating viterbi path
    path = []
    best_prob = -1
    best_last_state = -1
    for state in range(self.num_states):
      if m_1_t[state][0] > best_prob:
        best_last_state = state
        best_prob = m_1_t[state][0]

    path.append(best_last_state)
    cur_state = self.viterbi_mat[t][best_last_state][1]
    for time_slice in range(t-1, -1, -1):
      path.append(cur_state)
      cur_state = self.viterbi_mat[time_slice][cur_state][1]

    return list(reversed(path))

  def get_discrepancy(self, state, evidence):
    evidence = set(evidence)
    true_neighbour_dirns = set(self.adj_list[state].keys()) 
    return (len(evidence - true_neighbour_dirns) +
            len(true_neighbour_dirns - evidence))

  def get_O(self, evidence_new):
    O_new = []
    err_rate = self.error_rate
    for state in range(self.num_states):
      O_new.append([0 for i in range(self.num_states)])
      d_it = self.get_discrepancy(state, evidence_new)
      O_new[state][state] = (get_pow(1 - err_rate, 4 - d_it) * 
        get_pow(err_rate, d_it))

    return np.array(O_new)

  def forward_algo(self, O_new):
    f_new = np.dot(O_new, np.dot(self.trans_matrix.transpose(), self.f_1_t))
    alpha = 1/np.sum(f_new)
    return f_new * alpha

  def run(self):
    for time_slice in range(1, self.num_obs + 1):
      self.time_slice = time_slice
      # Get E_t+1
      evidence_new = self.robot_env.get_next_evidence(self.error_rate)
      # Get the sensor model values
      O_new = self.get_O(evidence_new)
      # Calculate forward algorithm
      self.f_1_t = self.forward_algo(O_new)

      # Get most likely state
      most_likely_state = np.argmax(self.f_1_t)
      # Calculate the viterbi path
      viterbi_path = self.get_Viterbi_path(evidence_new, O_new)

      self.localization_err.append(
        self.robot_env.get_localization_err(most_likely_state))
      self.path_acc.append(self.robot_env.get_path_acc(viterbi_path))

  def get_transition_matrix(self):
    trans_matrix = []
    for state in range(self.num_states):
      row = [0 for i in range(self.num_states)]
      neighbours = list(self.adj_list[state].values())
      for neighbour in neighbours:
        row[neighbour] = 1/len(neighbours)
      trans_matrix.append(row)

    # print("TRANS MATRIX:\n ", trans_matrix)
    return np.array(trans_matrix)


if __name__ == "__main__":
  import time
  import r_env
  re = r_env.RobotEnv(4, 16, r_env.OBSTACLE_SPACES)
  t = time.time()
  ra = RobotAgent(re, 0, 40)
  ra.run()
  plt.plot(ra.path_acc)
  plt.show()
  print("TIME: ", time.time()-t)