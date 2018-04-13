# Code for testing and plotting graphs
import matplotlib.pyplot as plt
import numpy as np

from r_agent import RobotAgent
from r_env import RobotEnv

NUM_RUNS = 300
ERROR_RATES = [0, 0.02, 0.05, 0.1, 0.2]
NUM_OBS = 40

graph_y = []
loc_y = []

env = RobotEnv()
for err_rate in ERROR_RATES:
  print("RUnning for error rate: ", err_rate)
  path_acc = np.array([0. for i in range(NUM_OBS)])
  loc_err = np.array([0. for i in range(NUM_OBS)])

  agent = RobotAgent(env, err_rate, NUM_OBS)

  for i in range(NUM_RUNS):
    env.reset()
    print("RUN: ", i)
    agent.reset()
    agent.run()
    path_acc += np.array(agent.path_acc)
    loc_err += np.array(agent.localization_err)

  path_acc /= NUM_RUNS
  loc_err /= NUM_RUNS

  graph_y.append(path_acc)
  loc_y.append(loc_err)

plt.figure()
ctr=0
for color in ['r-','b-','g-','c-', 'm-']:
  plt.plot(graph_y[ctr],color,label=str(ERROR_RATES[ctr]))
  ctr+=1
plt.xlabel('Number of Observations')
plt.ylabel('Path accuracy')
plt.legend()
plt.show()


plt.figure()
ctr=0
for color in ['r-','b-','g-','c-', 'm-']:
  plt.plot(loc_y[ctr],color,label=str(ERROR_RATES[ctr]))
  ctr+=1
plt.xlabel('Number of Observations')
plt.ylabel('Localization error')
plt.legend()
plt.show()

