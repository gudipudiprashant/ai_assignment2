import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Needs to hold 3 values:
#   low - 0.05
#   med - 1
#   high- 50
sigma_list = [0.05, 1, 50]


init_x = -1
NUM_SAMPLES = 1500

def target_distr(x):

    # returns probability density of x
    p1 = math.exp(-1*(math.pow(x, 4))) 
    p2 = (2 + math.sin(5*x) + math.sin(-2*(math.pow(x,2))))

    return p1*p2

def proposal_sample(mu, sigma):

    # returns sample from proposal distribution
    return np.random.normal(mu, sigma, 1)[0]


def metropolis_hastings_algorithm(sigma):

    x_prev = init_x
    x_cand = None
    acceptance_prob = None
    u = None

    sample_list = []
    sample_list.append(x_prev)

    for t in range(1, NUM_SAMPLES+1):

        x_cand = proposal_sample(x_prev, sigma)

        # The proposal distribution is symmetric. So we can ignore first part
        # of the expression.
        acceptance_prob = min(1, target_distr(x_cand)/target_distr(x_prev))

        u = random.uniform(0,1)

        if(u < acceptance_prob):
            x_prev = x_cand
        else:
            x_prev = x_prev

        sample_list.append(x_prev)


    return sample_list


if __name__ == '__main__':
    sample_list = []
    for sigma in sigma_list:
        cur_sample = metropolis_hastings_algorithm(sigma)
        sample_list.append(cur_sample)

        plt.figure()
        plt.hist(cur_sample,bins=20)
        plt.title("sigma="+str(sigma))
        plt.xlim(-4,4)
        # plt.ylim(-1.5,1.5)
        plt.xlabel('Sample')
        plt.ylabel('Frequency')

        plt.figure()
        plt.plot(cur_sample, 'ro')
        plt.title("sigma="+str(sigma))
        plt.ylim(-1.5,1.5)
        plt.xlabel('Iteration Number')
        plt.ylabel('Sample')


    # plt.figure()
    # plt.hist(sample_list[0])
    # plt.figure()
    # plt.hist(sample_list[1])

    plt.show()