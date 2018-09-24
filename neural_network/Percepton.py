import random
import math

w=[0,0]
b=[0]

def create_perceptron():
    w[0] = random.random()
    w[1] = random.random()
    b[0] = random.random()

def activate(x):
    sum=(w[0]*x[0]) + (w[1]*x[1]) + b[0]
    if sum > 0 :
        return 1
    else:
        return -1

def compute_error(target,predicted):
    error = target - predicted
    return error

def learn(error, learning_rate, x):
    w[0] = w[0] + learning_rate*error*x[0]
    w[1] = w[1] + learning_rate*error*x[1]
    b[0] = b[0] + learning_rate*error*1

def one_step_learn(x_sample, y_sample, learning_rate):
    predicted = activate(x_sample)
    error = compute_error(y_sample, predicted)
    #print error
    learn(error, learning_rate, x_sample)

def train(x_dataset, y_dataset, learning_rate, n_iteration):
    for i in range (n_iteration):
        rn = random.randint(0, len(x_dataset)-1)
        x_sample = x_dataset[rn]
        y_sample = y_dataset[rn]
        one_step_learn(x_sample, y_sample, learning_rate)

# if i %100:
#    print ('w old', w)
#    print ('b old', b)

if __name__ == "__main__":
    x_dataset = [[0,0],
                 [0,1],
                 [1,0],
                 [1,1]]
y_dataset = [-1,1,1,1]
#y_dataset = [-1,-1,-1,1]
print('activation before learning')

for x in x_dataset:
    print(activate(x))

learning_rate = 0.01
create_perceptron()
train(x_dataset, y_dataset, learning_rate, 500)

print('activation after learning')
for x in x_dataset:
    print(activate(x))