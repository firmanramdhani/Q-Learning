import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


# load data
data = np.loadtxt("102842_DataTugas3ML2019.txt")

start = (0, 0)
goal = (14, 14)
plt.figure(figsize=(20, 5))
plt.axis("off")
table = plt.table(cellText=data, loc="center", fontsize=1)
table.get_celld()[start].set_color("red")
table.get_celld()[goal].set_color("Blue")
plt.show()

episodes = 1000
maximum_move = 50
Q_table = np.zeros((15 * 15, 4))
alpha = 0.9
gamma = 0.8


def aksiSelanjutnya(p):
    if p == (0, 0):
        return random.choice([1, 2])
    elif p == (0, 14):
        return random.choice([3, 2])
    elif p == (14, 14):
        return random.choice([0, 3])
    elif p == (14, 0):
        return random.choice([0, 1])
    elif p[1] == 0:
        return random.choice([0, 2, 1])
    elif p[1] == 14:
        return random.choice([0, 2, 3])
    elif p[0] == 0:
        return random.choice([2, 3, 1])
    elif p[0] == 14:
        return random.choice([0, 3, 1])
    else:
        return random.choice([0, 1, 2, 3])


actions = {0: "Atas", 1: "Kanan", 2: "Bawah", 3: "Kiri"}


def perpindahan(p, act):
    if act == 4:
        return p
    if actions[act] == "Atas":
        return (p[0] - 1, p[1])
    elif actions[act] == "Kanan":
        return (p[0], p[1] + 1)
    elif actions[act] == "Bawah":
        return (p[0] + 1, p[1])
    elif actions[act] == "Kiri":
        return (p[0], p[1] - 1)
    return p


for episode in range(episodes):
    position = start
    for move in range(maximum_move):
        action = aksiSelanjutnya(position)
        if action == 4:
            break
        next_move = perpindahan(position, action)
        #         print(next_move)
        reward = data[next_move]
        id = 15 * next_move[1] + next_move[0]
        Qmax = max(Q_table[id])
        Q_table[id][action] = Q_table[id][action] + alpha * (reward + gamma * Qmax)
        position = next_move


Q_table[14]
best = []
for i, q in enumerate(Q_table):
    action = actions[np.argmax(q)]
    pos = (i // 15, i % 15)
    best.append([pos, action])
    print(pos, action)

plt.figure(figsize=(20, 5))
plt.axis("off")
table = plt.table(cellText=data, loc="center", fontsize=1)
for b in best:
    if b[1] == "Atas":
        table.get_celld()[b[0]].set_color("yellow")
    elif b[1] == "Kanan":
        table.get_celld()[b[0]].set_color("red")
    elif b[1] == "Bawah":
        table.get_celld()[b[0]].set_color("blue")
    elif b[1] == "Kiri":
        table.get_celld()[b[0]].set_color("green")
plt.show()

