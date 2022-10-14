import numpy as np

#true labels array
a = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0])
#top_k predictions array
b = np.array([[0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1]])

print("top1 missed predictions")
for i in range(len(a)):
    if a[i] != b[0][i]:
        print("index " + str(i) + " label: " + str(a[i]) + " pred: " + str(b[0][i]))

print("top"+str(len(b))+" missed predictions")
for i in range(len(a)):
    eq = 0
    for j in range(len(b)):
        if a[i] != b[j][i]:
            continue
        else:
            eq = 1
    if eq == 0:
        print("index " + str(i) + " label: " + str(a[i]) + " pred: " + str(b[:,i]))