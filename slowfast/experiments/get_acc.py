import os
import pandas as pd
import numpy as np

df = pd.read_excel("./data_sheet.xlsx")

missed_prediction_indices = [1, 4, 6, 12, 14, 18]
test_split = [9, 24, 39, 41, 44, 48, 57, 58, 65, 69, 70, 71, 72, 74, 79, 84, 85, 86, 87, 88, 89, 90, 97, 117, 119]

binary = False
ood_names = ["George H.W. Bush", "Freddie Roach", "Neil Diamond", "Bob Hoskins", "Freddie Roach"]

num_id_incorrect = 0.0
num_ood_incorrect = 0.0
total_id = 0.0
total_ood = 0.0

for i in test_split:
    if df.person[i] in ood_names:
        total_ood += 1.0
    else:
        total_id += 1.0

print(total_id)
print(total_ood)

for i in missed_prediction_indices:
    if df.person[test_split[i]] in ood_names:
        num_ood_incorrect += 1.0
    else:
        num_id_incorrect += 1.0


print("OOD Acc: " + str((total_ood - num_ood_incorrect) / total_ood))
print("ID Acc: " + str((total_id - num_id_incorrect) / total_id))