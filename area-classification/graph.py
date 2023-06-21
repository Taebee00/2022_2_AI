import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd

input_data=pd.read_csv("C:/Users/taebe/source/repos/AI_assignment/input.csv")
target_data=pd.read_csv("C:/Users/taebe/source/repos/AI_assignment/target.csv")

plt.figure()
ax1 = plt.axes(projection='3d')
ax1.set_xlim(-7, 7)
ax1.set_ylim(-7, 7)  
ax1.set_zlim(-7, 7)

input_x=input_data['input_x']
input_y=input_data['input_y']
input_z=input_data['input_z']

for i in range(300):
    if (i%3==0):   
        ax1.scatter(input_x[i],input_y[i],input_z[i],s=5,c="red")
    elif (i%3==1):   
        ax1.scatter(input_x[i],input_y[i],input_z[i],s=5,c="green")
    else:
        ax1.scatter(input_x[i],input_y[i],input_z[i],s=5,c="blue")
plt.show()