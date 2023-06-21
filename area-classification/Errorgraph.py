import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd

error_data=pd.read_csv("C:/Users/taebe/source/repos/AI_assignment/error.csv")
correct_data=pd.read_csv("C:/Users/taebe/source/repos/AI_assignment/correct.csv")

idx=error_data['sum_idx']
sum_sum_error=error_data['sum_sum_error']
correct=correct_data['correct']
plt.subplot(211)
error_graph=plt.plot(idx,sum_sum_error,'o-',color='red')
plt.subplot(212)
correct_grpah=plt.plot(idx,correct,'o-',color='blue')

plt.show()