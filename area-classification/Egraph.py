import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd

error_data=pd.read_csv("C:/Users/taebe/source/repos/AI_assignment/error_graph.csv")

idx=error_data['idx']
error=error_data['error']
error_graph=plt.plot(idx,error,color='red')
plt.show()