import matplotlib.pyplot as plt
import csv
import pandas as pd

weight_ij_data=pd.read_csv("C:/Users/taebe/source/repos/AI_assignment/wgraph_ij.csv")
weight_jk_data=pd.read_csv("C:/Users/taebe/source/repos/AI_assignment/wgraph_jk.csv")
weight_kl_data=pd.read_csv("C:/Users/taebe/source/repos/AI_assignment/wgraph_kl.csv")

weight_ij=weight_ij_data['wij']
weight_jk=weight_jk_data['wjk']
weight_kl=weight_kl_data['wkl']

plt.figure(figsize=(20, 10))

plt.subplot(1,3,1)
plt.xlabel('TIMES')
plt.ylabel('Weight')
plt.title('1-2 LAYER Weight Graph')
for i in range(15):
    xlist=[]
    ylist=[]
    for j in range(6000):
        xlist.append(j)
        ylist.append((weight_ij[i+15*j]))
    plt.plot(xlist,ylist)
    
plt.subplot(1,3,2)
plt.xlabel('TIMES')
plt.ylabel('Weight')
plt.title('2-3 LAYER Weight Graph')
for i in range(15):
    xlist=[]
    ylist=[]
    for j in range(6000):
        xlist.append(j)
        ylist.append((weight_jk[i+15*j]))
    plt.plot(xlist,ylist)
    
plt.subplot(1,3,3)
plt.xlabel('TIMES')
plt.ylabel('Weight')
plt.title('3-4 LAYER Weight Graph')
for i in range(9):
    xlist=[]
    ylist=[]
    for j in range(6000):
        xlist.append(j)
        ylist.append((weight_kl[i+9*j]))
    plt.plot(xlist,ylist)
    
plt.show()