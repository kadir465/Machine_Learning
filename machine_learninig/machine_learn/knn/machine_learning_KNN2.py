import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

class learn1():
    def __init__(self):
        x=np.sort(5*np.random.rand(40,1), axis=0)
        y=np.sin(x).ravel()

        y[::5]+=1*(0.5-np.random.rand(8))  

        T=np.linspace(0,5,500)[:,np.newaxis]

        for i, weight in enumerate(["uniform","distance"]):
            knn=KNeighborsRegressor(n_neighbors=5,weights=weight)
            y_pred=knn.fit(x,y).predict(T)

            plt.subplot(2, 1, i+1)
            plt.scatter(x,y, color="green",label="data")
            plt.scatter(T,y_pred,color="blue",label="prediction")
            plt.axis("tight")
            plt.legend()
            plt.title("KNN Regressor Weight = {}".format(weight))
            plt.savefig("learning1.png")
        plt.show()    
      
learn1()

