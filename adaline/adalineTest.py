import adaline as ad
import decisionRegions as dR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./iris.data',header=None)
y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setasa',-1,1)
X = df.iloc[0:100,[0,2]].values
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()
ada = ad.AdalineGD(n_iter=15,eta=0.01)
ada.fit(X_std,y)
dR.plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length[standardized]')
plt.ylabel('petal length[standardized]')
plt.legend(loc = 'upper left')
plt.show()
plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('sum-squared-error')
plt.show()
