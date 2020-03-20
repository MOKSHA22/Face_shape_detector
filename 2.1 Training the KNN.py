import numpy as np

data=np.load('data.npy')
target=np.load('target.npy')
    # load data and target in numpy files

print(data)
print(target)


from sklearn.neighbors import KNeighborsClassifier

algorithm=KNeighborsClassifier()

algorithm.fit(data,target)
    # .fit - trains the algorithm


# we can save the pre-trained algorithm using joblib library

import joblib

joblib.dump(algorithm,'KNN_model.sav')

