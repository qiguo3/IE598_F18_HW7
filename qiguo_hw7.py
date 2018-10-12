import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import	matplotlib.pyplot	as	plt 

#import X,y
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                       header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

#Part 1: Random forest estimators

#Set seed	for	reproducibility 
seed=1
#Split	dataset into 90% train and 10% test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,stratify=y,random_state=seed)

list_n=[]
list_insample_accuracyscore=[]
list_outofsample_accuracyscore=[]
for n in [25,50,100,150,200,250,300]:
    list_n.append(n)
    rf=RandomForestClassifier(n_estimators=n,
                             random_state=seed)
    insample_accuracy=cross_val_score(rf,X_train,y_train,cv=10,
                                  scoring='accuracy',
                                  n_jobs=1) 
    cv_error=np.mean(insample_accuracy)
    list_insample_accuracyscore.append(cv_error)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    outofsample_accuracy=accuracy_score(y_test,y_pred)
    list_outofsample_accuracyscore.append(outofsample_accuracy)
    
accuracyscore_rf=np.vstack((list_n,list_insample_accuracyscore,list_outofsample_accuracyscore))
df_acccuracyscore_rf=DataFrame(accuracyscore_rf,columns=['','','','','','',''])
df_acccuracyscore_rf.index=Series(['N_estimators','in-sample accuracies','out-of-sample accuracie'])
df_acccuracyscore_rf.to_excel('accuracyscore_rf.xls')
print(df_acccuracyscore_rf)

#Part 2: Random forest feature importance.
feat_labels = df_wine.columns[1:]
rf=RandomForestClassifier(n_estimators=50,
                          random_state=seed)
rf.fit(X_train,y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is QI GUO")
print("My NetID is: qiguo3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################










