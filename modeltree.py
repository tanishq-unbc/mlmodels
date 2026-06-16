from sklearn.tree import DecisionTreeClassifier
import joblib
x=[[19,50000], [25,60000],[17,20000],[15,10000]]
y=[1,1,1,0]
model=DecisionTreeClassifier()
model.fit(x,y)
joblib.dump(model,'decisiontree.pkl')
print("saved")