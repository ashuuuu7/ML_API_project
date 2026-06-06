import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib 
data = pd.read_csv("c:/Users/Ashutosh Giri/OneDrive/Pictures/Documents/My AI&ML/My ML Project/Project Data/placementdata.csv")
data.drop("StudentID", axis= 1 ,inplace= True)
# print(data)
# print(data.info())
# print(data.shape)
# print(data.isnull().sum())
# print(data.duplicated().sum())
# print(data["PlacementStatus"].value_counts())
data["PlacementStatus"] = data["PlacementStatus"].map({"Placed" : 1, "NotPlaced" : 0})
# print(data.groupby("CGPA")["PlacementStatus"].mean())
# print(data.groupby("Internships")["PlacementStatus"].mean())
# print(data.groupby("Projects")["PlacementStatus"].mean())
# print(data.groupby("AptitudeTestScore")["PlacementStatus"].mean())
# print(data.groupby("SoftSkillsRating")["PlacementStatus"].mean())
# print(data.groupby("Workshops/Certifications")["PlacementStatus"].mean())
# print(data.groupby("ExtracurricularActivities")["PlacementStatus"].mean())
# print(data.groupby("PlacementTraining")["PlacementStatus"].mean())
data["PlacementTraining"] = data["PlacementTraining"].map({"Yes" : 1, "No" : 0})
data["ExtracurricularActivities"] = data["ExtracurricularActivities"].map({"Yes" : 1, "No" : 0})
# sns.heatmap(data.corr())
# plt.show()
x = data.iloc[:, : -1]
y = data["PlacementStatus"]
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2 ,random_state= 42)
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
lr = LogisticRegression(class_weight= "balanced", C= 0.2)
lr.fit(x_train, y_train)
# print(lr.score(x_test, y_test), lr.score(x_train, y_train))
rfr = RandomForestClassifier(n_estimators=100)
rfr.fit(x_train, y_train)
# print(rfr.score(x_test, y_test), rfr.score(x_train, y_train))
knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
# print(knc.score(x_test, y_test), knc.score(x_train, y_train))
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
# print(dtc.score(x_test, y_test), dtc.score(x_train, y_train))
svc = SVC(C= 0.2)
svc.fit(x_train, y_train)
# print(svc.score(x_test, y_test), svc.score(x_train, y_train))
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
# print(gbc.score(x_test, y_test), gbc.score(x_train, y_train))
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
# print(xgb.score(x_test, y_test), xgb.score(x_train, y_train))
y_pred = lr.predict(x_test)
print(confusion_matrix(y_test, y_pred))
y_pred = svc.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
models = {"Logistic": lr, "RandomForest": rfr, "KNN": knc, "DecisionTree": dtc, "SVM": svc, "GradientBoost": gbc, "XGBoost": xgb}
for name, model in models.items():
    print(name, model.score(x_test, y_test))
y_pred = svc.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
params = [{'kernel': ['linear'], 'C': [0.1, 0.5, 1, 2, 5, 10]},{'kernel': ['rbf'], 'C': [0.1, 0.5, 1, 2, 5, 10], 'gamma': ['scale', 'auto']}]
grid = GridSearchCV(estimator=SVC(), param_grid=params, cv=5, scoring='accuracy')
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
print(data.corr()["PlacementStatus"].sort_values(ascending=False))
fi = pd.Series(rfr.feature_importances_, index= x.columns)
print(fi.sort_values(ascending= False))
joblib.dump(svc, "Ashutosh's_Placement_Prediction")
joblib.dump(ss, "scaler.pkl")