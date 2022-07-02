import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv('train.csv')
print(data)
#
# print(data.info)



dtc = DecisionTreeClassifier()
svm = svm.SVC()
gbc = GradientBoostingClassifier(n_estimators=10)
logr = LogisticRegression(random_state=0)
rfc = RandomForestClassifier(random_state=1)

wine_grade = []
for i in data['quality']:
    if i >7:
        wine_grade.append(1)
    else:
        wine_grade.append(0)

data['quality'] = wine_grade

x = data.drop(["type", "citric acid", "residual sugar", "density", "pH", "volatile acidity", "quality"], axis=1)
y = data['quality']
#

x.fillna(x.mean(), inplace=True)
#
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)
#
dtc = DecisionTreeClassifier(random_state=0)
logr.fit(x_train, y_train)
rfc.fit(x_train, y_train)
dtc.fit(x_train, y_train)

gbc.fit(x_train, y_train)
nb.fit(x_train, y_train)

ylogr_predict = logr.predict(x_test)
rfcy_predict = rfc.predict(x_test)
dtcy_predict = dtc.predict(x_test)
svmy_predict = svm.predict(x_test)

gbcy_predict = gbc.predict(x_test)
nby_predict = nb.predict(x_test)

print('Logistic:', accuracy_score(y_test, ylogr_predict))
print('Random Forest:', accuracy_score(y_test, rfcy_predict))
print('Decision Tree:', accuracy_score(y_test, dtcy_predict))
print('Support Vector:', accuracy_score(y_test, svmy_predict))

print('Gradient Boosting:', accuracy_score(y_test,  gbcy_predict))
print('Naive Bayes:', accuracy_score(y_test,  nby_predict))

