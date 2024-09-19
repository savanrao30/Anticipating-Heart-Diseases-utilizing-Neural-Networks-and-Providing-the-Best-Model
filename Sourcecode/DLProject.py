import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

heart_project = pd.read_csv("/content/sample_data/heart-disease-data.csv")
heart_project = heart_project[heart_project['BMI'].notnull()]

# X = heart_project.drop('CHD', axis=1)
X = heart_project.iloc[:,:-1]
y = heart_project.iloc[: , -1]
X_train, X_test, targettraindata, targettestdata  = train_test_split(X,y, test_size = 0.2, random_state = 1)
sc_X = StandardScaler()
featurestraindata=sc_X.fit_transform(X_train)
featurestestdata=sc_X.transform(X_test)
#featurestraindata, featurestestdata, targettraindata, targettestdata = train_test_split(X,y, test_size = 0.2, random_state = 1)

heart_project.info()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

results=[]

knnModel = KNeighborsClassifier(n_neighbors = 3, p=2)
knnModel.fit(featurestraindata,targettraindata)
targetpredicted = knnModel.predict(featurestestdata)
knnResult=knnModel.score(featurestestdata,targettestdata)
results.append(('KNN Model Accuracy', knnResult))
print("Accuracy of knnModel:",knnResult)
print(classification_report(targettestdata, targetpredicted))

from sklearn.svm import SVC

linearSVMModel = SVC(kernel='linear')
linearSVMModel.fit(featurestraindata,targettraindata)
targetpredicted = linearSVMModel.predict(featurestestdata)
linearSVMModeResult=linearSVMModel.score(featurestestdata,targettestdata)
results.append(('Linear SVM Model Accuracy', linearSVMModeResult))
print("Accuracy of linearSVMModel:",linearSVMModeResult)
print(classification_report(targettestdata, targetpredicted))


gaussianKernalSVMModel = SVC(kernel='rbf')
gaussianKernalSVMModel.fit(featurestraindata,targettraindata)
targetpredicted = gaussianKernalSVMModel.predict(featurestestdata)
gaussianKernalSVMResult=gaussianKernalSVMModel.score(featurestestdata,targettestdata)
results.append(('Gaussian Kernal SVM Model Accuracy',gaussianKernalSVMResult ))
print("Accuracy of gaussianKernalSVMModel:",gaussianKernalSVMResult)
print(classification_report(targettestdata, targetpredicted))


from sklearn.naive_bayes import GaussianNB

NaiveBayesModel = GaussianNB()
NaiveBayesModel.fit(featurestraindata,targettraindata)
targetpredicted = NaiveBayesModel.predict(featurestestdata)
NaiveBayesResult= NaiveBayesModel.score(featurestestdata,targettestdata)
results.append(('NaiveBayes Model Accuracy',NaiveBayesResult ))
print("Accuracy of NaiveBayesModel:", NaiveBayesResult)
print(classification_report(targettestdata, targetpredicted))


from sklearn.tree import DecisionTreeClassifier

decisionTreeModel = DecisionTreeClassifier()
decisionTreeModel.fit(featurestraindata,targettraindata)
targetpredicted = decisionTreeModel.predict(featurestestdata)
decisionTreeResult= decisionTreeModel.score(featurestestdata,targettestdata)
results.append(('Decision Tree Model Accuracy',decisionTreeResult ))
print("Accuracy of decisionTreeModel:",decisionTreeResult)
print(classification_report(targettestdata, targetpredicted))

from sklearn.ensemble import RandomForestClassifier

randomForestModel = RandomForestClassifier(n_estimators = 1)
randomForestModel.fit(featurestraindata,targettraindata)
targetpredicted = randomForestModel.predict(featurestestdata)
randomForestResult= randomForestModel.score(featurestestdata,targettestdata)
results.append(('Random Forest Model Accuracy',randomForestResult ))
print("Accuracy of randomForestModel:",randomForestResult)
print(classification_report(targettestdata, targetpredicted))


from sklearn.ensemble import VotingClassifier
SEED = 1

NaiveBayesModel = GaussianNB()
gaussianKernalSVMModel = SVC(kernel='rbf')
linearSVMModel = SVC(kernel='linear')
randomForestModel = RandomForestClassifier(n_estimators = 300)
knnModel = KNeighborsClassifier(n_neighbors = 3, p = 2)
decisionTreeModel = DecisionTreeClassifier(random_state = SEED)
classifiers = [('Random Forest',randomForestModel),
               ('SVM',linearSVMModel),
               ('Naive',NaiveBayesModel),
               ('K Nearest Neighbours',knnModel),
               ('Classification Tree',decisionTreeModel),
               ('RBF',gaussianKernalSVMModel)
               ]
votingClassifierModel = VotingClassifier(estimators=classifiers)
votingClassifierModel.fit(featurestraindata,targettraindata)
targetpredicted = votingClassifierModel.predict(featurestestdata)
votingClassifierResult= votingClassifierModel.score(featurestestdata,targettestdata)
results.append(('Voting Classifier Model Accuracy',votingClassifierResult ))
print("Accuracy of votingClassifierModel:",votingClassifierResult)
print(classification_report(targettestdata, targetpredicted))

from sklearn.ensemble import BaggingClassifier

knnModel = KNeighborsClassifier(n_neighbors = 3, p = 2)
baggingModel = BaggingClassifier(base_estimator=knnModel, n_estimators = 100,
                       n_jobs=-1)
baggingModel.fit(featurestraindata,targettraindata)
targetpredicted = baggingModel.predict(featurestestdata)
baggingResult= baggingModel.score(featurestestdata,targettestdata)
results.append(('Bagging Model Accuracy',baggingResult ))
print("Accuracy of baggingModel:",baggingResult)
print(classification_report(targettestdata, targetpredicted))

from sklearn.linear_model import LogisticRegression
logisticRegrModel = LogisticRegression()
logisticRegrModel.fit(featurestraindata,targettraindata)
targetpredicted = logisticRegrModel.predict(featurestestdata)
logisticRegrResult= logisticRegrModel.score(featurestestdata,targettestdata)
results.append(('Logistic Regression Accuracy',logisticRegrResult ))
print("Accuracy of LogisticRegressionModel:",logisticRegrResult)
print(classification_report(targettestdata, targetpredicted))

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

mlpModel = MLPClassifier(hidden_layer_sizes=(256),activation="logistic",random_state=1,max_iter=1000)
mlpModel.fit(featurestraindata,targettraindata)
targetpredicted = mlpModel.predict(featurestestdata)
mlpResult= mlpModel.score(featurestestdata,targettestdata)
results.append(('MLP Classifier Accuracy',mlpResult ))
print("Accuracy of MLPClassifierModel:",mlpResult)
print(classification_report(targettestdata, targetpredicted))

results







