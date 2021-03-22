import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import datasets
import seaborn as sns
import sys
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data[:, 2:]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

num_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100

mlflow.sklearn.autolog()
with mlflow.start_run(run_name="Iris RF Experiment") as run:
    rf = RandomForestClassifier(n_estimators=num_estimators)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print("  Accuracy: %f" % acc)
    
    print("Confusion Matrix : ")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    target = ['setosa', 'versicolor', 'virginica']
    sns.heatmap(cm, annot=True, xticklabels=target, yticklabels=target)
    plt.show()
    
    report = classification_report(y_test, predictions, target_names=target)
    print('Classification Report : ')
    print(report)
    