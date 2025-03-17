import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

data = 'car_evaluation.csv'
df = pd.read_csv(data, header=None)
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names

X = df.drop(['class'], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# 1. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print('Random Forest')
print('Accuracy:', accuracy_score(y_test, rf_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, rf_pred))
print('Classification Report:\n', classification_report(y_test, rf_pred))
print('-----------------------------------------------------')

# 2. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print('Decision Tree')
print('Accuracy:', accuracy_score(y_test, dt_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, dt_pred))
print('Classification Report:\n', classification_report(y_test, dt_pred))
print('-----------------------------------------------------')

# 3. Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print('Logistic Regression')
print('Accuracy:', accuracy_score(y_test, lr_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, lr_pred))
print('Classification Report:\n', classification_report(y_test, lr_pred))
print('-----------------------------------------------------')

# 4. Support Vector Machine (SVM)
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print('Support Vector Machine')
print('Accuracy:', accuracy_score(y_test, svm_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, svm_pred))
print('Classification Report:\n', classification_report(y_test, svm_pred))
print('-----------------------------------------------------')

# Model Accuracy Karşılaştırması
models = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'SVM']
accuracies = [
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, svm_pred)
]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'coral', 'lightgray'])
plt.title('Model Karşılaştırması (Accuracy)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
plt.show()


results_df = X_test.copy()
results_df['y_test'] = y_test.values  
results_df['RF_pred'] = rf_pred
results_df['DT_pred'] = dt_pred
results_df['LR_pred'] = lr_pred
results_df['SVM_pred'] = svm_pred
results_df.to_excel('model_predictions.xlsx', engine='openpyxl', index=False)
print("Sonuçlar 'model_predictions.xlsx' dosyasına kaydedildi.")


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

cm_rf = confusion_matrix(y_test, rf_pred)
cm_dt = confusion_matrix(y_test, dt_pred)
cm_lr = confusion_matrix(y_test, lr_pred)
cm_svm = confusion_matrix(y_test, svm_pred)

cms = [cm_rf, cm_dt, cm_lr, cm_svm]
titles = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'SVM']

for i, ax in enumerate(axes.flatten()):
    sns.heatmap(cms[i], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {titles[i]}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.show()
