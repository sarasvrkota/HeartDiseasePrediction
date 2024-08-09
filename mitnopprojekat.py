import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
##########################################################################################################################################
#PRIPREMA PODATAKA

# Učitavanje podataka
data = pd.read_csv('projekat.csv', header=None)

# Dodavanje imena kolona
column_names = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 
                'serum_cholestoral', 'fasting_blood_sugar', 'rest_ecg', 
                'max_heart_rate_achieved', 'exercise_induced_angina', 
                'oldpeak', 'slope', 'num_major_vessels', 'thalassemia', 'class']
data.columns = column_names

# Prikaz prvih nekoliko redova
print(data.head())

# Provera nedostajućih vrednosti
missing_values = data.isnull().sum()
print("Nedostajuće vrednosti:")
print(missing_values)
# Nema nedostajućih vrednosti u ovim podacima


# Pretvaranje kolone 'class' u binarni atribut
data['class'] = data['class'].map({'present': 1, 'absent': 0})

# Normalizacija podataka
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data.drop(['class'], axis=1)), columns=data.columns[:-1])

# Standardizacija podataka
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data.drop(['class'], axis=1)), columns=data.columns[:-1])

# Prikaz normalizovanih i standardizovanih podataka
print("\nNormalizovani podaci:")
print(data_normalized.head())

print("\nStandardizovani podaci:")
print(data_standardized.head())

# Statistički pregled podataka
print("\nStatistički pregled podataka:")
print(data.describe())

##############################################################################################################################

#GRAFIKONI-analiza 

import matplotlib.pyplot as plt
import seaborn as sns

# Provera broja osoba u svakoj starosnoj grupi
bins = [29, 39, 49, 59, 100]  # Uključujući i mlađe osobe
labels = ['30-39', '40-49', '50-59', '60+']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)

# Prikaz broja osoba po starosnim grupama
age_group_counts = data['age_group'].value_counts().sort_index()
print("Broj osoba po starosnim grupama:")
print(age_group_counts)

# Kreiranje grafikona brojnosti po starosnim grupama
plt.figure(figsize=(8, 6))
sns.countplot(x='age_group', hue='class', data=data)
plt.title('Brojnost po starosnim grupama sa i bez srčanih bolesti')
plt.xlabel('Starosna grupa')
plt.ylabel('Brojnost')

# Postavljanje legendi
plt.legend(['Nema', 'Ima'], title='Srčane bolesti', loc='upper right')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns


# Procenat osoba sa srčanim bolestima po polu
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', hue='class', data=data)
plt.title('Broj osoba sa srčanim bolestima po polu')
plt.xlabel('Pol (0 = ženski, 1 = muški)')
plt.ylabel('Brojnost')
plt.legend(['Bez bolesti', 'Sa bolešću'], loc='upper right')
plt.show()



###############################################################################################################################3
#PRAVLJENJE MODELA 


#LOGISTICKA REGRESIJA   
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Podela podataka na trening i test skup
X = data.drop(['class', 'age_group'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kreiranje logističkog modela
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)

# Predviđanja na test skupu
y_pred_lr = model_lr.predict(X_test)

# Evaluacija modela
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistička regresija - Tačnost: {accuracy_lr:.2f}')

# Matrica konfuzije
conf_mat_lr = confusion_matrix(y_test, y_pred_lr)
print('Matrica konfuzije:')
print(conf_mat_lr)

# Izveštaj o klasifikaciji
class_report_lr = classification_report(y_test, y_pred_lr)
print('Izveštaj o klasifikaciji:')
print(class_report_lr)


#SLUCAJNE SUME
from sklearn.ensemble import RandomForestClassifier

# Kreiranje modela slučajnih šuma
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Predviđanja na test skupu
y_pred_rf = model_rf.predict(X_test)

# Evaluacija modela
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Slučajne šume - Tačnost: {accuracy_rf:.2f}')

# Matrica konfuzije
conf_mat_rf = confusion_matrix(y_test, y_pred_rf)
print('Matrica konfuzije:')
print(conf_mat_rf)

# Izveštaj o klasifikaciji
class_report_rf = classification_report(y_test, y_pred_rf)
print('Izveštaj o klasifikaciji:')
print(class_report_rf)


#XGBOOST
# Kreiranje modela XGBoost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Kreiranje modela XGBoost
model_xgb = XGBClassifier(random_state=42)
model_xgb.fit(X_train, y_train)

# Predviđanja na test skupu
y_pred_xgb = model_xgb.predict(X_test)

# Evaluacija modela
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'XGBoost - Tačnost: {accuracy_xgb:.2f}')

# Matrica konfuzije
conf_mat_xgb = confusion_matrix(y_test, y_pred_xgb)
print('Matrica konfuzije:')
print(conf_mat_xgb)

# Izveštaj o klasifikaciji
class_report_xgb = classification_report(y_test, y_pred_xgb)
print('Izveštaj o klasifikaciji:')
print(class_report_xgb)

# Podela podataka na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kreiranje i evaluacija modela logističke regresije
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_mat_lr = confusion_matrix(y_test, y_pred_lr)
class_report_lr = classification_report(y_test, y_pred_lr)

print("\nLogistička regresija:")
print(f'Tačnost: {accuracy_lr:.2f}')
print('Matrica konfuzije:')
print(conf_mat_lr)
print('Izveštaj o klasifikaciji:')
print(class_report_lr)

# Kreiranje i evaluacija modela slučajnih šuma
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_mat_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)

print("\nSlučajne šume:")
print(f'Tačnost: {accuracy_rf:.2f}')
print('Matrica konfuzije:')
print(conf_mat_rf)
print('Izveštaj o klasifikaciji:')
print(class_report_rf)

# Kreiranje i evaluacija modela XGBoost
model_xgb = XGBClassifier(random_state=42)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
conf_mat_xgb = confusion_matrix(y_test, y_pred_xgb)
class_report_xgb = classification_report(y_test, y_pred_xgb)

print("\nXGBoost:")
print(f'Tačnost: {accuracy_xgb:.2f}')
print('Matrica konfuzije:')
print(conf_mat_xgb)
print('Izveštaj o klasifikaciji:')
print(class_report_xgb)
########################################################################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Kreiranje i evaluacija modela gradijentnog pojačavanja
model_gb = GradientBoostingClassifier(random_state=42)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
conf_mat_gb = confusion_matrix(y_test, y_pred_gb)
class_report_gb = classification_report(y_test, y_pred_gb)

print("\nGradijentno pojačavanje:")
print(f'Tačnost: {accuracy_gb:.2f}')
print('Matrica konfuzije:')
print(conf_mat_gb)
print('Izveštaj o klasifikaciji:')
print(class_report_gb)

# Prikaz rezultata kroz grafike
plt.figure(figsize=(14, 10))

# Grafikon tačnosti
plt.subplot(2, 2, 1)
models = ['Logistička regresija', 'Slučajne šume', 'XGBoost', 'Gradijentno pojačavanje']
accuracies = [accuracy_lr, accuracy_rf, accuracy_xgb, accuracy_gb]
sns.barplot(x=models, y=accuracies)
plt.ylim(0.7, 1)
plt.title('Tačnost modela')
plt.ylabel('Tačnost')
plt.xlabel('Model')

# Matrice konfuzije
plt.subplot(2, 2, 2)
plt.title('Matrice konfuzije')
sns.heatmap(conf_mat_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predviđeno')
plt.ylabel('Stvarno')

plt.subplot(2, 2, 3)
sns.heatmap(conf_mat_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predviđeno')
plt.ylabel('Stvarno')

plt.subplot(2, 2, 4)
sns.heatmap(conf_mat_xgb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predviđeno')
plt.ylabel('Stvarno')

plt.tight_layout()
plt.show()

##########################################

# Izračunavanje osetljivosti, specifičnosti i F1-mere za logističku regresiju
tn_lr, fp_lr, fn_lr, tp_lr = conf_mat_lr.ravel()
sensitivity_lr = tp_lr / (tp_lr + fn_lr)
specificity_lr = tn_lr / (tn_lr + fp_lr)
precision_lr = tp_lr / (tp_lr + fp_lr)
f1_lr = 2 * (precision_lr * sensitivity_lr) / (precision_lr + sensitivity_lr)

# Izračunavanje osetljivosti, specifičnosti i F1-mere za slučajne šume
tn_rf, fp_rf, fn_rf, tp_rf = conf_mat_rf.ravel()
sensitivity_rf = tp_rf / (tp_rf + fn_rf)
specificity_rf = tn_rf / (tn_rf + fp_rf)
precision_rf = tp_rf / (tp_rf + fp_rf)
f1_rf = 2 * (precision_rf * sensitivity_rf) / (precision_rf + sensitivity_rf)

# Izračunavanje osetljivosti, specifičnosti i F1-mere za XGBoost
tn_xgb, fp_xgb, fn_xgb, tp_xgb = conf_mat_xgb.ravel()
sensitivity_xgb = tp_xgb / (tp_xgb + fn_xgb)
specificity_xgb = tn_xgb / (tn_xgb + fp_xgb)
precision_xgb = tp_xgb / (tp_xgb + fp_xgb)
f1_xgb = 2 * (precision_xgb * sensitivity_xgb) / (precision_xgb + sensitivity_xgb)

# Izračunavanje osetljivosti, specifičnosti i F1-mere za gradijentno pojačavanje
tn_gb, fp_gb, fn_gb, tp_gb = conf_mat_gb.ravel()
sensitivity_gb = tp_gb / (tp_gb + fn_gb)
specificity_gb = tn_gb / (tn_gb + fp_gb)
precision_gb = tp_gb / (tp_gb + fp_gb)
f1_gb = 2 * (precision_gb * sensitivity_gb) / (precision_gb + sensitivity_gb)

# Prikaz rezultata osetljivosti, specifičnosti i F1-mere za svaki model
print("\nLogistička regresija:")
print(f'Osetljivost: {sensitivity_lr:.2f}')
print(f'Specifičnost: {specificity_lr:.2f}')
print(f'F1-mera: {f1_lr:.2f}')

print("\nSlučajne šume:")
print(f'Osetljivost: {sensitivity_rf:.2f}')
print(f'Specifičnost: {specificity_rf:.2f}')
print(f'F1-mera: {f1_rf:.2f}')

print("\nXGBoost:")
print(f'Osetljivost: {sensitivity_xgb:.2f}')
print(f'Specifičnost: {specificity_xgb:.2f}')
print(f'F1-mera: {f1_xgb:.2f}')

print("\nGradijentno pojačavanje:")
print(f'Osetljivost: {sensitivity_gb:.2f}')
print(f'Specifičnost: {specificity_gb:.2f}')
print(f'F1-mera: {f1_gb:.2f}')

# Prikaz rezultata kroz grafike
plt.figure(figsize=(14, 10))

# Grafikon tačnosti
plt.subplot(2, 2, 1)
models = ['Logistička regresija', 'Slučajne šume', 'XGBoost', 'Gradijentno pojačavanje']
accuracies = [accuracy_lr, accuracy_rf, accuracy_xgb, accuracy_gb]
sns.barplot(x=models, y=accuracies)
plt.ylim(0.7, 1)
plt.title('Tačnost modela')
plt.ylabel('Tačnost')
plt.xlabel('Model')

# Grafikon osetljivosti
plt.subplot(2, 2, 2)
sensitivities = [sensitivity_lr, sensitivity_rf, sensitivity_xgb, sensitivity_gb]
sns.barplot(x=models, y=sensitivities)
plt.title('Osetljivost modela')
plt.ylabel('Osetljivost')
plt.xlabel('Model')

# Grafikon specifičnosti
plt.subplot(2, 2, 3)
specificities = [specificity_lr, specificity_rf, specificity_xgb, specificity_gb]
sns.barplot(x=models, y=specificities)
plt.title('Specifičnost modela')
plt.ylabel('Specifičnost')
plt.xlabel('Model')

# Grafikon F1-mere
plt.subplot(2, 2, 4)
f1_scores = [f1_lr, f1_rf, f1_xgb, f1_gb]
sns.barplot(x=models, y=f1_scores)
plt.title('F1-mera modela')
plt.ylabel('F1-mera')
plt.xlabel('Model')

plt.tight_layout()
plt.show()
################################################################
#KOEFICIJENTI
# Koeficijenti logističke regresije
coefficients = pd.DataFrame(data=model_lr.coef_[0], index=X.columns, columns=['Coefficient'])
coefficients.sort_values(by='Coefficient', ascending=False, inplace=True)
print(coefficients)
# Važnost atributa za slučajne šume
feature_importance_rf = pd.DataFrame(model_rf.feature_importances_, index=X.columns, columns=['Importance'])
feature_importance_rf.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importance_rf)
# Važnost atributa za XGBoost
feature_importance_xgb = pd.DataFrame(model_xgb.feature_importances_, index=X.columns, columns=['Importance'])
feature_importance_xgb.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importance_xgb)
# Važnost atributa za gradijentno pojačavanje
feature_importance_gb = pd.DataFrame(model_gb.feature_importances_, index=X.columns, columns=['Importance'])
feature_importance_gb.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importance_gb)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Podaci
coefficients = pd.DataFrame(data=model_lr.coef_[0], index=X.columns, columns=['Coefficient'])
coefficients.sort_values(by='Coefficient', ascending=False, inplace=True)

feature_importance_rf = pd.DataFrame(model_rf.feature_importances_, index=X.columns, columns=['Importance'])
feature_importance_rf.sort_values(by='Importance', ascending=False, inplace=True)

feature_importance_xgb = pd.DataFrame(model_xgb.feature_importances_, index=X.columns, columns=['Importance'])
feature_importance_xgb.sort_values(by='Importance', ascending=False, inplace=True)

feature_importance_gb = pd.DataFrame(model_gb.feature_importances_, index=X.columns, columns=['Importance'])
feature_importance_gb.sort_values(by='Importance', ascending=False, inplace=True)

# Vizualizacija koeficijenata logističke regresije
plt.figure(figsize=(10, 6))
sns.barplot(x=coefficients['Coefficient'], y=coefficients.index, palette='viridis')
plt.title('Koeficijenti logističke regresije')
plt.xlabel('Koeficijent')
plt.ylabel('Atribut')
plt.show()

# Vizualizacija važnosti atributa za slučajne šume
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_rf['Importance'], y=feature_importance_rf.index, palette='viridis')
plt.title('Važnost atributa za slučajne šume')
plt.xlabel('Važnost')
plt.ylabel('Atribut')
plt.show()

# Vizualizacija važnosti atributa za XGBoost
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_xgb['Importance'], y=feature_importance_xgb.index, palette='viridis')
plt.title('Važnost atributa za XGBoost')
plt.xlabel('Važnost')
plt.ylabel('Atribut')
plt.show()

# Vizualizacija važnosti atributa za gradijentno pojačavanje
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_gb['Importance'], y=feature_importance_gb.index, palette='viridis')
plt.title('Važnost atributa za gradijentno pojačavanje')
plt.xlabel('Važnost')
plt.ylabel('Atribut')
plt.show()
