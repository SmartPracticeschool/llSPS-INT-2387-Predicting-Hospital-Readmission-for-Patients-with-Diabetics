#IMPORTING LIBRARIES AND DATASET
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle



dataset = pd.read_csv('C:\\Users\sanje\\Desktop\\Yajat\\dataset_diabetes\\diabetic_data.csv')

#PREPROCESSING
#As it is clear from observing the dataset, missing values have been marked by '?'.
#Replacing '?' with NaN 
dataset.replace('?', np.nan, inplace=True)


#patient_nbr is the unique ids of patients
#Removing multiple encounters of the same patient to avoid bias
dataset = dataset.drop_duplicates(subset= ['patient_nbr'], keep = 'first')


#Dropping encounter_id, patient_nbr as they are ids and is assumed would not affect the target = readmitted
#weight has almost 95% missing values, that's why it is better to drop it.
#payer_code will have no significant effect on the readmission rates and moreover it has almost 40% missing values, that's why it is dropped
#medical_specialty has almost 50% missing values, replacing those with mean, mode, etc. would introduce bias. That's why it is dropped.
dataset.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis = 1, inplace = True)

dataset = dataset[dataset.gender != 'Unknown/Invalid']     #records with 'unknown/invalid' tag were dropped
dataset['gender'].value_counts()



#race column has 2273 columns. Since race may depend on the locality of the hospital, most frequent method was chosen
#to fill the missing values

imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
dataset['race'] = imp.fit_transform(dataset[['race']])

#rows with missing values of primary, secondary and tertiary diagnoses is dropped
dataset.drop(dataset[dataset['diag_1'].isnull()].index, inplace = True, axis = 0)
dataset.drop(dataset[dataset['diag_2'].isnull()].index, inplace = True, axis = 0)
dataset.drop(dataset[dataset['diag_3'].isnull()].index, inplace = True, axis = 0)


#Features like citoglipton has all values as NO, which would be insignificant on our target feature. Hence removed.
#While features like metformin-rosiglitazone were removed because they had all values as NO, except for some (less than
#30 values) as steady.
dataset.drop(['citoglipton', 'examide','metformin-pioglitazone', 'metformin-rosiglitazone', 
              'glimepiride-pioglitazone', 'troglitazone', 'tolbutamide', 'acetohexamide'],
              axis = 1, inplace = True)



#Samples where patients have expired or discharged into hospice have been removed. Expired patients won't get readmitted
#for obvious reasons.
expired = [11, 13, 14, 19, 20, 21]
for i in expired:
    dataset.drop(dataset[dataset['discharge_disposition_id'] == i].index, inplace = True, axis = 0)


dataset.drop(['admission_type_id', 'discharge_disposition_id', 'admission_source_id'], axis = 1, inplace = True)


#Hospital needs to pay fines for readmissions within 30 days of 1st admission (as stated in the research paper https://www.hindawi.com/journals/bmri/2014/781670/)
#That's why patients readmitted after 30 days were considered as NO, as in not readmitted.
dataset['readmitted'].replace('>30', 'NO', inplace = True)



#Label encoding textual features such as race and gender
l_encode = LabelEncoder()
dataset['race'] = l_encode.fit_transform(dataset['race'])
dataset['gender'] = l_encode.fit_transform(dataset['gender'])
dataset['age'] = l_encode.fit_transform(dataset['age'])
dataset['diag_1'] = l_encode.fit_transform(dataset['diag_1'])
dataset['diag_2'] = l_encode.fit_transform(dataset['diag_2'])
dataset['diag_3'] = l_encode.fit_transform(dataset['diag_3'])
#Label encoding chemical features such as repaglinide, etc.
cols = list(dataset.columns)
for i in range(14,34):
    dataset[cols[i]] = l_encode.fit_transform(dataset[cols[i]])

#Dropping some more features based on low correlation values
dataset.drop(['race', 'gender', 'num_procedures', 'number_outpatient', 'rosiglitazone'], axis = 1, inplace = True)


#to create a template csv file with all the features. FOR PREDICTION
df1 = dataset.copy()
col = df1.columns.values
col = col[:-1]        #removing the target variable from the list
df = pd.DataFrame(data = None, columns = col , index = None)
df.to_csv('test.csv', sep=',', mode = 'w', index=False)







# x --> independent variables
# y --> dependent/target variable

x = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values


#SPLITTING OF DATA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



#SCALING
scale = MinMaxScaler(feature_range=(0,1))
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)


#TRAINING THE RANDOM FOREST MODEL
rfc = RandomForestClassifier(max_depth = 16, n_estimators=10, random_state=0)
rfc = rfc.fit(x_train, y_train)


#PICKLING OF THE MODEL
pickle.dump(rfc, open('rfc.pkl', 'wb'))













