import pandas as pd
import numpy as np
import math
#df = pd.read_excel(open('../data/default of credit card clients','rb'))

xlsx = pd.ExcelFile('../data/default of credit card clients_new.xls')
df = xlsx.parse("Data")

X_data = df [['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23']]
Y_data = df [['Y']]

X_data['X1']= (X_data['X1']-X_data['X1'].mean())/(X_data['X1'].std())
X_data['X5']= (X_data['X5']-X_data['X5'].mean())/(X_data['X5'].std())
X_data['X12']= (X_data['X12']-X_data['X12'].mean())/(X_data['X12'].std())
X_data['X13']= (X_data['X13']-X_data['X13'].mean())/(X_data['X13'].std())
X_data['X14']= (X_data['X14']-X_data['X14'].mean())/(X_data['X14'].std())
X_data['X15']= (X_data['X15']-X_data['X15'].mean())/(X_data['X15'].std())
X_data['X16']= (X_data['X16']-X_data['X16'].mean())/(X_data['X16'].std())
X_data['X17']= (X_data['X17']-X_data['X17'].mean())/(X_data['X17'].std())
X_data['X18']= (X_data['X18']-X_data['X18'].mean())/(X_data['X18'].std())
X_data['X19']= (X_data['X19']-X_data['X19'].mean())/(X_data['X19'].std())
X_data['X20']= (X_data['X20']-X_data['X20'].mean())/(X_data['X20'].std())
X_data['X21']= (X_data['X21']-X_data['X21'].mean())/(X_data['X21'].std())
X_data['X22']= (X_data['X22']-X_data['X22'].mean())/(X_data['X22'].std())
X_data['X23']= (X_data['X23']-X_data['X23'].mean())/(X_data['X23'].std())

X_np = X_data.as_matrix()
Y_np = Y_data.as_matrix()
data = np.concatenate((X_np, Y_np), axis=1)
np.random.shuffle(data)

Y = data[:,data.shape[1]-1]
X = data[:,:data.shape[1]- 1]
#Y_np=np.reshape(Y_np, 30000)

##########################################
# Resampling due to imbalanced data
##########################################
n_class , class_count = np.unique (Y, return_counts =True)
nsamp_class = int(math.ceil(np.median (class_count)))

def sampling(X,Y, nsamp): 
    X_train_samp = np.array([]).reshape(0,X.shape[0])
    Y_train_samp = np.array([]).reshape(0,1)

    Xsampled_list = []
    Ysampled_list = []

    for y in range (0, n_class.shape[0]):
        idxs = np.flatnonzero(Y == y)
        if(class_count[y]>nsamp):
            idxs = np.random.choice(idxs, nsamp, replace=False)     
        else: 
            idxs = np.random.choice(idxs, nsamp, replace=True)    
        for i, idx in enumerate(idxs):
            Xsampled_list.extend(X[idx].reshape(1,X.shape[1]))
            Ysampled_list.append(Y[idx])
    X_sampled = np.asarray(Xsampled_list)
    Y_sampled = np.asarray(Ysampled_list)
    return X_sampled, Y_sampled




def Rotation_Matrix_Perturbation(ang, X): 
    ##########################################
    # Rotation Matrix Generation
    ##########################################
    ang_rad= (np.pi/180)*(ang)
    R = np.matrix([[np.cos(ang_rad), -np.sin(ang_rad)], [np.sin(ang_rad), np.cos(ang_rad)]])
    R_row =R
    val_division, val_remainder= divmod(X_data.shape[1], 2)
    # Randomly pick column and row
    #rand_row = (R[np.random.randint(R.shape[0], size =1), :])
    #rand_col = (R[: , np.random.randint(R.shape[1], size =1)])
    
    #duplicate rows to grow column 
    for x in range(1, val_division + 1):
        if x != val_division:
            R_row = np.concatenate((R_row, R), axis=0)
        else:
            rand_row = (R[np.random.randint(R.shape[0], size =1), :])
            R_row = np.concatenate((R_row, rand_row), axis=0)   
    
    #duplicate column to grow row 
    R_col = R_row
    for x in range(1, val_division + 1):
        if x != val_division:
            R_col = np.concatenate((R_col, R_row), axis=1)
        else:
            rand_col = (R_row[: , np.random.randint(R_row.shape[1], size =1)])
            R_col = np.concatenate((R_col, rand_col), axis=1)  
            
    R_mat = R_col
    ##########################################
    # Apply Perturbation
    ##########################################
    X_tran = np.transpose(X)
    X_pertub = np.dot(R_mat, X_tran)
    X_Perturbation = np.transpose(X_pertub)
    return X_Perturbation


##########################################
# Classification
##########################################
def runknn(X_input, Y_input, n_neighbor): 
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(Y_input)
    target = le.transform(Y_input)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_input, target, test_size=0.9, random_state=42)

    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbor)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(neigh, X_train, y_train, cv=10)
    print(scores)  

    from sklearn import metrics
    from sklearn.model_selection import cross_val_predict
    Y_predict = cross_val_predict(neigh, X_test, y_test, cv=10)
    print ('Accuracy')
    print (metrics.accuracy_score(y_test, Y_predict))
    print ('Prediction')
    print (metrics.precision_score(y_test, Y_predict))
    print ('Recall')
    print (metrics.recall_score(y_test, Y_predict) )
    
    
def compute_ti(Y_i): 
    t_i = np.min(Y_i)/(np.max(Y_i)-np.min(Y_i))
    return t_i

def compute_si(Y_i): 
    s_i = 1/ (np.max(Y_i)-np.min(Y_i))
    return s_i


def compute_Ysi (X):
    X_trans = []
    for x in range(0, X.shape[1]):
        X_col = X[:,x]
        y_si = compute_si(X_col - compute_ti(X_col))
        X_trans.append(y_si)
    X_trans_array = np.asarray(X_trans)
    return (X_trans_array)


def compute_privacymetric(Yprime_si,Y_si):
    Std_D = np.std(Yprime_si - Y_si)
    Var_D = np.square(Std_D)
    print ("Unified Column Privacy Metric")
    print (Var_D)
    Min_PM = np.min(Yprime_si - Y_si)
    print ("Minimum Privacy Metric")
    print (Min_PM)
    Avg_PM = np.mean(Yprime_si - Y_si)
    print ("Average Privacy Metric")
    print (Avg_PM)

X_samp, Y_samp = sampling(X,Y,nsamp_class)
# Testing performance of raw data vs sampled data
runknn(X,Y,1)
runknn(X_samp,Y_samp,1)

# Testing performance of perturbed sampled data
X_5 = Rotation_Matrix_Perturbation(5, X_samp)
runknn(X_5,Y_samp,1)

Yprime_si = compute_Ysi (X_samp)
Y_si = compute_Ysi (X_5)
compute_privacymetric(Yprime_si,Y_si)

#X_20 = Rotation_Matrix_Perturbation(20, X_samp)
#runknn(X_20,Y_samp,1)
#Yprime_si = compute_Ysi (X_samp)
#Y_si = compute_Ysi (X_20)
#compute_privacymetric(Yprime_si,Y_si)


