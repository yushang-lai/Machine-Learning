#/usr/bin/python

import numpy as np
np.random.seed(13254);

xy_tr = np.loadtxt('../../../../../2018F-178/Project/train.txt');
Xtr,Ytr = xy_tr[:,:-1],xy_tr[:,-1];

###########################################################################################

X=Xtr; Y=Ytr; 
order = np.random.permutation( X.shape[0] );
nTrain = 20000;
nTest  = 20000;
Xtr,Ytr = X[order[:nTrain],:], Y[order[:nTrain]]
Xte,Yte = X[order[nTrain:nTrain+nTest],:], Y[order[nTrain:nTrain+nTest]]


###########################################################################################

Ytr01 = (Ytr>0.01).astype('int');
Yte01 = (Yte>0.01).astype('int');
np.savetxt('X_train.txt',Xtr,'%.4e');
np.savetxt('X_test.txt',Xte,'%.4e');

np.savetxt('Y_train.txt',Ytr01,'%d');
np.savetxt('Y_test.txt',Yte01,'%d');






