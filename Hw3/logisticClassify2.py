import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

# Fix the required "not implemented" functions for the homework ("TODO")

################################################################################
## LOGISTIC REGRESSION BINARY CLASSIFIER #######################################
################################################################################


class logisticClassify2(ml.classifier):
    """A binary (2-class) logistic regression classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for logisticClassify2 object.

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array
        """
        self.classes = [0,1]              # (default to 0/1; replace during training)
        self.theta = np.array([])         # placeholder value before training

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"


## METHODS ################################################################

    def plotBoundary(self,X,Y):
        """ Plot the (linear) decision boundary of the classifier, along with data """
        if len(self.theta) != 3: raise ValueError('Data & model must be 2D');
        ax = X.min(0),X.max(0); ax = (ax[0][0],ax[1][0],ax[0][1],ax[1][1]);
        ## TODO: find points on decision boundary defined by theta0 + theta1 X1 + theta2 X2 == 0
        x1b = np.array([ax[0],ax[1]]);  # at X1 = points in x1b
        x2b = (-self.theta[0]-self.theta[1]*x1b)/self.theta[2];      # TODO find x2 values as a function of x1's values
        ## Now plot the data and the resulting boundary:
        A = Y==self.classes[0]; # and plot it:
        plt.plot(X[A,0],X[A,1],'b.',X[~A,0],X[~A,1],'r.',x1b,x2b,'k-'); plt.axis(ax); plt.draw();

    def predictSoft(self, X):
        """ Return the probability of each class under logistic regression """
        raise NotImplementedError
        ## You do not need to implement this function.
        ## If you *want* to, it should return an Mx2 numpy array "P", with
        ## P[:,1] = probability of class 1 = sigma( theta*X )
        ## P[:,0] = 1 - P[:,1] = probability of class 0
        return P

    def predict(self, X):
        """ Return the predictied class of each data point in X"""
        ## TODO: compute linear response r[i] = theta0 + theta1 X[i,1] + theta2 X[i,2] + ... for each i
        ## TODO: if z[i] > 0, predict class 1:  Yhat[i] = self.classes[1]
        ##       else predict class 0:  Yhat[i] = self.classes[0]
        z = self.theta[0] + X.dot(self.theta[1:])
        Yhat = np.asarray(self.classes)[(z > 0).astype(int)]
        return Yhat
    def ltrain(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000,alpha=1,plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        from IPython import display
        M,N = X.shape;                     # initialize the model if necessary:
        self.classes = np.unique(Y);       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X)) # XX is X, but with an extra column of ones
        YY = ml.toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        # init loop variables:
        epoch=0; done=False; Jnll=[]; J01=[];
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                ri = 1.0 / (1.0 + np.exp(-XX[i,:].dot(self.theta)))     # TODO: compute linear response r(x)
                gradi = -(1-ri)*XX[i,:]+alpha*2*self.theta if YY[i] else ri*XX[i,:]+alpha*2*self.theta ;  
                self.theta -= np.float64(stepsize) * np.float64(gradi);  # take a gradient step

            J01.append( self.err(X,Y) )  # evaluate the current error rate

            ## TODO: compute surrogate loss (logistic negative log-likelihood)
            ##  Jsur = - sum_i [ (log si) if yi==1 else (log(1-si)) ]
            JL = 1.0/(1.0+np.exp(-(XX.dot(self.theta))))
            Jnll.append( -np.mean(YY*np.log(JL)+(1-YY)*np.log(1-JL)+alpha*self.theta.dot(self.theta.T)) ) # TODO evaluate the current NLL loss
            display.clear_output(wait=True);
            plt.subplot(1,2,1); plt.cla(); plt.plot(Jnll,'b-',J01,'r-'); # plot losses
            if N==2: plt.subplot(1,2,2); plt.cla(); self.plotBoundary(X,Y); # & predictor if 2D
            plt.pause(.01);                    # let OS draw the plot        

            ## For debugging: you may want to print current parameters & losses
            # print self.theta, ' => ', Jnll[-1], ' / ', J01[-1]
            # raw_input()   # pause for keystroke

            # TODO check stopping criteria: exit if exceeded # of epochs ( > stopEpochs)
            done = epoch>=stopEpochs or (epoch>1 and abs(Jnll[-1]-Jnll[-2])<stopTol);   # or if Jnll not changing between epochs ( < stopTol )
    def train(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000, plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        from IPython import display
        M,N = X.shape;                     # initialize the model if necessary:
        self.classes = np.unique(Y);       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X)) # XX is X, but with an extra column of ones
        YY = ml.toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        # init loop variables:
        epoch=0; done=False; Jnll=[]; J01=[];
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                ri = 1.0 / (1.0 + np.exp(-XX[i,:].dot(self.theta)))     # TODO: compute linear response r(x)
                gradi = -(1-ri)*XX[i,:] if YY[i] else ri*XX[i,:];     # TODO: compute gradient of NLL loss
                self.theta -= np.float64(stepsize) * np.float64(gradi);  # take a gradient step

            J01.append( self.err(X,Y) )  # evaluate the current error rate

            ## TODO: compute surrogate loss (logistic negative log-likelihood)
            ##  Jsur = - sum_i [ (log si) if yi==1 else (log(1-si)) ]
            JL = 1.0/(1.0+np.exp(-(XX.dot(self.theta))))
            Jnll.append( -np.mean(YY*np.log(JL)+(1-YY)*np.log(1-JL)) ) # TODO evaluate the current NLL loss
            display.clear_output(wait=True);
            plt.subplot(1,2,1); plt.cla(); plt.plot(Jnll,'b-',J01,'r-'); # plot losses
            if N==2: plt.subplot(1,2,2); plt.cla(); self.plotBoundary(X,Y); # & predictor if 2D
            plt.pause(.01);                    # let OS draw the plot        

            ## For debugging: you may want to print current parameters & losses
            # print self.theta, ' => ', Jnll[-1], ' / ', J01[-1]
            # raw_input()   # pause for keystroke

            # TODO check stopping criteria: exit if exceeded # of epochs ( > stopEpochs)
            done = epoch>=stopEpochs or (epoch>1 and abs(Jnll[-1]-Jnll[-2])<stopTol);   # or if Jnll not changing between epochs ( < stopTol )

    ################################################################################
################################################################################
################################################################################
