import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn import model_selection
from math import sqrt

class basicAnalysis:
    def __init__(self):
        self.train = pd.read_csv("train.csv")

    #computes the average purchase price of each product in the training set
    def computeAverages(self):
        uniqueProducts = self.train['Product_ID'].unique()
        self.productdict = {elem : pd.DataFrame for elem in uniqueProducts}
        for key in self.productdict.keys():
            self.productdict[key] = self.train[1:][self.train['Product_ID'] == key]
        print("Dictionary Complete")

        valsarray = np.zeros(len(self.productdict))
        self.vals = pd.Series(data=np.zeros(uniqueProducts.size),index=uniqueProducts) #pandas series containing average values for each product
        for key in self.productdict.keys():
            x = self.productdict[key]['Purchase'].sum() / self.productdict[key]['Purchase'].size
            self.vals[key] = x

    #computes the average for each age bracket-product id pair in the training set
    def computeAveragesByAge(self):
        uniqueProducts = self.train['Product_ID'].unique()
        self.productdict = {elem : pd.DataFrame for elem in uniqueProducts}
        for key in self.productdict.keys():
            self.productdict[key] = self.train[1:][self.train['Product_ID'] == key]
        print("Dictionary Complete")

        self.vals0_17 = pd.Series(data=np.zeros(uniqueProducts.size),index=uniqueProducts)
        self.vals18_25 = pd.Series(data=np.zeros(uniqueProducts.size),index=uniqueProducts)
        self.vals26_35 = pd.Series(data=np.zeros(uniqueProducts.size),index=uniqueProducts)
        self.vals36_45 = pd.Series(data=np.zeros(uniqueProducts.size),index=uniqueProducts)
        self.vals46_50 = pd.Series(data=np.zeros(uniqueProducts.size),index=uniqueProducts)
        self.vals51_55 = pd.Series(data=np.zeros(uniqueProducts.size),index=uniqueProducts)
        self.vals55 = pd.Series(data=np.zeros(uniqueProducts.size),index=uniqueProducts)

        for key in self.productdict.keys():
            if (self.productdict[key][self.productdict[key]['Age'] == '0-17']['Purchase'].size > 0):
                x0 = self.productdict[key][self.productdict[key]['Age'] == '0-17']['Purchase'].sum() / self.productdict[key][self.productdict[key]['Age'] == '0-17']['Purchase'].size
                self.vals0_17[key] = x0
            else:
                self.vals0_17[key] = 0
            if (self.productdict[key][self.productdict[key]['Age'] == '18-25']['Purchase'].size > 0):
                x0 = self.productdict[key][self.productdict[key]['Age'] == '18-25']['Purchase'].sum() / self.productdict[key][self.productdict[key]['Age'] == '18-25']['Purchase'].size
                self.vals18_25[key] = x0
            else:
                self.vals18_25[key] = 0
            if (self.productdict[key][self.productdict[key]['Age'] == '26-35']['Purchase'].size > 0):
                x0 = self.productdict[key][self.productdict[key]['Age'] == '26-35']['Purchase'].sum() / self.productdict[key][self.productdict[key]['Age'] == '26-35']['Purchase'].size
                self.vals26_35[key] = x0
            else:
                self.vals26_35[key] = 0
            if (self.productdict[key][self.productdict[key]['Age'] == '36-45']['Purchase'].size > 0):
                x0 = self.productdict[key][self.productdict[key]['Age'] == '36-45']['Purchase'].sum() / self.productdict[key][self.productdict[key]['Age'] == '36-45']['Purchase'].size
                self.vals36_45[key] = x0
            else:
                self.vals36_45[key] = 0
            if (self.productdict[key][self.productdict[key]['Age'] == '46-50']['Purchase'].size > 0):
                x0 = self.productdict[key][self.productdict[key]['Age'] == '46-50']['Purchase'].sum() / self.productdict[key][self.productdict[key]['Age'] == '46-50']['Purchase'].size
                self.vals46_50[key] = x0
            else:
                self.vals46_50[key] = 0
            if (self.productdict[key][self.productdict[key]['Age'] == '51-55']['Purchase'].size > 0):
                x0 = self.productdict[key][self.productdict[key]['Age'] == '51-55']['Purchase'].sum() / self.productdict[key][self.productdict[key]['Age'] == '51-55']['Purchase'].size
                self.vals51_55[key] = x0
            else:
                self.vals51_55[key] = 0
            if (self.productdict[key][self.productdict[key]['Age'] == '55+']['Purchase'].size > 0):
                x0 = self.productdict[key][self.productdict[key]['Age'] == '55+']['Purchase'].sum() / self.productdict[key][self.productdict[key]['Age'] == '55+']['Purchase'].size
                self.vals55[key] = x0
            else:
                self.vals55[key] = 0
        print(self.vals0_17)

    #function to map a data entry to a value based on the age and product ID 
    def calcVals(self,entry):
        if entry['Age'] == '0-17':
            if self.vals0_17[entry['Product_ID']] != 0:
                return self.vals0_17[entry['Product_ID']]
            else:
                return self.vals[entry['Product_ID']]
        if entry['Age'] == '18-25':
            if self.vals18_25[entry['Product_ID']] != 0:
                return self.vals18_25[entry['Product_ID']]
            else:
                return self.vals[entry['Product_ID']]
        if entry['Age'] == '26-35':
            if self.vals26_35[entry['Product_ID']] != 0:
                return self.vals26_35[entry['Product_ID']]
            else:
                return self.vals[entry['Product_ID']]
        if entry['Age'] == '36-45':
            if self.vals36_45[entry['Product_ID']] != 0:
                return self.vals36_45[entry['Product_ID']]
            else:
                return self.vals[entry['Product_ID']]
        if entry['Age'] == '46-50':
            if self.vals46_50[entry['Product_ID']] != 0:
                return self.vals46_50[entry['Product_ID']]
            else:
                return self.vals[entry['Product_ID']]
        if entry['Age'] == '51-55':
            if self.vals51_55[entry['Product_ID']] != 0:
                return self.vals51_55[entry['Product_ID']]
            else:
                return self.vals[entry['Product_ID']]
        if entry['Age'] == '55+':
            if self.vals55[entry['Product_ID']] != 0:
                return self.vals55[entry['Product_ID']]
            else:
                return self.vals[entry['Product_ID']]
        
    #function to determine the RMS error for the product ID averages
    def predict(self):
        self.computeAverages()
        self.train['Predicted_Vals'] = self.train['Product_ID'].apply(lambda x: self.vals[x])
        error = self.train['Predicted_Vals'] - self.train['Purchase']
        vals = np.dot(error,error)/len(error)
        result = np.sqrt(vals)
        print(result)

    #function to determine the RMS error for the age / product ID pair averages
    def predictWithAges(self):
        self.computeAverages()
        self.computeAveragesByAge()
        self.train['Predicted_Vals'] = self.train.apply(lambda x: self.calcVals(x),axis=1)
        print(self.train['Predicted_Vals'])
        error = self.train['Predicted_Vals'] - self.train['Purchase']
        vals = np.dot(error,error)/len(error)
        result = np.sqrt(vals)
        print(result)
        
#v1 = basicAnalysis()
#v1.predictWithAges()

#################### STOCHASTIC GRADIENT DESCENT ####################
        
class SGDRegression:
    def __init__(self):
        self.train = pd.read_csv("train.csv")
        self.test = pd.read_csv("test.csv")

    #converts categorical variables into dummy variables for both the training and the test sets
    def getDummies(self):
        self.traindummies = pd.get_dummies(self.train,columns=["Gender","Age","Occupation","City_Category",
                                         "Stay_In_Current_City_Years","Marital_Status",
                                         "Product_Category_1","Product_Category_2",
                                         "Product_Category_3"])                        #adjusted data frame with dummy variables
        self.testdummies = pd.get_dummies(self.test,columns=["Gender","Age","Occupation","City_Category",
                                         "Stay_In_Current_City_Years","Marital_Status",
                                         "Product_Category_1","Product_Category_2",
                                         "Product_Category_3"])                        #adjusted data frame with dummy variables
        print(self.testdummies.columns)
        x = self.traindummies.drop(['Purchase'],axis=1)
        for col in x.columns:
            if col not in self.testdummies.columns:
                self.traindummies = self.traindummies.drop(col,axis=1)

    #trains the data using a scikit stochastic gradient descent algorithm
    def SGDtraining(self,train):
        self.getDummies()
        sgdreg = SGDRegressor(penalty='l2', alpha=0.001, n_iter=200) #regressor for algorithm
        df_x = self.traindummies.drop(['User_ID','Product_ID','Purchase'],axis=1)
        df_y = self.traindummies['Purchase']
        sgdreg.fit(df_x,df_y)
        if train == True:
            p = sgdreg.predict(df_x)
            err = p - df_y
            total_error = np.dot(err,err)
            rmse = np.sqrt(total_error/len(p))
        else:
            p = sgdreg.predict(self.testdummies.drop(['User_ID','Product_ID'],axis=1))
        return p

    #performs cross validation for the SGD algorithm on the training set, to determine whether it will work consistently well
    def SGDCV(self):
        self.getDummies()

        self.shuffled = self.traindummies.sample(frac=1) #shuffles data
        self.x = self.shuffled.drop(['User_ID','Product_ID','Purchase'],axis=1)
        self.y = self.shuffled['Purchase']
        
        sgdreg = SGDRegressor(penalty='l2', alpha=0.0001, n_iter=200) #regressor for algorithm
        scores = model_selection.cross_val_score(sgdreg,self.x,self.y,scoring='neg_mean_squared_error',cv=5) #performs regression with cross validation
        scores = np.sqrt(-1*scores)
        
        print(scores)

    #performs a separate SGD regression for each product, assuming that it is purchased by at least 50 users, if not, the general SGD equation is used
    #this algorithm was the most successful, yielding a RMS error of 2776.55 on the test set
    def SGDsplit(self,train):
        self.getDummies()
        uniqueProducts = self.traindummies['Product_ID'].unique()
        self.productdict = {elem : pd.DataFrame for elem in uniqueProducts}
        
        for key in self.productdict.keys():
            self.productdict[key] = self.traindummies[1:][self.traindummies['Product_ID'] == key]
        print("Dictionary Complete")
        learningdict = {}
        
        for key in self.productdict.keys():
            if self.productdict[key].size > 50:
                sgdreg = SGDRegressor(penalty='l2', alpha=0.0001, n_iter=200)
                newdf = self.productdict[key].drop(['User_ID','Product_ID','Purchase'],axis=1)
                y = self.productdict[key]['Purchase']
                sgdreg.fit(newdf,y)
                learningdict[key] = sgdreg
        print("Begin")
        if train == True:
            finalvals = np.zeros(550068)
            for index, row in self.traindummies.iterrows():
                if row['Product_ID'] in learningdict.keys():
                    finalvals[index] = learningdict[row['Product_ID']].predict(row.drop(['User_ID','Product_ID','Purchase']).reshape(1,-1))
                    #print(finalvals[index])
                if index%500 == 0:
                    print(index)
        else:
            finalvals = np.zeros(233599)
            for index, row in self.testdummies.iterrows():
                #print(index)
                if row['Product_ID'] in learningdict.keys():
                    finalvals[index] = learningdict[row['Product_ID']].predict(row.drop(['User_ID','Product_ID']).reshape(1,-1))
                    #print(finalvals[index])
                if index%500 == 0:
                    print(index)
        print("Done")
        print(finalvals)
        return finalvals

################### K NEAREST NEIGHBORS ####################

class KNNRegression:
    def __init__(self):
        self.train = pd.read_csv("train.csv")
        self.dist = np.matrix(np.ones((6040,6040)) * np.inf) #6040 is the max customer id
        

    #converts data into a more manageable form
    def preProcessing(self):
        self.data = self.train[['User_ID','Product_ID','Purchase']]
        self.data['User_ID'] = self.data['User_ID'].apply(str)
        self.data['User_ID'] = self.data['User_ID'].str.slice(start=1)
        self.data['User_ID'] = self.data['User_ID'].str.lstrip("0")
        self.data['User_ID'] = self.data['User_ID'].apply(int)
        self.data['User_ID'] = self.data['User_ID'] - 1
        print(self.data['User_ID'])

    #creates a dictionary object with users as keys, and a pandas dataframe of entries relating to that user as values
    def getIDs(self):
        self.preProcessing()
        uniqueIDs = self.data['User_ID'].unique()
        self.userdict = {elem : pd.DataFrame for elem in uniqueIDs}
        for key in self.userdict.keys():
            self.userdict[key] = self.data[1:][self.data['User_ID'] == key]

    #computes a distance metric between each customer, determined by finding the intersecting products between customers, and taking the
    #average RMS difference between them
    #this method takes a VERY long time
    def compute_distances(self):
        ival = 0
        jval = 0
        self.getIDs()
        for key in self.userdict.keys():
            print(key)
            for key2 in self.userdict.keys():
                frame = pd.merge(self.userdict[key], self.userdict[key2], how='inner', on=['Product_ID'])
                frame['diffs'] = frame['Purchase_x'] - frame['Purchase_y']
                frame['diffs'] = frame['diffs']**2
                sumvals = np.sqrt(frame['diffs'].sum())
                if (int(sumvals) == 0 and key != key2):
                    continue
                self.dist[key,key2] = sumvals
            np.save('dists.npy',self.dist) #saves distances after each iteration to ensure that data is not lost if python crashes

    #method called for each data point, predicts the purchase value for a specific user and product based on
    #the K nearest users to the current user who also bought that product
    def getNeighbors(self,keyvalue,k,ID):
        x = np.sort(self.dists[keyvalue])
        count = 0
        vals = 0
        index = 0
        for y in np.nditer(x):
            index += 1
            if y < 0.1 or y == np.inf:
                continue
            else:
                if (index,ID) in self.df.index:
                    count += 1
                    #print("vals is " + str(vals))
                    vals += float(self.train.loc[index,ID])
            if count == k:
                break
        if count == 0:
            return 0
        else:
            return vals/count

    #main KNN method
    def KNN(self,k):
        self.preProcessing()
        finalvals = np.zeros(550068)
        self.dists = np.load('dists.npy')
        self.sorteddata = self.data.sort_values('User_ID')
        self.sorteddata = self.sorteddata.set_index(['User_ID','Product_ID'])
        for i, row in self.data.iterrows():
            
            val = self.getNeighbors(row['User_ID'],k,row['Product_ID'])
            finalvals[i] = val
            np.save('knnvals.npy',finalvals)
        np.save('knnvals.npy',finalvals)

    #print(df_adjusted.columns)

#main method to call initialize objects and run regression algorithms
def main():
    sgd = SGDRegression()

    vals1 = sgd.SGDtraining(False)
    vals2 = sgd.SGDsplit(False)

    for i,val in np.ndenumerate(vals2):
        if vals2[i] == 0:
            vals2[i] = vals1[i]
    df = sgd.test[['User_ID','Product_ID']]
    df['Purchase'] = pd.Series(vals2)
    df.to_csv('values2.csv',index=False)

main()

