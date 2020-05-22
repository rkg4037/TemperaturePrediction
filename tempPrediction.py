import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, median_absolute_error,mean_squared_error



'''
    Given data contains date wise details of meantemp ,mintemp,maxtemp, and 
    {mindewm,maxdewm,minhumidity,maxhumidity,etc} of last 3 days stored date wise
'''
df = pd.read_csv("A:\pythonPractice\ML\Databases\weather.csv") ;
#df.info()
df.set_index('date',inplace=True)
#print(df.columns)
#print(df.head(10))
print("Correlation of other feature with mintempm\n") ;
print(df.corr()[['meantempm']].sort_values('meantempm')) ;
print("\n\n") ;
# correlation will tell how much dependency they actually have with meantempm
# we consider all values with correlation more than 0.8 to determine the data
# we will also not consider mintempm and maxtempm to determine meantemp as they will be obviously very close to mean

testSize = 0.40 ;
# predictors were chosen on the basis of which paramneters are highly correlated to meantempm (correlation > 0.8)
predictors = ['meantempm_1',  'meantempm_2',  'meantempm_3',
              'mintempm_1',   'mintempm_2',   'mintempm_3',
              'meandewptm_1', 'meandewptm_2', 'meandewptm_3',
              'maxdewptm_1',  'maxdewptm_2',  'maxdewptm_3',
              'mindewptm_1',  'mindewptm_2',  'mindewptm_3',
              'maxtempm_1',   'maxtempm_2',   'maxtempm_3']

df2 = df[['meantempm'] + predictors] ;

X = df2[predictors] ;
y = df2['meantempm'] ;

'''
for feature in predictors:
    plt.title("meantempm vs "+feature) ;
    plt.xlabel(feature) ;
    plt.ylabel('meantempm') ;
    plt.scatter(X[feature],y,color='r') ;
    plt.show()
'''
regr = LinearRegression() ;

#Linear Regression Model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=testSize) ;
regr.fit(X_train,y_train) ;

prediction = regr.predict(X_test) ;
'''
for feature in predictors:
    plt.title("meantempm vs "+feature+" on test data-set") ;
    plt.xlabel(feature) ;
    plt.ylabel('meantempm') ;
    #  actual plot for test data set
    plt.scatter(X_test[feature],y_test,color='red',label="actual") ;
    # plot of predicted values for test data set
    plt.scatter(X_test[feature],prediction,color='black',label="predicted") ;
    plt.show()
'''
print("Errors for Linear Regression") ;
print("The Mean Squared Error: %.2f degrees celsius" % mean_squared_error(y_test, prediction)) ;
print("The Variance: %.2f" % regr.score(X_test, y_test)) ;
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction)) ;
print("The Median Absolute Error: %.2f degrees celsius\n\n" % median_absolute_error(y_test, prediction)) ;


#NonLinear Regression (of order 2)
poly = PolynomialFeatures(degree=2) ;
X_poly = poly.fit_transform(X) ;        #total 190 new features created from 18 features

#print(X_poly.shape," ",y.size)

X_poly_train,X_poly_test,y_poly_train,y_poly_test = train_test_split(X_poly,y,test_size=testSize) ;
#print(X_train.shape," ",X_test.shape)

poly.fit(X_poly,y) ;
regr.fit(X_poly_train,y_poly_train) ;
quadPrediction = regr.predict(X_poly_test) ;
#print(quadPrediction.shape) ;
X_train1 = X_poly_train.transpose() ;
X_test1 = X_poly_test.transpose() ;

# plotting some of the comboination of feature against meantempm
# red are the actual values
# blue are predicted by models
for j in range(6) :
    i = np.random.randint(0,189) ;
    plt.ylabel('meantempm') ;
    plt.xlabel('Any combination of feature of order 2')
    plt.scatter(X_test1[i],y_poly_test,color='red',label='actual') ;
    plt.scatter(X_test1[i],quadPrediction,color='blue',label='predicted') ;
    plt.show()

print("Errors for NonLinear Regression of order 2") ;
print("The Mean Squared Error: %.2f degrees celsius" % mean_squared_error(y_poly_test, quadPrediction)) ;
print("The Variance: %.2f" % regr.score(X_poly_test, y_poly_test)) ;
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_poly_test, quadPrediction)) ;
print("The Median Absolute Error: %.2f degrees celsius\n\n" % median_absolute_error(y_poly_test, quadPrediction)) ;


#NonLinear Regression (of order 3)
poly = PolynomialFeatures(degree=3) ;
X_poly = poly.fit_transform(X) ;        #total 1330 features are created from different combinatons of 18 features

#print(X_poly.shape," ",y.size)

X_poly_train,X_poly_test,y_poly_train,y_poly_test = train_test_split(X_poly,y,test_size=testSize) ;
#print(X_train.shape," ",X_test.shape)

poly.fit(X_poly,y) ;
regr.fit(X_poly_train,y_poly_train) ;
cubePrediction = regr.predict(X_poly_test) ;
#print(cubePrediction.shape) ;
X_train1 = X_poly_train.transpose() ;
X_test1 = X_poly_test.transpose() ;

# plotting some of the comboination of feature against meantempm
# red are the actual values
# green are predicted by models
for j in range(8) :
    i = np.random.randint(0,1329) ;
    plt.ylabel('meantempm') ;
    plt.xlabel('Any combination of feature of order 3')
    plt.scatter(X_test1[i],y_poly_test,color='red',label='actual') ;
    plt.scatter(X_test1[i],quadPrediction,color='green',label='predicted') ;
    plt.show()


print("Errors for NonLinear Regression of order 3") ;
print("The Mean Squared Error: %.2f degrees celsius" % mean_squared_error(y_poly_test, cubePrediction)) ;
print("The Variance: %.2f" % regr.score(X_poly_test, y_poly_test)) ;
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_poly_test, cubePrediction)) ;
print("The Median Absolute Error: %.2f degrees celsius\n\n" % median_absolute_error(y_poly_test, cubePrediction)) ;
