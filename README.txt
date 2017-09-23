*************** README file for Black Friday Project ****************

For this project I shall be examining a data set containing the purchasing behavior of customers for specific products, 
with attributes relating to both customers and products. 
The aim is to predict the amount that customers will pay for products in the test set, 
based on insights gained from the training set.
The training set consists of approximately 550,000 entries, consisting of the product ID, 
the customer ID, the purchase price the customer paid for the product, 
and additional categorical attributes such as customer age range, gender, etc.
I will be examining this data using the Python programming language, specifically utilizing the libraries NumPy, pandas, and scikit-learn.
I will employ three different analyses on this data.

The first will be using basic analysis techniques such as averaging over specific variables. 

The second will employ a scikit regression technique, stochastic gradient descent, 
to make predictions based on the product and customer attributes.

The third will be a K nearest neighbors algorithm, using a distance matrix computed from how ‘close’ 
customers are to one another based on their purchase prices for specific items in the training set.

*************** Code ****************

My first class is called basicAnalysis. This includes methods to average over the data based on
product ID alone, and based on the combination of age and product ID. There are also methods to
predict the purchase price for the training set using these techniques.

My second class is called SGDRegression. Here, I implement Stochastic Gradient Descent on the dataset
after converting all categorical variables to dummy variables. One methods uses SGD as a predictor for the
whole dataset, and another uses a separate SGD predictor for each product ID with sufficient customers
purchasing it. I also implemented a cross validation method to determine whether this technique worked
consistently well.

My third class is called KNNRegression, and employs K nearest neighbors as a predictor. First, I generate the
distance matrix between customers, based on the RMS of differences between what they paid for each overlapping
product (product they both purchased). Then, I employ KNN, using 3 neighbors currently, calling a method to find
each customer's 'neighbors' based on the precomputed distance matrix.

I use a main() method to initialize objects and call the desired methods.

*************** Results ****************

I tested each regression method on the training set, and chose the most successful one to implement on the test set.

The RMSE results for the training set are as follows:

Averaging over Product IDs: 2648.82475038
Averaging over Product IDs and Age Ranges: 2591.32738023
Stochastic Gradient Descent: 2981.54695633
K Nearest Neighbors: 3072.24347807
Split Stochastic Gradient Descent: 2440.81694048

Since the split SGD method performed the best, I implemented that on the test set, with a RMSE of 2776.55251.

*************** Conclusions ****************

As an extension to this project, I could use the distance matrix I generated as a means of clustering users, and then performing regressions on
each cluster for a possibly more accurate result.
