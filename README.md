# What's in this project?

* README what you are reading now
* DATA . the kings county housing data of houses in kings county Seattle
* final.ipynb a Jupyter notebook of the steps taken to get the results below
* presentation.pdf a PDF with a PowerPoint presentation to be shown to theoretical shareholders


# What makes a house price go up?

Which factor/factors of a houses are the biggest determining factor towards increasing it's price?
What should you look for in a house you wish tp purchase?

i assume it's bathrooms, but that's completely subjective opinion.
let's see what the data says


# inspect the data

at least *two* of the columns have null values that need to be filled with some value, *six* are objects, which need to be made into numerical columns
date is date sold at


# cleaning

### a few of the columns are objects. make them int64, or label encode them later
especially sqft_basement

`homes['sqft_basement'].unique

homes['sqft_basement'] = homes['sqft_basement'].str.replace('?', '0').astype(np.float64)

homes['sqft_basement'].unique`

waterfront, view, condition, grade, yr_built, yr_renovated, and zipcode are **categorical**. i'm not sure which ones i will use, but i think i'll use LabelEncoding to make them usable

### zipcode is categorical
but it's too many. Morgan gave me the great idea of sepperating the zip codes by county and making binary columns


### cleaning steps
i will be dropping longitude and latitude
'date' is date sold. i need two columns , one the time between year build and date sold, another the time between date renovated and date sold
1) drop 'id' 'long' and 'lat'
2) turn date into an int
3) turn yr renovated into an int
4) make comparisson columns
5) fill nan with zeros
6) turn objects into ints

`homes.drop(columns=['id', 'lat', 'long'], inplace=True)`

### convert the sell date object to a datetime object
`homes['date'] = pd.to_datetime(homes['date'])`


### make a new column of just the years the house was sold, as an integer 
`homes['sell_yr'] = homes['date'].dt.year.astype(int)`

i'm only making a column that represents the difference in years, as the data set
only has the years of when the house was build, and not the exact date

there doesn't appear to be any obvious similarities between the houses pre-sold

# fill N/A values with zeros, and convert to int, just as a precaution
`homes['yr_renovated'] = homes['yr_renovated'].fillna(0).astype(int)

homes['yr_renovated']`

### inspect categorical columns

some categories look ordinal. let's make them numbers (and int64 type)

Morgan Jones gave me the idea to split Zipcodes by inside seattle and outside, and to treat it as a boolean variable

besides for waterfront, which is binary, the rest appear to be ordinal.

view, grade and condition appear to have more points to their variables. 

condition seems like it will be most useful for our problem, as the houses we want to be dealing with will be built already. but i intend to use all three after label encoding

# sort dataframe by price, which is our dependent variable
`homes.sort_values(by='price', inplace=True)`


# Make a Simple linear regression model as a baseline

then check data for three assumptions. linearity, normality, homoscedasticity


# Make a correlation matrix (heat map)

Make a correlation matrix of the data to see which variables might have more potential correlation

`import seaborn as sns

plt.figure(figsize=(16, 8))
sns.heatmap(homes.corr(), annot=True)
plt.show()`


# Make Model of Price per Square Feet above sea level

let's make a formula of X and Y being `sqft_above`, and `price`

`form = 'price~sqft_above'

pri_sqft_model = ols(formula=form, data=homes).fit()`

# check the summery
`pri_sqft_model.summary()`


# scatter plot to check for linearity
`plt.scatter(homes['price'], homes['sqft_above'])
plt.title("Linearity check")
plt.xlabel('price')
plt.ylabel('square feet (besides basement)')
plt.show()`

# Check for normality and homoscedasticity

`homes['price'].hist()`


# Make a new DataFrame without the outliers
 
2.5M seems like an okay cuttoff

let's cut off the outliers and make a new DataFrame to use

no_outliers = homes.loc[homes['price'] < 2500000]

print(len(homes) - len(no_outliers))

102 out of 21,596 seems an okay amount to chop off


## check the DataFrame without the outliers to see if any variable has more of a linear relationship
another scatter matrix, and heatmap

`pd.plotting.scatter_matrix(no_outliers,figsize  = [20, 20]);
plt.show()

plt.figure(figsize=(16, 8))
sns.heatmap(no_outliers.corr(), annot=True)
plt.show()`



### Try a model with `yr_built` even though it's semi categorical

`plt.scatter(no_outliers['price'], no_outliers['yr_built'])
plt.title("Linearity check")
plt.xlabel('price')
plt.ylabel('Year built in')
plt.show()

formula_1 = 'price~yr_built'

model_1 = ols(formula=formula_1, data=homes).fit()

model_1.summary()`

0.03 R squared value

this model is worse. let's go back to `square feet`

Square feet of living space is the highest corellating factor to price, followed by square feet besides the basement, and amount of bathrooms.

the amount of bathrooms may be an effect of having more square feet. if it's an effect of multicolinearity, we will have to remove it


## Same model as above, with new DataFrame

`form = 'price~sqft_above'

price_sqft_model = ols(formula=form, data=no_outliers).fit()

price_sqft_model.summary()`

this model only accounts for 35% of the data, worse than 36, but not that bad

let's move on to multi linear regression

# Label encoding categorical variables

going to label encode the columns in place, not OHE. the below code is based on examples that Morgan gave me

`from sklearn.preprocessing import LabelEncoder
laibel = LabelEncoder()

no_outliers['zipcode'] = laibel.fit_transform(no_outliers['zipcode'])
no_outliers['view'] = laibel.fit_transform(no_outliers['view'])
no_outliers['condition'] = laibel.fit_transform(no_outliers['condition'])
no_outliers['bathrooms'] = laibel.fit_transform(no_outliers['bathrooms'])
no_outliers['bedrooms'] = laibel.fit_transform(no_outliers['bedrooms'])
no_outliers['floors'] = laibel.fit_transform(no_outliers['floors'])
no_outliers['waterfront'] = laibel.fit_transform(no_outliers['waterfront'])

no_outliers.head()`

# drop columns i don't intend to use
`no_outliers.drop(['date'], axis=1, inplace=True)`


# Make multi variable regression model
`y_var = 'price'
x_vars = no_outliers.drop('price', axis=1)
all_columns = '+'.join(x_vars.columns)
multi_formula_1 = y_var + '~' + all_columns

model_ver_1 = ols(formula=multi_formula_1, data=no_outliers).fit()
model_ver_1.summary()`

## 'floors' variable has a high P value, dropping it

`no_outliers.drop(['floors'], axis=1, inplace=True)

model_ver_1.fvalue`

this model acounts for %67 percent of the data though

maybe i should do some log transformations 

# get MAE to see how much error is in our model
`y_predic = model_ver_1.resid
y = homes['price']
mae_resid = np.mean(np.abs(y - y_predic))
mae_resid`

# and RMSE because i intend to make another model
since at least one variable has a P value that is too high
and several coeficients are very negative 

`model_ver_1.mse_resid

rmse_residuals = np.sqrt(model_ver_1.mse_resid)
rmse_residuals

print(rmse_residuals - mae_resid)

resids = model_ver_1.resid

sns.scatterplot(y_predic,resids)

sns.distplot(resids,kde=True)

from scipy import stats

fig = sm.graphics.qqplot(resids, dist=stats.norm, line='45', fit=True)
fig.show()`

## check for autocorrelation

using a method i got from Morgan

`from statsmodels.stats.stattools import durbin_watson

durbin_watson(resids)`

# log transformations and scaling

everything looks pretty normal. not sure if log transformations are neccessary. feature sczaling is though, most of the variables with high coeficients have a vastly different scale from our dedpendent variable

`from sklearn import preprocessing

standard_vars = preprocessing.StandardScaler().fit_transform(no_outliers)`


# Make a new model with the scaled variables

`x_vars = homes_2.drop('price', axis=1)
all_columns = '+'.join(x_vars.columns)
formula_2 = y_var + '~' + all_columns
multi_formula_1 = y_var + '~' + all_columns
x_vars_int2 = sm.add_constant(x_vars)
model_ver_2 = sms.OLS(homes_2['price'], x_vars_int2).fit()
model_ver_2.summary()`

### coeficients are scaled, a bit better R squared, and better for prediction with test-train ing



## MUCH better
the `model_ver_2` model accounts for 67% of the data. not perfect, but good enough to progress

# Use scikit-learn now for the 'split_train_test' function

`from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
predictors = homes_2.drop('price', axis=1)
y = homes_2['price']
linreg.fit(predictors, y)
linreg.coef_`



# Drop variables that have decreasing coeficients, and see what our R squared value is

predictors2 = homes_2.drop(columns=['view','yr_built','sqft_lot15'])
model_ver_3 = sms.OLS(y, predictors2).fit()
model_ver_3.summary()

when i drop some variables with negative coeficients, others that were positive, are now negative

and the R squared value is perfect, an impossibility.

i will go back to using the second model

for kicks, let's just drop `bathrooms` and see what we get


# fit and transform (and make test and train data sets)


`from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sc = StandardScaler()`


i will use the train and test data to gradually add features to the base model to see if any improvement has occured

`f_t_pred = sc.fit_transform(predictors)
X_train, X_test, y_train, y_test = train_test_split(f_t_pred, y,random_state = 0,test_size=0.20)`

### get R squared score of training data

`from sklearn.metrics import r2_score


split_regr = LinearRegression().fit(X_train,y_train)
y_split_pred = split_regr.predict(X_train)
r2_score(y_true=y_train,y_pred=y_split_pred)`


### this model is okay

get y hat value to get the Mean Squared Error

`from sklearn.metrics import mean_squared_error`

## function to get mse of model

`def get_mse(X_t, X_te, y_t, y_te):
    split_regr = LinearRegression().fit(X_t,y_t)
    y_hat_train = split_regr.predict(X_t)
    y_hat_test = split_regr.predict(X_te)

    train_mse = mean_squared_error(y_t, y_hat_train)
    test_mse = mean_squared_error(y_te, y_hat_test)
    print('train ',train_mse, ' test ', test_mse)
get_mse(X_train, X_test, y_train, y_test)`

# check for multicolinearity, and homoscedacity


make sure the R squared values are good, and continually check if the model we are using is producing better results than before

***BUT*** beware of over fitting, multicolinearity, confounding variables

## check for Homoscedasticity 

`residuals = y_train.values-y_split_pred

mixplot = sns.scatterplot(x=y_split_pred,y=residuals)
mixplot = sns.lineplot([-2,5],[0,0],color='green')`


## and normality

`sns.distplot(residuals,kde=True)`


###  now to check for multicolinearity

`plt.figure(figsize=(10,10))
sns.lineplot(y=y_split_pred,x=residuals,marker='x',color='orange')

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

y_hat_train = split_regr.predict(X_train)
y_hat_test = split_regr.predict(X_test)

print('Training MAE', mean_absolute_error(y_train, y_hat_train))
print('Testing MAE', mean_absolute_error(y_test, y_hat_test))`

# Takeaway:

does it answer our business question? how does it answer it?

it seems that the testing error is higher than the training error. this means that our model is useful.

the coeficients for `sqft_living` , `sqft_lot` and `grade` being the highest, i would recomend a potential investor looking for
a house in the seatle area that has a higher square footage of the lot and living room, if possible near the water.

for every increase in grade, a %39 increase in price
for every increase in square feet of living room, or general lot, a %20 increase each in price
if the house is on the waterfront, expect a %10 increase in price
and if the house is within the seatle area, expect a %20 increase in price
additionally, if neighboring houses have larger living rooms, you can expect a %17 increase in price

strangely, having more bathrooms consistently had correlation with a higher price. as a NYer, i am biased to think this is an important factor, but objectively, it is most probably due to larger, more expensive houses with more rooms, naturally having more bathrooms
