Statistical Learning with R
========================================================
author: Guy Freeman <guy@dataguru.hk>
date: Monday 20th October 2014

Source: rpubs.com/slygent/statlearn

Statistical learning with R
========================================================
width: 3200
height: 1800


- Why learning?
- Why statistical learning?
- Why R?
- Why me?
 - I use R to carry out statistical learning almost every day.

![](knn1.png)

R, the statistical programming language
========================================================

I know that Mart has been using Python with you, but I want to show you an alternative.

I've been using R for 15 years now, and I want to show you the advantages (and maybe a few disadvantages) of using a programming language designed purely for statistical analysis. Then you can use different tools for different situations.

The New York Times approves of R! http://www.nytimes.com/2009/01/07/technology/business-computing/07program.html

And most recently, Bayesian statistics, the foundation of statistical learning: http://www.nytimes.com/2014/09/30/science/the-odds-continually-updated.html

Statistical learning -- it's like machine learning, but better
========================================================

These are my subjective definitions:

*Machine learning* is running algorithms to understand patterns in data and make predictions from inputs.

_Statistical learning_ is machine learning where you understand why the algorithms work and when they work.

I highly recommend the book "An Introduction to Statistical Learning: With Applications in R" by Gareth James et al for an overview. I heavily leaned on this book for today's lesson :)


An Introduction to Statistical Learning: With Applications in R
====
![](cover.jpg)

Introduction to R
===

"R is a language and environment for statistical computing and graphics" -- the official description of R at its homepage www.r-project.org

It also happens to be Free Software (just like Python). You can use it without paying money, and you can copy it and modify it and distribute it.

![](Rlogo.jpg)

You can download it from http://cran.r-project.org/

I highly recommend you also install RStudio, an Integrated Development Environment (IDE) for R that works on Windows, Mac and Linux, from http://www.rstudio.com/products/rstudio/download/

Functions in R
===

R uses *functions* to carry out procedures on (optional) inputs (aka *arguments*).

For example, the `sum` function calculates the sum of its inputs (so original!).


```r
sum(3,5,500,pi)
```

```
[1] 511.1416
```

Vectors in R
===

To create a vector of numbers we use the `c` function [short for "concatenate"]. We can save this vector to a variable using `<-`.


```r
x <- c(1,8,7,15)
x
```

```
[1]  1  8  7 15
```

Many functions in R are "vectorized", which means they know what to do with vectors. Adding two vectors, for example, works how you'd expect:


```r
y <- c(20,8,4,1)
y + x
```

```
[1] 21 16 11 16
```

Objects in R
===

What objects have we created so far? Use `ls()` to find out:


```r
ls()
```

```
[1] "x" "y"
```

Delete anything you don't want any more with `rm()`:


```r
rm(y)
ls()
```

```
[1] "x"
```

Matrices in R (with some help)
===

The `matrix` function in R creates matrices. How? Let's use R's help system.


```r
?matrix
matrix(data=c(1,2,3,4), nrow=2, ncol=2)
```

```
     [,1] [,2]
[1,]    1    3
[2,]    2    4
```

```r
matrix(c(1,2,3,4), 2, 2)
```

```
     [,1] [,2]
[1,]    1    3
[2,]    2    4
```

Random variables in R
===

We can simulate a sample from common probability distributions and calculate summary statistics easily.


```r
set.seed(3) # to make sure our random numbers are the same!
y <- rnorm(100) # 100 random normal variables with mean 0 and s.d. 1
```

```r
mean(y)
```

```
[1] 0.01103557
```

```r
var(y)
```

```
[1] 0.7328675
```

Random variables in R
===
Standard deviation is the square root of the variance.

```r
sqrt(var(y))
```

```
[1] 0.8560768
```

```r
sd(y)
```

```
[1] 0.8560768
```

Graphics in R
===


```r
x=rnorm(100)
y=rnorm(100)
plot(x,y)
```

<img src="gapres-figure/unnamed-chunk-11-1.png" title="plot of chunk unnamed-chunk-11" alt="plot of chunk unnamed-chunk-11" height="550px" style="display: block; margin: auto;" />

Annotations in R
===

```r
plot(x,y,xlab="this is the x-axis",ylab="this is the y-axis",
main="Plot of X vs Y")
```

<img src="gapres-figure/unnamed-chunk-12-1.png" title="plot of chunk unnamed-chunk-12" alt="plot of chunk unnamed-chunk-12" height="550px" style="display: block; margin: auto;" />

Contour plots
=== 

```r
x <- seq(-pi,pi,length=50) # from -pi to pi in 50 steps
y <- x
f <- outer(x,y,function (x,y)cos(y)/(1+x^2)) # 
contour(x,y,f)
```

<img src="gapres-figure/unnamed-chunk-13-1.png" title="plot of chunk unnamed-chunk-13" alt="plot of chunk unnamed-chunk-13" height="450px" style="display: block; margin: auto;" />

Indexing data
===

```r
A <- matrix(1:16,4,4)
A
```

```
     [,1] [,2] [,3] [,4]
[1,]    1    5    9   13
[2,]    2    6   10   14
[3,]    3    7   11   15
[4,]    4    8   12   16
```

```r
A[2,3] # 2nd row, 3rd column
```

```
[1] 10
```

```r
A[c(1,3),c(2,4)]
```

```
     [,1] [,2]
[1,]    5   13
[2,]    7   15
```

Indexing data
===

```r
A[,2:1] # 2nd then 1st columns, all rows
```

```
     [,1] [,2]
[1,]    5    1
[2,]    6    2
[3,]    7    3
[4,]    8    4
```

```r
A[-c(1,3),] # ignore 1st and 3rd rows
```

```
     [,1] [,2] [,3] [,4]
[1,]    2    6   10   14
[2,]    4    8   12   16
```

Loading data into data frame
===

```r
auto <- read.csv("http://www-bcf.usc.edu/~gareth/ISL/Auto.csv")
colnames(auto)
```

```
[1] "mpg"          "cylinders"    "displacement" "horsepower"  
[5] "weight"       "acceleration" "year"         "origin"      
[9] "name"        
```

```r
head(auto$mpg) # access column
```

```
[1] 18 15 18 16 17 15
```

Exercises (to make sure you're awake)
===
1. Use `read.csv` to store the data from http://www-bcf.usc.edu/~gareth/ISL/College.csv
2. Look at the first six rows using the `head` function.
3. What's going on with the first column? Use it to name the rows of the data using `row.names`.
4. Now delete the redundant first column.
5. Use the `summary` function on the data frame.
6. Use the `pairs` function on the first ten rows of the data frame.
7. Use the `hist` function to produce some histograms of the data. Try different bin widths.

Now we can start statistical learning!
===

Statistical learning means using $input$ variables to predict $output$ variables. That's it. But that's easier said than done...


```r
Advertising <- read.csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv")
head(Advertising, n=3)
```

```
  X    TV Radio Newspaper Sales
1 1 230.1  37.8      69.2  22.1
2 2  44.5  39.3      45.1  10.4
3 3  17.2  45.9      69.3   9.3
```

How can we best predict Sales from the TV, Radio and Newspaper advertising budgets? (Why would we want to?)

Statistical learning in one equation
===

Let $Y$ be the output variable (e.g. sales), and $X$ the vector of input variables $X_1, X_2, X_3, ...$ 

Then

$$Y = f(X) + \epsilon $$

*We want to work out what $f$ is*. $\epsilon$ is unavoidable noise that is independent of $X$.

How do we *estimate* $f$ from the data, and how do we evaluate our estimate? That is statistical learning.

Prediction and its limits
===

Once we have an estimate $\hat{f}$ for $f$, we can predict unavailable values of $Y$ for known values of $X$: $$ \hat{Y} = \hat{f}(X) $$

How good an estimate of $Y$ is $\hat{Y}$? The difference between the two values can be partitioned into *reducible* and *irreducible* errors:

$$ E(Y - \hat{Y})^2 = [f(X) - \hat{f}(X)]^2 + \sigma_\epsilon^2 $$

where $[f(X) - \hat{f}(X)]^2$ is the reducible error.

How to estimate f
===

Two main approaches:

1) Parametric

   An assumption is made about the form of $f$. For example, the $linear$ model states that $$\hat{f}(X) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p $$ Then we use the *training data* to choose the values of $\beta_0, \beta_1, ..., \beta_p$, the *parameters*.
   
   Advantage: Much easier to estimate parameters than whole function.
   
   Disadvantage: Our choice of the form of $f$ might be wrong, or even very wrong.
   
How to estimate f
===   
   
   We can try to make our parametric form more *flexible* in order to reduce the risk of choosing the wrong $f$, but this also makes $\hat{f}$ more complex and potentially following the noise too closely, thereby *overfitting*.
   
2) Non-parametric
   
   Just get $f$ as close as possible to the data points, subject to not being too wiggly or too unsmooth. 
   
   Advantage: More likely to get $f$ right, especially if $f$ is weird.
   
   Disadvantage: Far more data is needed to obtain a good estimate for $f$.
   
Supervised vs unsupervised learning
===

What if we are only given input variables and no outputs? Then our learning will be *unsupervised*; we are blind. 

What can we do? We can try to understand the relationship between the input variables or between the observations. One example is to *cluster* observations or variables together into groups.

We will focus on *supervised* learning with outputs today.

Regression vs classification
===

Variables can be either *quantitative* or *qualitative*. (Anyone who went to my Introduction to Data Science lecture will remember I am obsessed with this distinction). 

We care about this because different variable types affect the class of values we can make predictions from and therefore the possible nature of $f$, and also how to measure the size of an error from a wrong prediction.

When the output variable is quantitative, prediction is called *regression*.

When the output variable is qualitative, prediction is *classification*.

Assessing model accuracy
===

How good is our $\hat{f}$? This is equivalent to asking how close each $\hat{f}(x)$ was to $y$. The most commonly-used score for regression is the *Mean Squared Error* (MSE): $$ MSE = \frac{1}{n} \sum{(y_i - \hat{f}(x_i))^2} $$

i.e. just sum up the squares of the differences between the predictions $\hat{f}(x_i)$ and the actual outputs $y_i$.

But actually we are not very interested in how well $\hat{f}$ predicts $y$ we already know; we want it to predict future $y$ we haven't seen yet.

But we haven't seen the future $y$ yet...

Training MSE vs test MSE
===

If we only try to minimize the MSE for old (*training*) data, we might *overfit* and have terrible MSE for new (*test*) data.

![](mse.png)

Increasing flexibilty leads to more accuracy, but after a while we are predicting training data TOO accurately!

Bias vs variance
===

Why does the test MSE go down and then up? Because

$$ E(y - \hat{f}(x))^2 = Var(\hat{f}(x)) + [Bias(\hat{f}(x))]^2 + Var(\epsilon) $$

What is the *variance* of $\hat{f}$? It is the amount $\hat{f}(x)$ would change if we had different data to train with from the same source population. Ideally, $\hat{f}$ shouldn't change for different data that is fundamentally the same. **More flexible models have higher variance.**

The *bias* of $\hat{f}$ is the error due to using $\hat{f}$ instead of the true (but unknown) $f$. **Simpler models have higher bias.**

**We are trying to decrease the bias without increasing the variance more**.

Exercises
===
For which of these situations would a flexible model/method be better than an inflexible one?

1. Large sample size with a small number of predictors.
2. Small sample size with a large number of predictors.
3. The relationship between the input and output variables is highly non-linear.
4. The variance of the irreducible error is very high.

Linear regression
===

Back to `Advertising` at last. Are the various media budgets *linearly* associated with sales?

```r
library(ggplot2)
library(reshape2)
advmelt <- melt(Advertising, id.vars = c("X", "Sales"))
```
Hmm.
***

```r
ggplot(advmelt, aes(x = value, y = Sales)) + geom_point() + facet_wrap(~ variable, scales="free_x")
```

<img src="gapres-figure/unnamed-chunk-19-1.png" title="plot of chunk unnamed-chunk-19" alt="plot of chunk unnamed-chunk-19" style="display: block; margin: auto;" />

Linear regression in R
===

```r
summary(lm(Sales ~ TV + Radio + Newspaper, data = Advertising))
```

```

Call:
lm(formula = Sales ~ TV + Radio + Newspaper, data = Advertising)

Residuals:
    Min      1Q  Median      3Q     Max 
-8.8277 -0.8908  0.2418  1.1893  2.8292 

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  2.938889   0.311908   9.422   <2e-16 ***
TV           0.045765   0.001395  32.809   <2e-16 ***
Radio        0.188530   0.008611  21.893   <2e-16 ***
Newspaper   -0.001037   0.005871  -0.177     0.86    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 1.686 on 196 degrees of freedom
Multiple R-squared:  0.8972,	Adjusted R-squared:  0.8956 
F-statistic: 570.3 on 3 and 196 DF,  p-value: < 2.2e-16
```

Linear regression
===
Why does newspaper advertising seem to have no effect on sales, despite the graph earlier seeming to show otherwise? 

Consider the relationship between newspaper and radio budgets:
<img src="gapres-figure/unnamed-chunk-21-1.png" title="plot of chunk unnamed-chunk-21" alt="plot of chunk unnamed-chunk-21" height="350px" style="display: block; margin: auto;" />
Higher newspaper and radio budgets go together. So even if only radio affects sales, there will still be a (spurious) association between newspaper advertising budgets and sales.

Model fit
===

How good is our linear model? One way to ask this: How well does our model fit the data? 

One measure is $R^2$, the square of the correlation between the predicted and actual sales values. For this model that value is 0.8972. Excluding newspapers, $R^2$ is... 0.8972. This suggests newspaper budgets are indeed unnecessary for predicting sales once TV and radio are taken into account.

Linear in reality?
===
But how linear is the data? Below is a graph of the prediction errors for different values of TV advertising.
<img src="gapres-figure/unnamed-chunk-22-1.png" title="plot of chunk unnamed-chunk-22" alt="plot of chunk unnamed-chunk-22" height="350px" style="display: block; margin: auto;" />
It can be seen that for low and high values of TV advertising, the model over-predicts sales, while underestimating in the middle. This systematic error indicates the true relationship is not linear.

Extending the linear model
===
One way to extend the linear model is to include *interactions*.

$$ \hat{Y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_{12} X_1 X_2... $$

Now the effect of $X_1$ on $\hat{Y}$ depends on the value of $X_2$, and vice versa.

Is there an interaction between TV and Radio advertising budgets? It turns out there is.

Interactions in R
===
<small>

```

Call:
lm(formula = Sales ~ TV + Radio + TV:Radio)

Residuals:
    Min      1Q  Median      3Q     Max 
-6.3366 -0.4028  0.1831  0.5948  1.5246 

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept) 6.750e+00  2.479e-01  27.233   <2e-16 ***
TV          1.910e-02  1.504e-03  12.699   <2e-16 ***
Radio       2.886e-02  8.905e-03   3.241   0.0014 ** 
TV:Radio    1.086e-03  5.242e-05  20.727   <2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.9435 on 196 degrees of freedom
Multiple R-squared:  0.9678,	Adjusted R-squared:  0.9673 
F-statistic:  1963 on 3 and 196 DF,  p-value: < 2.2e-16
```
</small>

Non-linear linear models
===
I mean *polynomial regression*, e.g. 

$$ \hat{Y} = \beta_0 + \beta_1 X_1 + \beta_{2} X_1^2 + ... $$
<small>

```

Call:
lm(formula = Sales ~ TV + I(TV^2))

Residuals:
    Min      1Q  Median      3Q     Max 
-7.6844 -1.7843 -0.1562  2.0088  7.5097 

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  6.114e+00  6.592e-01   9.275  < 2e-16 ***
TV           6.727e-02  1.059e-02   6.349 1.46e-09 ***
I(TV^2)     -6.847e-05  3.558e-05  -1.924   0.0557 .  
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 3.237 on 197 degrees of freedom
Multiple R-squared:  0.619,	Adjusted R-squared:  0.6152 
F-statistic: 160.1 on 2 and 197 DF,  p-value: < 2.2e-16
```
</small>

Example
===

```r
library(MASS)
library(ISLR)
# fix(Boston)
names(Boston)
```

```
 [1] "crim"    "zn"      "indus"   "chas"    "nox"     "rm"      "age"    
 [8] "dis"     "rad"     "tax"     "ptratio" "black"   "lstat"   "medv"   
```

Example
===

```r
bostonlm <- lm(medv~lstat,data=Boston) # median home value vs low economic status
plot(Boston$lstat, Boston$medv)
abline(bostonlm)
```

<img src="gapres-figure/unnamed-chunk-26-1.png" title="plot of chunk unnamed-chunk-26" alt="plot of chunk unnamed-chunk-26" height="450px" style="display: block; margin: auto;" />
Is this linear?

Example
===
Let's look at the behaviour of the _residuals_ (prediction errors):

```r
plot(predict(bostonlm), residuals(bostonlm))
```

<img src="gapres-figure/unnamed-chunk-27-1.png" title="plot of chunk unnamed-chunk-27" alt="plot of chunk unnamed-chunk-27" style="display: block; margin: auto;" />

Example -- multiple linear regression
===
<small>

```r
summary(lm(medv~lstat+age,data=Boston))
```

```

Call:
lm(formula = medv ~ lstat + age, data = Boston)

Residuals:
    Min      1Q  Median      3Q     Max 
-15.981  -3.978  -1.283   1.968  23.158 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 33.22276    0.73085  45.458  < 2e-16 ***
lstat       -1.03207    0.04819 -21.416  < 2e-16 ***
age          0.03454    0.01223   2.826  0.00491 ** 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 6.173 on 503 degrees of freedom
Multiple R-squared:  0.5513,	Adjusted R-squared:  0.5495 
F-statistic:   309 on 2 and 503 DF,  p-value: < 2.2e-16
```
</small>

Example -- one for all....
===

```

Call:
lm(formula = medv ~ ., data = Boston)

Residuals:
    Min      1Q  Median      3Q     Max 
-15.595  -2.730  -0.518   1.777  26.199 

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  3.646e+01  5.103e+00   7.144 3.28e-12 ***
crim        -1.080e-01  3.286e-02  -3.287 0.001087 ** 
zn           4.642e-02  1.373e-02   3.382 0.000778 ***
indus        2.056e-02  6.150e-02   0.334 0.738288    
chas         2.687e+00  8.616e-01   3.118 0.001925 ** 
nox         -1.777e+01  3.820e+00  -4.651 4.25e-06 ***
rm           3.810e+00  4.179e-01   9.116  < 2e-16 ***
age          6.922e-04  1.321e-02   0.052 0.958229    
dis         -1.476e+00  1.995e-01  -7.398 6.01e-13 ***
rad          3.060e-01  6.635e-02   4.613 5.07e-06 ***
tax         -1.233e-02  3.760e-03  -3.280 0.001112 ** 
ptratio     -9.527e-01  1.308e-01  -7.283 1.31e-12 ***
black        9.312e-03  2.686e-03   3.467 0.000573 ***
lstat       -5.248e-01  5.072e-02 -10.347  < 2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 4.745 on 492 degrees of freedom
Multiple R-squared:  0.7406,	Adjusted R-squared:  0.7338 
F-statistic: 108.1 on 13 and 492 DF,  p-value: < 2.2e-16
```

Example -- non-linear transformation
===

```

Call:
lm(formula = medv ~ lstat + I(lstat^2), data = Boston)

Residuals:
     Min       1Q   Median       3Q      Max 
-15.2834  -3.8313  -0.5295   2.3095  25.4148 

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept) 42.862007   0.872084   49.15   <2e-16 ***
lstat       -2.332821   0.123803  -18.84   <2e-16 ***
I(lstat^2)   0.043547   0.003745   11.63   <2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 5.524 on 503 degrees of freedom
Multiple R-squared:  0.6407,	Adjusted R-squared:  0.6393 
F-statistic: 448.5 on 2 and 503 DF,  p-value: < 2.2e-16
```


Exercises
===
Using the `Auto` dataset:

1. Carry out linear regression of `mpg` against `horsepower`. Use `summary` to print the results.
  - Is there a relationship?
  - How strong is the relationship?
  - Is the relationship positive?
  - What is the predicted `mpg` associated with a `horsepower` of 98?

2. Plot the response against the predictor, and the line of best fit with `abline`.

Exercises
===
3. Use the `lm` function to regress `mpg` against all other variables (except `name`).
  - Is there a relationship?
  - Which predictors are statistically significant?
  - What does the coefficient for `year` suggest?
  
4. Use * and : to fit regression models with interactions. Are the interactions statistically significant?

Classification
===
Linear regression was a fine first step for quantitative responses. What about qualitative (categorical) responses?

We don't want to make predictions on a continuous scale, so linear regression can't be used. But it turns out we can predict probabilities of belonging to a class (i.e. numbers continuous between 0 and 1) and then classifying with that...

Motivating example
===
Can we predict if someone will default on their credit card payment on the basis of their annual income and credit card balance? Let's find out using the `Default` dataset in the `ISLR` package.

```r
library(ISLR)
ggplot(Default, aes(x = balance, y = income)) + geom_point(aes(colour = default), shape = 3)
```

Motivating example
===
Can we predict if someone will default on their credit card payment on the basis of their annual income and credit card balance? Let's find out using the `Default` dataset in the `ISLR` package.
<img src="gapres-figure/unnamed-chunk-32-1.png" title="plot of chunk unnamed-chunk-32" alt="plot of chunk unnamed-chunk-32" style="display: block; margin: auto;" />

Why not linear regression?
===
If output is "default" and "not default", how can we run regression?

You could argue we can code response as 0 and 1. But then order of coding and scale of coding are both arbitrary. For two classes situation is not too bad, but...

... still predicting a continuous value. We can use a threshold to classify the continuous prediction, but with linear regression this prediction could fall outside [0,1] range. How do we restrict to [0,1]?

Logistic regression
===

![](log.png)

Rather than modelling default directly, we can model the *probability* of default $Pr(Y = 1|X) = p(X)$. How?

Logistic regression
===
Instead of 

$$p(X) = \beta_0 + \beta_1 X $$

try

$$ p(X) = \frac{\exp(\beta_0 + \beta_1 X)}{1 + \exp(\beta_0 + \beta_1 X)} $$

This will ensure the prediction is between 0 and 1. Re-arranging:

$$ \frac{p(X)}{1 - p(X)} = \beta_0 + \beta_1 X $$

Linear again! But in a transformed variable.

Logistic regression
===

```r
summary(glm(default~balance,family = binomial,data=Default))
```

```

Call:
glm(formula = default ~ balance, family = binomial, data = Default)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.2697  -0.1465  -0.0589  -0.0221   3.7589  

Coefficients:
              Estimate Std. Error z value Pr(>|z|)    
(Intercept) -1.065e+01  3.612e-01  -29.49   <2e-16 ***
balance      5.499e-03  2.204e-04   24.95   <2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 2920.6  on 9999  degrees of freedom
Residual deviance: 1596.5  on 9998  degrees of freedom
AIC: 1600.5

Number of Fisher Scoring iterations: 8
```

Logistic regression
===

```r
summary(glm(default~balance+income+student,family = binomial,data=Default))
```

```

Call:
glm(formula = default ~ balance + income + student, family = binomial, 
    data = Default)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.4691  -0.1418  -0.0557  -0.0203   3.7383  

Coefficients:
              Estimate Std. Error z value Pr(>|z|)    
(Intercept) -1.087e+01  4.923e-01 -22.080  < 2e-16 ***
balance      5.737e-03  2.319e-04  24.738  < 2e-16 ***
income       3.033e-06  8.203e-06   0.370  0.71152    
studentYes  -6.468e-01  2.363e-01  -2.738  0.00619 ** 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 2920.6  on 9999  degrees of freedom
Residual deviance: 1571.5  on 9996  degrees of freedom
AIC: 1579.5

Number of Fisher Scoring iterations: 8
```

Model goodness
===
A binary classifier can make two errors: false positives and false negatives. Predicting using the model in the last slide and a threshold p(X) of 0.5, the *confusion matrix* for the `Default` is as follows:

```
              Predicted default
Actual default FALSE TRUE
           No   9627   40
           Yes   228  105
```
So although overall error rate is low, the false negative error rate (those we mistakenly said won't default) is 228/(228+105) = 0.68. That's not the best error rate ever... But the false positive rate is only 40/(9627+40) = 0.004...

ROC curves
===
Why use 0.5 as the threshold? Would another value be better? What if we choose 0.2?

```
              Predicted default
Actual default FALSE TRUE
           No   9390  277
           Yes   130  203
```
Now the false negative error rate is 0.39, although the false positive rate has increased to 0.03. Changing the threshold always results in this trade-off.

ROC curves
===
What are the error rates across thresholds? The *ROC curve* can show this.

```r
#out.height="300px"}
library(ROCR)
pred <- prediction(Default$preds, Default$default)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))
```

<img src="gapres-figure/unnamed-chunk-37-1.png" title="plot of chunk unnamed-chunk-37" alt="plot of chunk unnamed-chunk-37" style="display: block; margin: auto;" />

ROC curves
===
<img src="gapres-figure/unnamed-chunk-38-1.png" title="plot of chunk unnamed-chunk-38" alt="plot of chunk unnamed-chunk-38" style="display: block; margin: auto;" />
As we want the curve to touch the top-left hand corner, we can use the *area under the curve* (AUC) to determine the quality of the classifier. This time the AUC is 0.95.

Exercises
===
Use the `Weekly` data from the `ISLR` package.

1. Produce some summaries of the data. Are there any patterns?
2. Fit a logistic regression model to explain `Direction` using the `Lag`s and `Volume`. Which factors are significant?
3. Make a confusion matrix for this model using `table`.
4. Fit a logistic model and a KNN model with K=1 with only `Lag2` as a predictor. (Check out `knn` function). Which model is better?

Advanced R features -- data.table
===

(Inspired by http://adv-r.had.co.nz/)

It turns out that `data.frame`s, the base form of data structure in R, have an annoying syntax. Enter `data.table`s.

```r
library(data.table)
defaultdt <- as.data.table(Default)
defaultdt
```

```
       default student   balance   income        preds
    1:      No      No  729.5265 44361.63 1.428724e-03
    2:      No     Yes  817.1804 12106.13 1.122204e-03
    3:      No      No 1073.5492 31767.14 9.812272e-03
    4:      No      No  529.2506 35704.49 4.415893e-04
    5:      No      No  785.6559 38463.50 1.935506e-03
   ---                                                
 9996:      No      No  711.5550 52992.38 1.323097e-03
 9997:      No      No  757.9629 19660.72 1.560251e-03
 9998:      No      No  845.4120 58636.16 2.896172e-03
 9999:      No      No 1569.0091 36669.11 1.471436e-01
10000:      No     Yes  200.9222 16862.95 3.322825e-05
```

Advanced R features -- data.table
===

It turns out that `data.frame`s, the base form of data structure in R, have an annoying syntax. Enter `data.table`s.

```r
library(data.table)
defaultdt <- as.data.table(Default)
defaultdt[student=="No",list(meanbalance = mean(balance), sdbalance = sd(balance), meanincome = mean(income)),by=default]
```

```
   default meanbalance sdbalance meanincome
1:      No    744.5044  445.5151   39993.52
2:     Yes   1678.4295  330.9141   40625.05
```

Advanced R features -- Grammar of Graphics (ggplot2)
===

This builds graphs using a grammar to describe how aesthetic (geometrical) elements relate to the data/statistics, rather than detailing where every point goes. This allows for a concise and coherent description of complex graphs. For example, http://stats.stackexchange.com/questions/87132/what-is-the-proper-name-for-a-river-plot-visualisation shows how to re-create the famous Minard graph of the fate of Napoleon's Grand Army in the 1812 Russian campaign.

Advanced R features -- Grammar of Graphics (ggplot2)
===

![](http://www.datavis.ca/gallery/minard/orig.gif)
![](http://www.datavis.ca/gallery/minard/ggplot2/temps.jpg)

Advanced R features -- closures
===

Closures are functions that return functions, e.g.

```r
power <- function(exponent) {
  function(x) {
    x ^ exponent
  }
}
square <- power(2)
square(2)
```

```
[1] 4
```

```r
square(4)
```

```
[1] 16
```

Advanced R features -- functionals
===

Functionals take functions as arguments, e.g.

```r
randomise <- function(f) f(runif(1e3))
randomise(mean)
```

```
[1] 0.4929153
```

```r
randomise(mean)
```

```
[1] 0.5073444
```

```r
randomise(sum)
```

```
[1] 512.9456
```

Advanced R features -- domain-specific languages
===

Formulas (`y ~ x`) and ggplot2 provide new syntaxes within R. Here's another example, converting R code to SQL:

```r
library(dplyr)
translate_sql(sin(x) + tan(y))
```

```
<SQL> SIN("x") + TAN("y")
```

```r
translate_sql(x < 5 & !(y >= 5))
```

```
<SQL> "x" < 5.0 AND NOT(("y" >= 5.0))
```

Advanced R features -- Use C++ within R
====

Sometimes you want to use C++ code to do something, probably because it's much faster than R or it has a ready-made implementation of a complicated data structure or algorithm that R lacks.

```r
library(Rcpp)
cppFunction('int add(int x, int y, int z) {
  int sum = x + y + z;
  return sum;
}')
add(1,2,3)
```

```
[1] 6
```

Advanced R features -- Use C++ within R
====

```r
cppFunction('double meanC(NumericVector x) {
  int n = x.size();
  double total = 0;
  for(int i = 0; i < n; ++i) {
    total += x[i];
  }
  return total / n;
}')
library(microbenchmark)
x <- runif(1e5)
microbenchmark(
  mean(x), # internal mean
  meanC(x) # new mean function
)
```

```
Unit: microseconds
     expr     min       lq     mean  median      uq     max neval
  mean(x) 200.074 200.5095 207.3000 200.890 204.231 287.093   100
 meanC(x)  98.654  98.8580 102.1444  99.098  99.746 185.675   100
```
