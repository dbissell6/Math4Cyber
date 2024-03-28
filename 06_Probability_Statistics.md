# Intro

Most people have more expereince with probability than linear algebra. Becasue of that we can review more definitions upfront, they shouldnt be so foriegn. 

# Definitions

Probability theory - quantitative study of random processes

Sample space - set of all possible outcomes

A coin flip has 2 possible outcomes.
Rolling dice has 6 possible outcomes.

A molecule in a box would be set of all possible positions of the molecule in the box.

Event - One specific outcome

Probability - Frequency of observing an event

random variable - a function that maps an event to a number

Heads -> 1
Tails -> 2

Dice are easy because they already output a number.
die 1-6 == 1-6

Same with the molecule(if thats how we created our grid)
molecule (x0,y0)-> (xn,yn)

# Independence of random variables

Independence of 2 random variables, knowing something about one variable doesn't tell us anything about another variable.

A coin and die would be independent variables. Flipping heads or tails, doesnt change or impact the next flip.

probability of b given we know A doesnt change us knowing prob of b without knowing A. and flip we know is 50/50 even if we knew first flip was H.

As an example of something that wouldnt have independence, imagine pulling cards from a deck. If we pull a card, then leave that card out, the probability of the next card we pick, will be impacted by that previous pick.
We can fix this independce if after every pick, we put the card back and shuffle again. Replacement makes it an independent process.

# Discrete distribution and continuous

Flipping a coin (H or T) is discrete, binary. 

Mapping the molecule in phase space would be continuous. 

these cases are mapped with different functions.

continuous - probability density function (PDF)

Total probabilities of each event should equal 1. 

to calculate something like what is prob that die role will be greater than or equal to 3? 1/6 * 4 = 1/2

to do a similar thing with continuous space we need to use an integral from calc. Called the cumulative distribution function or CDF. 

to convert a discrete distribution into a continuous one we must use integral. to get a probability density function from a discrete probability distribution

# Mulivariate probability

more than one random variable are used to describe the outcome of the event. yahtzee is good example of multivariate prob. 

# Moments of discrete and continuous distributions

one way to characterize these distributions is by calculating thier moments. Moments are different properties of distributions that give us information about underlying probabilities.

moments are a series of different scaler values.

in continuous space instead of summing over all values of x, we integrate over all values of x, also get an expected value (average)

multivariate

x and y could be dependent or independent. 

# Linearity of expected value

interesting property of average(expected value) is linear. 

multiplication by a scaler. 

if variables are independent pdf. can calculate expected values of products based on calculating individual expected values.

# Combinatorics

Mathematical science of how we count.

How many ways are there to order a set of objects.

4 dice(4 anything) 4* 3 * 2 * 1= 24

N! for any number

Different if we had to arrange 4 die and the positions that they were in - permutations

ways to arrange indistinguishable objects - combinations

Serilings approzimation is importnat for engineering becasue they often encounter formulas that use natural log of a factorial.

# Variance, covariance and correlation

The variance, covariance and correlation are special properties we can calculate from expected values(average). 

Imagine our boss wanted to know more about failed login attempts and they gave us these data over the past 14 days.

```
[50, 55, 52, 53, 60, 500, 654, 52, 51, 50, 48, 44, 465, 721]
```

One of the first things we can look at is the variance.

Variance measures how much the data points in a dataset are spread out from their average value.

variance - how much a random variable deviates from its own mean.  to get the number we square it to remove the possible of getting a negative value.

Python makes this easy. 

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/afbec22b-1d14-4847-ac00-860f79acc389)

We can see a high variance? why is that? Well most days there are ~50 failed login attempts. 2 days a week there are ~500. For the variance to be low, each day should have close to the same number of failed login attempts.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/1cee0170-6604-4a09-b399-7369819dce65)

What if every day has exactly 50 failed login attempts?

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/916b5c81-1ca8-465f-90cd-892cda2482db)

<details>

<summary>Python code for variance</summary>

```
   import numpy as np

# Pseudo data: Number of failed login attempts to a system over 14 days
login_attempts = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50])
variance = np.var(login_attempts)
print(login_attempts)
print(f"Variance of login attempts: {variance}")



#%%
def calculate_variance(data):
    # Calculate mean
    mean = sum(data) / len(data)
    # Calculate variance
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance

# Example data: Number of login attempts to a system over 10 days
login_attempts = [50, 55, 52, 53, 500, 54, 52, 51, 50, 48]
print(f"Variance of login attempts: {calculate_variance(login_attempts)}")
```

</details>

What if we wanted to understand how two things interact with each other? For instance say now our boss gives us this data for the past 10 days of failed logins and external ips.

failed logins - `[50, 55, 52, 53, 500, 54, 52, 51, 50, 48]`
external  ips - `[10, 15, 11, 13, 50, 12, 10, 14, 15, 11]`

To look at the relationship between these variables we can use covariance. Look at how these variables vary in relationship to eachother.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/7c5d688f-1264-4639-9c7e-90a697874573)

This has a high covariance. why? becsaue on the day failed logins shoots up to 500(x10 of every otherday), external ips ALSO shoots up to 50(x5 of other days).
We can change that external ips value to 11(which is about the average). And see the covariance plummets.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/58acbf4e-3df4-4051-af60-857703e92d63)


covariance variance is covariance of x with itself

When x and y are independent covariance = 0. However, getting a 0 doesnt mean they are independent tho.  

<details>

<summary>Python code for covariance</summary>

```
import numpy as np

# Pseudo data: Failed logins and external IPs accessing the network over 10 days
failed_logins = np.array([50, 55, 52, 53, 500, 54, 52, 51, 50, 48])
external_ips = np.array([10, 15, 11, 13, 11, 12, 10, 14, 15, 11])

covariance = np.cov(failed_logins, external_ips)[0][1]
print(f'Failed Logins: {failed_logins}')
print(f'External Ips: {external_ips}')
print(f"Covariance between failed logins and external IPs: {covariance}")

#%%

def calculate_covariance(x, y):
    if len(x) != len(y):
        return "Arrays must be of equal length."
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)
    return covariance

# Example data: Email traffic and external IPs accessing the network over 10 days
email_traffic = [200, 220, 210, 205, 500, 200, 198, 215, 220, 210]
external_ips = [10, 15, 11, 13, 11, 12, 10, 14, 15, 11]
print(f"Covariance between email traffic and external IPs: {calculate_covariance(email_traffic, external_ips)}")
```

Correlation assesses the strength and direction of the linear relationship between two variables. It is a dimesnionless quantity in range -1 to 1. Closer +1 strongly positivily correlated, -1 strongly negativly correlated, 0 is no relationship. We can think of correlation as a way to normalize covariance.

Og example with data being 500,50 showing high correlation.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/36eff89a-dceb-4f54-a2a3-12f405eab127)



