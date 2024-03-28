# Intro

Most people have more expereince with probability than linear algebra. 

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

Variance measures how much the data points in a dataset are spread out from their average value.
variance - how much a random variable deviates from its own mean.  to get the number we square it to remove the possible of getting a negative value.

covariance variance is covariance of x with itself
when x and y are independent covariance = 0. 0 doesnt mean they are independent tho.  

Correlation assesses the strength and direction of the linear relationship between two variables. - dimesnionless quantity. in range -1 to 1. things closer +1 strongly correlated, -1 negativly correlated, 0 is no relationship. can think of corellation as a way to normalize covariance.



