---
layout: post
title:  "Three Euler projects"
date:   2020-09-11
image:  images/leonhard-euler.jpg
tags:   [study]
---

Hello, welcome to my blog! This post will share the solutions of three **[Eular projects][Eular projects]**.

Please visit my **[GitHub][GitHub]** for more information. 

# About Project Euler

Project Euler is a series of challenging mathematical/computer programming problems that will require more than just mathematical insights to solve. Although mathematics will help you arrive at elegant and efficient methods, the use of a computer and programming skills will be required to solve most problems.

The motivation for starting Project Euler, and its continuation, is to provide a platform for the inquiring mind to delve into unfamiliar areas and learn new concepts in a fun and recreational context.


# Three problems I chose

## 13. Large sum (Solved by 225,164)

### question
    Work out the first ten digits of the sum of the following one-hundred 50-digit numbers.
        37107287533902102798797998220837590246510135740250
        46376937677490009712648124896970078050417018260538
        74324986199524741059474233309513058123726617309629
        91942213363574161572522430563301811072406154908250
        23067588207539346171171980310421047513778063246676
        89261670696623633820136378418383684178734361726757
        28112879812849979408065481931592621691275889832738
        44274228917432520321923589422876796487670272189318
        47451445736001306439091167216856844588711603153276
        70386486105843025439939619828917593665686757934951
        ...
        *a hundred line*

<br>

### approach
I saved those one-hundred 50-digit numbers into a text file and read all lines. Using function `largeSum` I am able to convert the list of number strings into a list of int numbers and get the int sum of them. After converting the int sum into string sum, I can take the first 10 digits.

### code
{% highlight python %}
def largeSum(lines):
    """A function used to calculate the first ten digits of the sum of numbers.

    Parameters
    ----------
    lines : list of str
        A list of numbers, which are str type first and should be convert to int. 
    
    Returns
    -------
    str
        Calculate the sum of the following one-hundred 50-digit numbers, convert the int into str and return the first ten digits.
    
    Examples
    --------
    >>> lines = ['11111111111', '22222222222']
    >>> largeSum(lines)
    3333333333

    """
    return str(sum([int(i) for i in lines]))[:10]
{% endhighlight %}

### result
{% highlight python %}
if __name__ == "__main__":
    f = open("number.txt")
    lines = f.readlines()
    print(largeSum(lines))
    f.close() 
# 5537376230
{% endhighlight %}

The result of the first ten digits of the sum of the one-hundred 50-digit numbers give is *5537376230*.

<br>

## 31. Coin sums (Solved by 83,763)

### question
    In the United Kingdom the currency is made up of pound (£) and pence (p). There are eight coins in general circulation:
        1p, 2p, 5p, 10p, 20p, 50p, £1 (100p), and £2 (200p).
    It is possible to make £2 in the following way:
        1×£1 + 1×50p + 2×20p + 1×5p + 1×2p + 3×1p
    How many different ways can £2 be made using any number of coins?

<br>

### approach
I used dynamic programming solution to solve this coin change problem. Basically, the strategy is that for a new amount of coin, the ways to make up a specific amount, which equals another smaller amount plus this coin amount, should be the number of ways of the smaller amount plus the number of ways of this specific amount. For example, if the current coin iterated now is 5 and we want to update `dp[11]`, we should have `dp[11] = dp[11] + dp[6]`.

First initiate number of ways array with the base case "no coins": `dp[0] = 1`, and all the rest set to 0. For each type of coins, iterate through 0 to lenth of dp minus the amount of this coin, then update the `dp[i+c] += dp[i]`.

### code
{% highlight python %}
def coinSums(amount):
    """A function used to calculate the ways to make up a specific amount.

    Parameters
    ----------
    amount : int
        An amount of pence. 
    
    Returns
    -------
    int
        How many different ways can the amount of pence be made using any number of coins.
    
    Examples
    --------
    >>> coinSums(5)
    3
    
    """
    dp = [1] + [0]*amount
    coin = [1, 2, 5, 10, 20, 50, 100, 200]
    for c in coin:
        for i in range(len(dp) - c):
            dp[i + c] += dp[i]
    return dp[-1]
{% endhighlight %}

### result
{% highlight python %}
if __name__ == "__main__":
    print(coinSums(200))
# 73682
{% endhighlight %}
Run the coinSums() function. The result is £2 can be made *73682* ways using any number of coins.

<br>

## 124. Ordered radicals (Solved by 13,048)

### question 
    The radical of n, rad(n), is the product of the distinct prime factors of n. For example, 504 = 23 × 32 × 7, so rad(504) = 2 × 3 × 7 = 42.
    If we calculate rad(n) for 1 ≤ n ≤ 10, then sort them on rad(n), and sorting on n if the radical values are equal, we get:

            Unsorted       Sorted
            n  rad(n)   n   rad(n)  k
            1    1      1     1     1
            2    2      2     2     2
            3    3      4     2     3
            4    2      8     2     4
            5    5      3     3     5
            6    6      9     3     6
            7    7      5     5     7
            8    2      6     6     8 	
            9    3      7     7     9
            10   10     10    10    10 	

    Let E(k) be the kth element in the sorted n column; for example, E(4) = 8 and E(6) = 9.
    If rad(n) is sorted for 1 ≤ n ≤ 100000, find E(10000).

<br>

### approach
I used sieving to find primes and apply them as factors to calculate the integer radical of n. First initiate an array, of which each element is an array of radical and index, to store and update the radicals and later sort them to find out kth element. For example, if we have `rad = [[1,0],[1,1],[1,2],[1,3],[1,4]]` and we start from index 2, we will find that `rad[2][0]==1` and update `rad[2][0]` to 2 and `rad[4][0]` to 2 as well, by applying 2 as a factor to calculate the integer radical of 4.

### code
{% highlight python %}
def orderedRadicals(n, k):
    """A function used to find kth element in the sorted n column. 

    Parameters
    ----------
    n : int
        How many radicals we want to get.
    k : int
        kth element in the sorted n column.
    
    Returns
    -------
    int
        E(10000) of 100000 sorted radicals.
    
    Examples
    --------
    >>> orderedRadicals(10, 4)
    8
    
    """
    rad = [[1,_] for _ in range(n + 1)]
    for i in range(2, len(rad)):
        if rad[i][0] == 1:
            for j in range(i, len(rad), i):
                rad[j][0] *= i
    return sorted(rad)[k][1]
{% endhighlight %}

### result
{% highlight python %}
if __name__ == "__main__":
  	print(orderedRadicals(100000, 10000))
# 21417
{% endhighlight %}
If rad(n) is sorted for 1 ≤ n ≤ 100000, the result of E(10000) is *21417*.

<br>


[Eular projects]: https://projecteuler.net/archives
[GitHub]: https://github.com/eveyimi/eveyimi.github.io