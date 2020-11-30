# Project1

The bank robber algorithm

--------------

You are a bank robber who's looking to rob as many banks in a day before you flee the country.

You got your hands on a list of banks in the area, with their location, the amount of money they have and the time they will take to rob. It looks like this:

```
id, x_coordinate, y_coordinate, money, time (hr)
0, 11.4, 3.3, 5000, 0.6
1, 6.9, 7.1, 15000, 0.3
2, 1.4, 13.2, 900, 1.1
```

This list of banks is in `bank_data.csv`

You have **24 hours** to make as much money as possible then escape.

# Rules:

- Your run can start anywhere on the map but it has to end at the **helicopter escape zone**: coordinates (0,0)
    - If you try to rob too many banks and can't get to the helicopter in 24 hours, you get caught and go to jail.

- You solution is a list or array of integers (eg. `[580, 433, 24, 998]`) where the numbers are the IDs of each banks. The ID of each bank is their index (their row index).

- You travel between banks at 30 km/h. You have to travel from one bank to the next!
    - Remember the formula to calculate the distance between two points.
    - The coordinates are in kilometers.
        - So (1, 1) and (1, 2) are one kilometer apart. 
        - This would take 1 / 30 hour = 2 minutes to travel

- Your solution should be an approximative/heuristic algorithm
    - This problem is NP-Hard, you won't find a fast enough algorithm that has the perfect solution
    - It doesn't need to find the best solution
    - Find the best solution you can!

- Your solution has to run on a common laptop in under 3 minutes for the 10,000 cities dataset
    - You can use everything you want to achieve this:
        - Use numpy, pandas, functions, algorithms
        - You can use parallelism
        - Use optimied libraries (pandas, numba, scipy, etc.)
    - Test your code on **small subsets** of the data so they run fast
        - Then scale your code up to bigger chunks of the data

- Your input is a pandas dataframe with the bank data. Your output is a list of bank IDs in order that you rob them:

**Ex:**

```
df = pd.read_csv('bank_data.csv')
robber_algorithm(df)

# Output is a list of bank IDs
[OUTPUT] --> [664, 2341, 26, 998, 9583, 24, 1, 444, 6783]
```

# Checking Your Solution:

You can use the `check_solution` function from `check_solution.py` to test if your solution is valid and verify the score.

# Hints:

- Most of the design paradigms we saw in class will work for this:
    - Divide-and-conquer
    - Brute Force
    - Greedy Algorithm
    - Dynamic Programming
    - Backtracking
    - Breadth-first & Depth-first search
    - Some we haven't seen:
        - Branch & Bound
        - Prune & Search
 
 Start with something that's easier (brute-force or greedy algorithm) and then work towards a better design once it works.
 
 - Because there are too many banks at each step, you will need to select only some candidates to explore
 
 - If you find yourself doing many **Nearest neighbors** type queries, consider using a [KD-Tree](https://en.wikipedia.org/wiki/K-d_tree) or a Ball Tree to speed it up.
     - There are good implementations of KD-Trees and nearest neighbours in scipy, sklearn and [this library](https://github.com/lmcinnes/pynndescent)

- You can work your algorithm backwards (starting at the end and backing up to the starting point) or forwards (finding a starting point and looping until there is no time left). They will lead to different designs however
