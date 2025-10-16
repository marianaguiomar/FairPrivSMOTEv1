import math, random

# --- Configurations ---
class The:
    b = 1000        # Bootstrap iterations
    conf = 0.95     # Confidence threshold
    a12 = 0.6       # Minimum A12 effect size
    useA12 = True   # If False, use Cliff's delta

# --- Utilities ---
def median(lst):
    lst = sorted(lst)
    n = len(lst)
    mid = n // 2
    if n % 2 == 0:
        return (lst[mid-1]+lst[mid])/2
    return lst[mid]

def mean(lst):
    return sum(lst)/len(lst)

def bootstrap(y0, z0):
    combined = y0 + z0
    n, m = len(y0), len(z0)
    y = [random.choice(combined) for _ in range(n)]
    z = [random.choice(combined) for _ in range(m)]
    return mean(y) - mean(z)

def cliffsDelta(lst1, lst2):
    n = len(lst1)*len(lst2)
    more = sum(1 for x in lst1 for y in lst2 if x>y)
    less = sum(1 for x in lst1 for y in lst2 if x<y)
    return (more - less)/n

def A12(lst1, lst2):
    more = sum(1 for x in lst1 for y in lst2 if x>y)
    ties = sum(1 for x in lst1 for y in lst2 if x==y)
    return (more + 0.5*ties)/(len(lst1)*len(lst2))

def significant(lst1, lst2):
    if The.useA12:
        return A12(lst1, lst2) >= The.a12 or A12(lst2,lst1) >= The.a12
    else:
        return abs(cliffsDelta(lst1,lst2)) >= The.a12

def bootstrapSignificant(lst1, lst2):
    obs = mean(lst1) - mean(lst2)
    count = sum(1 for _ in range(The.b) if abs(bootstrap(lst1,lst2)) >= abs(obs))
    return count/float(The.b) <= 1-The.conf

# --- Scott-Knott recursive splitting ---
def scottknott(data, rank=1):
    if len(data) == 1:
        return [(rank, data[0])]
    data = sorted(data, key=lambda x: median(x[1:]))
    best_score = -1
    best_split = None
    for i in range(1, len(data)):
        left = data[:i]
        right = data[i:]
        lmed = [x[1:] for x in left]
        rmed = [x[1:] for x in right]
        lmed_flat = [v for sub in lmed for v in sub]
        rmed_flat = [v for sub in rmed for v in sub]
        score = (len(lmed_flat)*mean(lmed_flat)**2 + len(rmed_flat)*mean(rmed_flat)**2)
        if score > best_score:
            best_score = score
            best_split = i
    left = data[:best_split]
    right = data[best_split:]
    left_flat = [v for x in left for v in x[1:]]
    right_flat = [v for x in right for v in x[1:]]
    if significant(left_flat,right_flat) and bootstrapSignificant(left_flat,right_flat):
        left_ranks = scottknott(left, rank)
        right_ranks = scottknott(right, rank+1)
        return left_ranks + right_ranks
    else:
        return [(rank, x) for x in data]

# --- Display results ---
def rdivDemo(data):
    ranked = scottknott(data)
    print("-"*60)
    for r,x in ranked:
        name = x[0]
        vals = x[1:]
        med = median(vals)
        print(f"{r} | {name:10} | median={med:.3f} | {vals}")
    print("-"*60)

# --- Example usage ---
if __name__ == "__main__":
    data = [
        ["optimizer1", 0.34, 0.49, 0.51, 0.6],
        ["optimizer2", 0.6, 0.7, 0.8, 0.9],
        ["optimizer3", 0.15, 0.25, 0.35, 0.4],
        ["optimizer4", 0.6, 0.7, 0.8, 0.9],
        ["optimizer5", 0.1, 0.2, 0.3, 0.4]
    ]
    rdivDemo(data)