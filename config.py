INITIAL_LABEL_BUDGET = 100
INITIAL_COMPUTE_BUDGET = 1000
MEMORY_CAPACITY = 10

# cL > 0. One label acquisition spends this many budget units.
LABEL_COST = 1

# CC(0) = 0 and CC(c1) < CC(c2) < CC(c3).
COMPUTE_COSTS = {
    0: 0,
    1: 250,
    2: 500,
    3: 1000,
}

# cM(DISCARD) = 0. STORE and REPLACE have operation overhead only.
MEMORY_OPERATION_COSTS = {
    "DISCARD": 0,
    "STORE": 1,
    "REPLACE": 1,
}