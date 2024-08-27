def dot(a, b):
    sum = 0
    for i, j in zip(a, b):
        sum += i * j
    print(sum)
    return sum


va = [2, 2]
vb = [3, 4]
assert dot(va, vb) == 14
