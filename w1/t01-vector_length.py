def length(v):
    sum = 0
    for i in v:
        sum += i**2

    return sum**0.5


va = [2, 2]
vb = [3, 4]
print("a", length(va))
print("b", length(vb))
assert length(va) == 8**0.5
assert length(vb) == 5
