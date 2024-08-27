def length2(v):
    return sum([e**2 for e in v]) ** 0.5


va = [2, 2]
vb = [3, 4]
print("a", length2(va))
print("b", length2(vb))
assert length2(va) == 8**0.5
assert length2(vb) == 5
