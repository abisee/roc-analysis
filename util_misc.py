import math

# Partially from http://billchambers.me/tutorials/2014/12/21/tf-idf-explained-in-python.html
def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)