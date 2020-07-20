import numpy as np
data = dict()
with open("embed.txt") as f:
    content = f.read().strip().split('\n')
    content = [i for i in content if i!='']
    print(content[:5])
    for i in range(0, len(content), 2):
        k = content[i]
        val = content[i+1].split()
        val = np.asarray(val, dtype = 'float')
        data[k] = val
        
def cosine_sim(A, B):
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

def most_similar(w):
    v = data[w]
    top = float('-inf')
    most_sim = None
    for i in data:
        sim = cosine_sim(v, data[i])
        if sim>top:
            most_sim = i
    return most_sim

most_similar("Thanos")