import random
def get_heros():
    heros = []
    for _ in range(40000):
        char1 = chr(65 + random.randint(0, 25))
        char2 = chr(65 + random.randint(0, 25))
        heros.append(char1 + char2)
    return heros
