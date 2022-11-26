


s = "0"
N = 1
while len(s) < 1000 :

    temp = ""
    n = N
    while n > 0:
        temp += str(n % 2)
        n //= 2
    s += temp[::-1]

    N += 1

print(N)
