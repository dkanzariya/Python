def can_coincide(X, Y):
    i = 0
    while i < len(X):
        if X[i] == Y[i]:
            i += 1
        elif X[i] == 'A' and X[i+1] == 'B':
            X = X[:i] + 'B' + 'A' + X[i+2:]
        elif X[i] == 'C':
            if Y[i] == 'A':
                X = X[:i] + 'A' + X[i+1:]
            elif Y[i] == 'B':
                X = X[:i] + 'B' + X[i+1:]
            else:
                return False

        else:
            return False
    return X == Y

T = 1
for _ in range(T):
    N, X, Y = 2, "BA", "AB"
    result = can_coincide(X, Y)
    if result:
        print("Yes")
    else:
        print("No")
