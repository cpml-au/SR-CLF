from sympy import (
    symbols,
    Matrix,
    sin,
    cos,
)

# Define symbolic variables
x1, x2, x3, x4 = symbols('x1 x2 x3 x4')


# Define x(x) as a vector
def xSR(x1, x2, x3, x4):
    return Matrix([x1, x2, x3, x4])


# Define f(x) as a vector
def fSR(x1, x2, x3, x4):
    # Constants
    g = -10       # gravity (m/s^2)
    L = 2         # length of pole (m)
    d = 1         # damping coefficient
    m = 1         # mass of pendulum (kg)
    M = 5         # mass of cart (kg)

    Sx = -sin(x3)
    Cx = -cos(x3)
    D = m * L * L * (M + m * (1 - Cx**2))

    f1 = x2
    f2 = (1 / D) * (-m**2 * L**2 * g * Cx * Sx + m * L**2 * (m * L * x4**2 * Sx - d * x2))
    f3 = x4
    f4 = (1 / D) * ((m + M) * m * g * L * Sx - m * L * Cx * (m * L * x4**2 * Sx - d * x2))
    
    return Matrix([f1, f2, f3, f4])


# Define G(x) as a matrix
def GSR(x1, x2, x3, x4):
    
    # Constants
    g = -10       # gravity (m/s^2)
    L = 2         # length of pole (m)
    d = 1         # damping coefficient
    m = 1         # mass of pendulum (kg)
    M = 5         # mass of cart (kg)

    Sx = -sin(x3)
    Cx = -cos(x3)
    D = m * L * L * (M + m * (1 - Cx**2))

    u1 = m * L * L * (1 / D)  
    u2 = m * L * Cx * (1 / D) 

    G1 = [0, 0, 0, 0]
    G2 = [0, u1, 0, 0]
    G3 = [0, 0, 0, 0]
    G4 = [0, -u2, 0, 0]
    
    return Matrix([G1, G2, G3, G4])


# Define Q(x) as a matrix
def QSR(x1, x2, x3, x4):
    Q1 = [1, 0, 0, 0]
    Q2 = [0, 1, 0, 0]
    Q3 = [0, 0, 1, 0]
    Q4 = [0, 0, 0, 1]
    
    return Matrix([Q1, Q2, Q3, Q4])


# Define R(x) as a matrix
def RSR(x1, x2, x3, x4):
    R1 = [0.0001, 0, 0, 0]
    R2 = [0, 0.0001, 0, 0]
    R3 = [0, 0, 0.0001, 0]
    R4 = [0, 0, 0, 0.0001]
    
    return Matrix([R1, R2, R3, R4])
