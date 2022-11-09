import numpy as np

DOMAIN = { 'Sphere' : [-5.12, 5.12],
            'Zakharov' : [-5, 10],
            'Rosenbrock' : [-5, 10],
            'Michalewicz' : [0, np.pi],
            'Ackley' : [-32.768, 32.768]
         }

def sphere(x: np.array) -> float:
    return _sphere(x)

def _sphere(x: np.array) -> float:
    return np.sum(x**2)

def zakharov(x: np.array) -> float:
    sum_2, sum_ixi = _zakharov(x)
    return sum_2 + sum_ixi**2 + sum_ixi**4

def _zakharov(x: np.array) -> float:
    sum_2, sum_ixi = 0, 0

    for index, value in enumerate(x):
        sum_2 += value**2
        sum_ixi += 0.5 * (index + 1) * value
    return sum_2, sum_ixi

def rosenbrock(x: np.array) -> float:
    sum_value = 0

    for i in range(len(x) - 1):
        sum_value += _rosenbrock(x[i], x[i+1])
    return sum_value

def _rosenbrock(xcurrent: float ,xnext: float) -> float:
    return 100 * ((xnext - xcurrent**2)**2) + (xcurrent - 1)**2

def michalewicz(x: np.array) -> float:
    sum_value = 0

    for index, value in enumerate(x):
        sum_value += _michalewicz(index+1, value)
    return -sum_value

def _michalewicz(index: int, value: float, m: int = 10) -> float:
    return np.sin(value) * ((np.sin(index * value**2 / np.pi))**(2 * m))

def ackley(x: np.array) -> float:
    return _ackley(x)

def _ackley(x: np.array, a: float = 20, b: float = 0.2, c: float = 2 * np.pi) -> float:
    sum_x2, sum_cosx = 0, 0

    for value in x:
        sum_x2 += value**2
        sum_cosx += np.cos(c * value)

    exp_x2, exp_cosx = -a * np.exp(-b * np.sqrt(sum_x2 / len(x))), -np.exp(sum_cosx/len(x))
    return exp_x2 + exp_cosx + a + np.exp(1)

def clip_numbers(x: np.array, objective_function: str) -> np.array:
    try:
        x_min, x_max =  DOMAIN[objective_function]
        x = np.clip(x, x_min, x_max)
        return x
    except KeyError:
        print("Please check the objective function, only in ('Sphere', 'Zakharov', 'Rosenbrock', 'Ackley')")
        return 
        
if __name__ == '__main__':
    print(sphere(np.array([0,0,0])))
    print(zakharov(np.array([0,0])))
    print(rosenbrock(np.array([1,1,1,1])))
    print(michalewicz(np.array([2.20,1.57])))
    print(ackley(np.array([0,0])))
    # print(clip_numbers(np.array([10,20,30,3,4,5,6,35,-45]),'Ackle'))
    # draw_3d()