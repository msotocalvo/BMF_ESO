import numpy as np

class complexity:
    name = "Complexity Evaluation"
    iterations = 1_000_000
    optimal = 0
    bounds = (-100,100)   
        
        
    @staticmethod
    def function(x):
        for i in range(1, 2 + 1):
            x = 0.55 + float(i)
            x += x + i
            x /= 2
            x = x * x
            x = np.sqrt(x)
            x = np.log(x)
            x = np.exp(x)
            x = x / (x + 2)
        return x    
    
class ackley1:
    name = 'Ackley 1'
    optimal = 0
    bounds = (-35,35)
    type = 'multimodal'
       
    @staticmethod
    def function(x):        
        return -20 * np.exp(-0.2 * np.sqrt(sum(x_i**2 for x_i in x) / len(x))) - np.exp(sum(np.cos(2 * np.pi * x_i) for x_i in x) / len(x)) + 20 + np.e

class ackley2:   
    name = 'Ackley 2'
    optimal = -200
    bounds = (-32,32)
    type = 'unimodal'
       
    @staticmethod
    def function(x):
        
        return -200 * np.exp(-0.02 * np.sqrt(x[0] ** 2 + x[1] ** 2))    
 
class adjiman: ## OK
    name = 'Adjiman'
    optimal = -2.0218
    bounds = (-1,2), (-1,1)
    type = 'multimodal'
    
    @staticmethod
    def function(x):
        return np.cos(x[0]) * np.sin(x[1]) - x[0] / (x[1]**2 + 1)
    
class alpine_1:
    name = 'Alpine 1'
    optimal = 0
    bounds = (-10,10)
    type = 'multimodal'
    
    @staticmethod
    def function(x):
        return sum([np.abs(xi * np.sin(xi) + 0.1 * xi) for xi in x])    
    
class bartels_conn:
    name = 'Bartels Conn'
    optimal = 0
    bounds = (-500,500)
    type = 'multimodal'
    
    @staticmethod
    def function(x):        
        return  abs(x[0]**2 + x[1]**2 + x[0] * x[1]) + abs(np.sin(x[0])) + abs(np.cos(x[1]))     

class Beale:
    name = 'Beale'
    optimal = 0
    bounds = (-4.5,4.5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2

class Bohachevsky1:
    name = 'Bohachevsky 1'
    optimal = 0
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7
        
class BentCigar:
    name = 'Bent Cigar'
    optimal = 0.0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return x[0]**2 + 10**6 * sum([xi**2 for xi in x[1:]])

class Booth:
    name = 'Booth'
    optimal = 0.00000
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    
class Branin01:  ## OK
    name = 'Branin 01'
    optimal = 0.39788735772973816
    bounds = (-5,10), (0,15)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return ((x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)    

class Branin02: 
    name = 'Branin 02'
    optimal = 5.559037
    bounds = (-5,15)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return ((x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
                + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) * np.cos(x[1]) + np.log(x[0] ** 2.0 + x[1] ** 2.0 + 1.0) + 10)

class Brent:
    name = 'Brent'
    optimal = 0.00000
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return (x[0]+10)**2 + (x[1]+10)**2 + np.exp(-x[0]**2-x[1]**2)
    
class Brown:
    name = 'Brown'
    optimal = 0.00000
    bounds = (-1,4)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum((x[0] ** 2.0) ** (x[1] ** 2.0 + 1.0) + (x[1] ** 2.0) ** (x[0] ** 2.0 + 1.0))   

class Bukin2: ## OK
    name = 'Bukin 2'
    optimal = -124.75
    bounds = (-15,-5), (-3,3)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 100 * (x[1] ** 2 - 0.01 * x[0] ** 2 + 1.0) + 0.01 * (x[0] + 10.0) ** 2.0
    
class Bukin4:
    name = 'Bukin 4'
    optimal = 0.00000
    bounds = (-15,-5), (-3,3)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 100 * x[1] ** 2 + 0.01 * abs(x[0] + 10)   
    
class Bukin6:
    name = 'Bukin 6'
    optimal = 0.00000
    bounds = (-15,-5), (-3,3)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 100 * np.sqrt(abs(x[1] - 0.01 * (x[0]**2))) + 0.01 * abs(x[0] + 10)

class ChungReynolds:
    name = 'Chung Reynolds'
    optimal = 0.0
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return (sum(x_i**2 for x_i in x))**2

class CrownedCross:
    name = 'Crowned cross'
    optimal = 0.0001
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))) + 1) ** (0.1)
        
class CrossInTray: ## OK
    name = 'CrossInTray'
    optimal = -2.0626
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2)/np.pi))) + 1)**0.1

class CrossLegTable:
    name = 'Cross Leg Table'
    optimal = -1
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -(np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))) + 1) ** (-0.1)

class Corana:
    name = 'Corana'
    optimal = 0.00000
    bounds = (-500,500)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum(0.15 * (0.2 * np.floor(abs(x[i]) / 0.2) + 0.49999 * np.sign(x[i]) - 0.05 * np.sign(x[i]))**2 * [1, 1000, 10, 100][i] if abs(x[i] - (0.2 * np.floor(abs(x[i]) / 0.2) + 0.49999 * np.sign(x[i]))) < 0.05 else [1, 1000, 10, 100][i] * x[i]**2 for i in range(4))

class Cola:
    name = 'Cola'
    optimal = 11.7464
    bounds = (0,4), (-4,4)
    type = 'multimodal'

    @staticmethod
    def function(x):
        d = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1.27, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1.69, 1.43, 0, 0, 0, 0, 0, 0, 0, 0],
                 [2.04, 2.35, 2.43, 0, 0, 0, 0, 0, 0, 0],
                 [3.09, 3.18, 3.26, 2.85, 0, 0, 0, 0, 0, 0],
                 [3.20, 3.22, 3.27, 2.88, 1.55, 0, 0, 0, 0, 0],
                 [2.86, 2.56, 2.58, 2.59, 3.12, 3.06, 0, 0, 0, 0],
                 [3.17, 3.18, 3.18, 3.12, 1.31, 1.64, 3.00, 0, 0, 0],
                 [3.21, 3.18, 3.18, 3.17, 1.70, 1.36, 2.95, 1.32, 0, 0],
                 [2.38, 2.31, 2.42, 1.94, 2.85, 2.81, 2.56, 2.91, 2.97, 0.]])
        
        xi = np.atleast_2d(np.asarray([0.0, x[0]] + list(x[1::2])))
        xj = np.repeat(xi, np.size(xi, 1), axis=0)
        xi = xi.T

        yi = np.atleast_2d(np.asarray([0.0, 0.0] + list(x[2::2])))
        yj = np.repeat(yi, np.size(yi, 1), axis=0)
        yi = yi.T
        inner = (np.sqrt(((xi - xj) ** 2 + (yi - yj) ** 2)) - d) ** 2
        inner = np.tril(inner, -1)
        return np.sum(np.sum(inner, axis=1))
    
class CosineMixture:
    name = 'Cosine Mixture'
    optimal = 0
    bounds = (-1,1)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        return -0.1 * sum([np.cos(5*np.pi*x[i]) for i in range(n)]) - sum([x[i]**2 for i in range(n)])

class Csendes:
    name = 'Csendes'
    optimal = 0.00000
    bounds = (-1,1)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum([xi**6 * (2 + np.sin(1/xi)) for xi in x])
        
class Cube:
    name = 'Cube'
    optimal = 0.0
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 100 * (x[1] - x[0]**3)**2 + (1 - x[0])**2

class Cone:
    name = 'Cone'
    optimal = 0.0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sqrt(np.sum(np.array(x)**2))
        
class Damavandi:
    name = 'Damavandi'
    optimal = 0
    bounds = (0,14)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return (1 - (np.abs((np.sin(np.pi * (x[0] - 2)) * np.sin(np.pi * (x[1] - 2))) / (np.pi**2 * (x[0] - 2) * (x[1] - 2))))**5) * (2 + (x[0]-7)**2 + 2*(x[1]-7)**2)

class DeVilliersGlasser01:
    name = 'De Villiers Glasser 01'
    optimal = 0
    bounds = (-500,500)
    type = 'multimodal'
        
    @staticmethod
    def function(x):
        n = 24
        return np.sum((x[0] * x[1]**(0.1 * (i - 1)) * np.sin(x[2] * (0.1 * (i - 1)) + x[3]) - 60.137*(1.371)**(0.1 * (i - 1)) * np.sin(3.112*(0.1 * (i - 1)) + 1.761)) ** 2 for i in range(1, n + 1))

class DeVilliersGlasser02:
    name = 'De Villiers Glasser 02'
    optimal = 0
    bounds = (-10,10)
    type = 'multimodal'
        
    @staticmethod
    def function(x):
        n = 24
        return np.sum((x[0] * x[1]**(0.1 * (i - 1)) * np.tanh(x[2] * (0.1 * (i - 1)) + np.sin(x[3] * (0.1 * (i - 1)))) * np.cos((0.1 * (i - 1)) * np.exp(x[4] * (0.1 * (i - 1)))) - (53.81 * np.tanh(3.01 * (0.1 * (i - 1)) + np.sin(2.13 * (0.1 * (i - 1)))) * np.cos( np.exp(0.507) * (0.1 * (i - 1))))) ** 2 for i in range(1, n + 1))

class Deceptive: ## Mas o menos
    name = 'Deceptive'
    optimal = -1
    bounds = (0,1)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        alpha = np.arange(1.0, n + 1.0) / (n + 1.0)
        beta = 2.0
        g = np.zeros((n,))
        for i in range(n):
            if x[i] <= 0.0:
                g[i] = x[i]
            elif x[i] < 0.8 * alpha[i]:
                g[i] = -x[i] / alpha[i] + 0.8
            elif x[i] < alpha[i]:
                g[i] = 5.0 * x[i] / alpha[i] - 4.0
            elif x[i] < (1.0 + 4 * alpha[i]) / 5.0:
                g[i] = 5.0 * (x[i] - alpha[i]) / (alpha[i] - 1.0) + 1.0
            elif x[i] <= 1.0:
                g[i] = (x[i] - 1.0) / (1.0 - alpha[i]) + 4.0 / 5.0
            else:
                g[i] = x[i] - 1.0
        return -((1.0 / n) * np.sum(g)) ** beta

class Discus:
    name = 'Discus'
    optimal = 0
    bounds = (-100,100)
    type = 'Unimodal'

    @staticmethod
    def function(x):
        n = len(x)
        return (x[0] - 1)**2 + np.sum([i*(2*x[i]**2 - x[i-1])**2 for i in range (n-1)])


class Dolan:
    name = 'Dolan'
    optimal = 0
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return (abs((x[0] + 1.7 * x[1]) * np.sin(x[0]) - 1.5 * x[2] - 0.1 * x[3] * np.cos(x[3] + x[4] - x[0]) + 0.2 * x[4] ** 2 - x[1] - 1))

class DropWave: ## OK
    name = 'Drop Wave'
    optimal = -1
    bounds = (-5.12,5.12)
    type = 'multimodal'

    @staticmethod
    def function(x):
        norm_x = np.sum(x ** 2)
        return -(1 + np.cos(12 * np.sqrt(norm_x))) / (0.5 * norm_x + 2)

class DixonPrice:
    name = 'Dixon Price'
    optimal = 0
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        i = np.arange(2, len(x) + 1)
        s = i * (2.0 * x[1:] ** 2.0 - x[:-1]) ** 2.0
        return np.sum(s) + (x[0] - 1.0) ** 2.0

class Easom: ## OK
    name = 'Easom'
    optimal = -1.00000
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))    

class EggCrate:
    name = 'Egg Crate'
    optimal = 0.00000
    bounds = (-5,5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return (x[0]**2 + x[1]**2 + 25) * (np.sin(x[0])**2 + np.sin(x[1])**2)
    
class EggHolder:
    name = 'Egg Holder'
    optimal = -959.640662711
    bounds = (-512,512)
    type = 'multimodal'

    @staticmethod
    def function(x):
        vec = (-(x[1:] + 47) * np.sin(np.sqrt(abs(x[1:] + x[:-1] / 2. + 47))) - x[:-1] * np.sin(np.sqrt(np.abs(x[:-1] - (x[1:] + 47)))))
        return np.sum(vec)    

class Ellipse:
    name = 'Ellipse'
    optimal = 0.0
    bounds = (-10,10)
    type = 'unimodal'
    
    @staticmethod
    def function(x):
        n = len(x)
        return sum((10**((i)/(n-1)) * x[i])**2 for i in range(len(x)))
    
class Exponential:
    name = 'Exponential'
    optimal = 0.0
    bounds = (-1,1)
    type = 'unimodal'
    
    @staticmethod
    def function(x):
        
        return -np.exp(0-.5*np.sum(x**2))
   
class ElAttarVidyasagarDutta:
    name = 'El Attar Vidyasagar Dutta'
    optimal = 1.712780354
    bounds = (-500,500)
    type = 'unimodal'

    @staticmethod
    def function(x):
        
        return ((x[0] ** 2 + x[1] - 10) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2 + (x[0] ** 2 + x[1] ** 3 - 1) ** 2)    
    
class Griewank:
    name = 'Griewank'
    optimal = 0.00000
    bounds = (-600,600)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return np.sum(x ** 2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1., np.size(x) + 1.)))) + 1
    
class GoldsteinPrice: ## OK
    name = 'Goldstein Price'
    optimal = 3
    bounds = (-2,2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return (1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)) * (30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))   
    
class Gulf:
    name = 'Gulf'
    optimal = 0
    bounds = (0,60)
    type = 'multimodal'

    @staticmethod
    def function(x):
        m = 99
        i = np.arange(1., m + 1)
        u = 25 + (-50 * np.log(i / 100.)) ** (2 / 3.)
        vec = (np.exp(-((np.abs(u - x[1])) ** x[2] / x[0])) - i / 100.)
        return np.sum(vec ** 2)    

class HappyCat:
    name = 'Happy Cat'
    optimal = 0
    bounds = (-2,2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        alpha = 1.0/8
        return ((np.sum(x**2) - len(x))**2)**alpha + (0.5*np.sum(x**2)+np.sum(x))/len(x) + 0.5
    
class HighConditionedElliptic:
    name = 'High Conditioned Elliptic'
    optimal = 0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        D = len(x)  # Dimensionalidad de la entrada
        return np.sum([(10**6)**((i - 1)/(D - 1)) * (xi**2) for i, xi in enumerate(x, start=1)])    
    
class HgBat:
    name = 'HgBat'
    optimal = 0
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        sum_squares = np.sum(x**2)
        sum_fourth_powers = np.sum(x**4)
        D = len(x)  # Dimensionalidad de la entrada
        term1 = (sum_fourth_powers - sum_squares**2)**0.5
        term2 = (0.5 * sum_squares + np.sum(x)) / D
        return term1 + term2 + 0.5  
    
class HimmelBlau:
    name = 'HimmelBlau'
    optimal = 0
    bounds = (-6,6)
    type = 'multimodal'

    @staticmethod
    def function(x):
        
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2    

class HolderTable: ## OK
    name = 'Holder Table'
    optimal = -19.208
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2)/np.pi)))

class Kowalik:
    name = 'Kowalik'
    optimal = 0.00030748610
    bounds = (-5,5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        a = np.asarray([4.0, 2.0, 1.0, 1 / 2.0, 1 / 4.0, 1 / 6.0, 1 / 8.0,
                          1 / 10.0, 1 / 12.0, 1 / 14.0, 1 / 16.0])
        b = np.asarray([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
                          0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        vec = b - (x[0] * (a ** 2 + a * x[1]) / (a ** 2 + a * x[2] + x[3]))
        return np.sum(vec ** 2)
    
class Katsuura: ## OK
    name = 'Katsuura'
    optimal = 1
    bounds = (0,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        d = 32
        n = len(x)
        k = np.atleast_2d(np.arange(1, d + 1)).T
        idx = np.arange(0., n * 1.)
        inner = np.round(2 ** k * x) * (2. ** (-k))
        return np.prod(np.sum(inner, axis=0) * (idx + 1) + 1)
    
class Langermann:
    name = 'Langermann'
    optimal = -5.1621259
    bounds = (-1.2,1.2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        a = np.array([3, 5, 2, 1, 7])
        b = np.array([5, 2, 1, 4, 9])
        c = np.array([1, 2, 5, 2, 3])
        return (-np.sum(c * np.exp(-(1 / np.pi) * ((x[0] - a) ** 2 +
                (x[1] - b) ** 2)) * np.cos(np.pi * ((x[0] - a) ** 2 + (x[1] - b) ** 2)))) 
    
class Leon:
    name = 'Leon'
    optimal = 0.00000
    bounds = (-1.2,1.2)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    
class LennardJones: ## OK
    name = 'Lennard Jones'
    # optimal = [-1.0, -3.0, -6.0, -9.103852, -12.712062,
    #                    -16.505384, -19.821489, -24.113360, -28.422532,
    #                    -32.765970, -37.967600, -44.326801, -47.845157,
    #                    -52.322627, -56.815742, -61.317995, -66.530949,
    #                    -72.659782, -77.1777043]
    optimal = -3.0 ## Optimum for D=10
    bounds = (-4,4)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        k = int(n / 3)
        s = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                a = 3 * i
                b = 3 * j
                xd = x[a] - x[b]
                yd = x[a + 1] - x[b + 1]
                zd = x[a + 2] - x[b + 2]
                ed = xd * xd + yd * yd + zd * zd
                ud = ed * ed * ed
                if ed > 0.0:
                    s += (1.0 / ud - 2.0) / ud
        return s    

class Levy13:
    name = 'Levy 13'
    optimal = 0.00000
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
         return (x[0] - 1)**2 * np.sin(3*np.pi*x[1])**2 + (x[1] - 1)**2 * (1 + np.sin(2*np.pi*x[1])**2) +  (np.sin(3*np.pi*x[0])**2)


class Matyas:
    name = 'Matyas'
    optimal = 0.00000
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
    
class Michaelwicz:
    name = 'Michaelwicz'
    optimal = -1.8013
    bounds = (0,np.pi)
    type = 'multimodal'

    @staticmethod
    def function(x):
        m = 10
        return -sum([np.sin(xi) * np.sin((i+1) * xi**2 / np.pi)**(2*m) for i, xi in enumerate(x)])  

class Mishra01: ## OK
    name = 'Mishra 01'
    optimal = 2
    bounds = (0,1)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        xn = n - np.sum((x[:-1] + x[1:]) / 2.0)
        return (1 + xn) ** xn     
    
class Mishra02:
    name = 'Mishra 02'
    optimal = 2
    bounds = (0,1)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        xn = n - - np.sum(x[0:-1])
        return (1 + xn) ** xn     
    
class Mishra03: ## OK
    name = 'Mishra 03'
    optimal = -0.19990562
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 0.01 * (x[0] + x[1]) + np.sqrt(np.abs(np.cos(np.sqrt(np.abs(x[0] ** 2 + x[1] ** 2)))))    
    
class Mishra04: ## OK
    name = 'Mishra 04'
    optimal = -0.17767
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 0.01 * (x[0] + x[1]) + np.sqrt(np.abs(np.sin(np.sqrt(abs(x[0] ** 2 + x[1] ** 2)))))  
    
class NewFuntion01: ## OK
    name = 'New Funtion 01'
    optimal = -0.18459899925
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return ((np.abs(np.cos(np.sqrt(np.abs(x[0] ** 2 + x[1]))))) ** 0.5 + 0.01 * (x[0] + x[1]))  
    
class NewFuntion02: ## OK
    name = 'New Funtion 02'
    optimal = -0.19933159253
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return ((np.abs(np.sin(np.sqrt(np.abs(x[0] ** 2 + x[1]))))) ** 0.5 + 0.01 * (x[0] + x[1]))   
    
class OddSquare:
    name = 'Odd Square'
    optimal = -1.0084
    bounds = (-5*np.pi,5*np.pi)
    type = 'multimodal'

    @staticmethod
    
    def function( x):
        a = np.array([1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4, 1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4])
        b = a[:19]
        d = 19 * np.max((x - b)**2)
        h = np.sum((x - b)**2)
        return -np.exp(-d/(2.0*np.pi))*np.cos(np.pi*d)*(1.0 + 0.02*h/(d + 0.01))          
    
class PowellSingular1:
    name = 'Powell Singular 1'
    optimal = 0.00000
    bounds = (-4.5,5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum((x[4*i-4] + 10*x[4*i-3])**2 + 5*(x[4*i-2] - x[4*i-1])**2 + (x[4*i-3] - 2*x[4*i-2])**4 + 10*(x[4*i-4] - x[4*i-1])**4 for i in range(1, len(x)//4 + 1))

class PowellSingular2:
    name = 'Powell Singular 2'
    optimal = 0.00000
    bounds = (-4,5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum((x[i-1]+10*x[i])**2 + 5*(x[i+1]-x[i+2])**2 + (x[i]-2*x[i+1])**4 + 10 * (x[i-1]-x[i+2])**4 for i in range(len(x)-2))

class PowellSum:
    name = 'Powell Sum'
    optimal = 0.00000
    bounds = (-1,1)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum(abs(x_i)**(i+1) for i, x_i in enumerate(x))

class Price1:
    name = 'Price 1'
    optimal = 0
    bounds = (-500,500)
    type = 'multimodal'
    
    @staticmethod
    def function(x):
        return (np.abs(x[0]) - 5)**2 + (np.abs(x[1]) - 5)**2
    
class Price2:
    name = 'Price 2'
    optimal = 0.9
    bounds = (-500,500)
    type = 'multimodal'
    
    @staticmethod
    def function(x):
        return 1 + np.sin(x[0])**2 + np.sin(x[1])**2 - 0.1*np.exp(-x[0]**2 - x[1]**2)    

class Quadric:
    name = 'Unimodal'
    optimal = 0.0
    bounds = (-1.28,1.28)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum((sum(x[j] for j in range(i))**2) for i in range(len(x)))
           
class Rastrigin:
    name = 'Rastrigin'
    optimal = 0.00000
    bounds = (-5,5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum(x_i**2 - 10 * np.cos(2 * np.pi * x_i) + 10 for x_i in x)
    
class Ridge:
    name = 'Ridge'
    optimal = 0.00000
    bounds = (-5,5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return x[0] + 2*np.sum(x[1:]**2)**0.5    
    
class Ripple01: ## OK
    name = 'Ripple 01'
    optimal = -2.2
    bounds = (0,1)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return  sum(-np.exp(-2 * np.log(2) * ((x[i]/0.8)**2)) * (np.sin(5 * np.pi * x[i])**6 + 0.1 * np.cos(500 * np.pi * x[i])**2) for i in range(2))   

class Rosenbrock:
    name = 'Rosenbrock'
    optimal = 0.0
    bounds = (-30,30)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum(100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2 for i in range(len(x) - 1))
    
class RosenbrockModified:
    name = 'Rosenbrock Modified'
    optimal = 34.37
    bounds = (-2,2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 74 + 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2 - 400 * np.exp(-((x[0] + 1)**2 + (x[1] + 1)**2) / 0.1)
    
class RotatedEllipse:
    name = 'Rotated Ellipse'
    optimal = 0.0
    bounds = (-500,500)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 7*x[0]**2 - 6*np.sqrt(3)*x[0]*x[1] + 13*x[1]**2
    
class Rump:
    name = 'Rump'
    optimal = 0.0
    bounds = (-500,500)
    type = 'unimodal'

    @staticmethod
    def function(x):
        term1 = (333.75 - x[0]**2)*x[1]**6 + x[0]**2 * (11*x[0]**2 * x[1]**2 - 121*x[1]**4 - 2)
        term2 =5.5*x[1]**8 + (x[0]/(2*x[1]))

        # Sum the terms to get the result.
        return term1 + term2    

class Salomon:
    name = 'Salomon'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 1 - np.cos(2 * np.pi * np.sqrt(np.sum([xi**2 for xi in x]))) + 0.1 * np.sqrt(np.sum([xi**2 for xi in x]))

class Schaffer:
    name = 'Schaffer'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001*np.sum(x**2))**2

class Schaffer2:
    name = 'Schaffer 2'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001*np.sum(x**2))**2
    
class Schaffer3:
    name = 'Schaffer 3'
    optimal = 0.00156685
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.5 + (np.sin(np.cos(np.abs( x[0]**2 - x[1]**2 ))) - 0.5) / (1 + 0.001*np.sum(x**2))**2
    
class Schaffer4:
    name = 'Schaffer 4'
    optimal = 0.292579
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.5 + (np.cos(np.sin(np.abs( x[0]**2 - x[1]**2 ))) - 0.5) / (1 + 0.001*np.sum(x**2))**2

class Schwefel01:
    name = 'Schwefel 01'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'
    
    @staticmethod
    def function(x):
        return sum(x_i**2 for x_i in x)**2
    
class Schwefel02:
    name = 'Schwefel 02'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'    

    @staticmethod
    def function(x):
        total_sum = 0
        for i in range(len(x)):
            inner_sum = sum(x[j] for j in range(i + 1))
            total_sum += inner_sum ** 2
        return total_sum    

   
class Schwefel20:
    name = 'Schwefel 20'
    optimal = 0.0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(abs(x))

class Schwefel21:
    name = 'Schwefel 21'
    optimal = 0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return max(abs(x))

class Schwefel22:
    name = 'Schwefel 22'
    optimal = 0.0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(abs(x)) + np.prod(abs(x))
    
class Schwefel23:
    name = 'Schwefel 23'
    optimal = 0.0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(x**10)    

class Sphere:
    name = 'Sphere'
    optimal = 0
    bounds = (0,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(np.square(x))

class SineEnvelop:
    name = 'Sine Envelop'
    optimal = 0
    bounds = (-20,20)
    type = 'multimodal'

    @staticmethod
    def function(x):       
        return - np.sum((np.sin(np.sqrt(x[i+1]**2 + x[i]**2) - 0.5)**2) / ((0.001*(x[i+1]**2 + x[i]**2) + 1)**2) + 0.5 for i in range(len(x) - 1))

class Stochastic:
    name = 'Stochastic'
    optimal = 0.00000
    bounds = (-5,5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        epsilon = np.random.uniform(0, 1, size=len(x))
        return np.sum(epsilon * np.abs(x - 1 / (np.arange(1, len(x) + 1))))

class Step1:
    name = 'Step 1'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum([np.floor(abs(xi)) for xi in x])

class Step2:
    name = 'Step 2'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(np.floor(x + 0.5)**2)

class Step3:
    name = 'Step 3'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(np.floor(x**2))
    
class Stepint:
    name = 'Stepint'
    optimal = 0.00000
    bounds = (-5.12,5.12)
    type = 'unimodal'

    @staticmethod
    def function(x):
        n = len(x)
        return  np.sum(np.abs(x[i]) for i in range(n))    

class StretchedVSineWave:
    name = 'Stretched V Sine Wave'
    optimal = 0
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum(((x[i]**2 + x[i+1]**2)**0.25) * (np.sin(50 * (x[i]**2 + x[i+1]**2)**0.1)**2 + 0.1) for i in range(len(x) - 1))
    
class SumSquares:
    name = 'Sum squares'
    optimal = 0.00000
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum((i+1)*x[i]**2 for i in range(len(x)))

class Trefethen: ## OK
    name = 'Trefethen'
    optimal = -3.3068
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return np.exp(np.sin(50 * x[0])) + np.sin(60 * np.exp(x[1])) + np.sin(70 * np.sin(x[0])) + np.sin(np.sin(80 * x[1])) - np.sin(10 * (x[0] + x[1])) + (x[0]**2 + x[1]**2) / 4

class Trid06:
    name = 'Trid 06'
    optimal = -50
    bounds = (-20,20)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum([(x[i] - 1)**2 for i in range(len(x))]) - sum([x[i] * x[i - 1]  for i in range(len(x)-1)])
    
class Tripod:
    name = 'Tripod'
    optimal = 0
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    
    def p(x):
        return 1 if x >= 0 else 0
    
    def function(x):
        return Tripod.p (x[1]) * (1 + Tripod.p(x[0])) + abs(x[0] + 50*Tripod.p(x[1]*(1-2*Tripod.p(x[0]))))  + abs(x[1] - 50*(1 - 2* Tripod.p (x[1])))    
    
class Ursem1: ## OK
    name = 'Ursem 1'
    optimal = -4.81681406371
    bounds = (-2.5,3), (-2,2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -np.sin(2 * x[0] - 0.5 * np.pi) - 3.0 * np.cos(x[1]) - 0.5 * x[0]   

class Weierstrass:
    name = 'Weierstrass'
    optimal = 0.00000
    bounds = (-0.5, 0.5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return np.sum([np.sum([0.5**(i+1) * np.cos(2 * np.pi * 3**i * (xi + 0.5)) for i in range(20)]) for xi in x]) - len(x) * np.sum([np.sum([0.5**(i+1) * np.cos(2 * np.pi * 3**i * 0.5) for i in range(20)])])
    
class WayburnSeader1:
    name = 'Wayburn Seader 1'
    optimal = 0.0
    bounds = (-5,5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return (x[0]**6 + x[1]**4 - 17)**2 + (2*x[0] + x[1] - 4)**2   
    
class WayburnSeader2:
    name = 'Wayburn Seader 2'
    optimal = 0.0
    bounds = (-50,50)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return (1.613 - 4*(x[0] - 0.3125)**2 - 4*(x[1] - 1.625)**2)**2 + (x[1] - 1)**2
    

class XinSheYang1:
    name = 'Xin-She Yang 1'
    optimal = 0
    bounds = (-5,5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum(np.random.uniform(0, 1) * abs(x[i])**i for i in range(len(x)))

class XinSheYang2:
    name = 'Xin-She Yang 2'
    optimal = 0
    bounds = (-2*np.pi,2*np.pi)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return np.sum(np.abs(x[i]) for i in range(len(x))) / np.exp(sum(np.sin(x[i]**2) for i in range(len(x))))

class XinSheYang3:
    name = 'Xin-She Yang 3'
    optimal = -1
    bounds = (-20,20)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return np.exp(-np.sum((x / 15.0)**(2.0 * 5))) - 2.0 * np.exp(-np.sum(x**2)) * np.prod(np.cos(x)**2)
    
class XinSheYang4:
    name = 'Xin-She Yang 4'
    optimal = -1
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        t1 = np.sum(np.sin(x)**2)
        t2 = -np.exp(-np.sum(x**2))
        t3 = -np.exp(np.sum(np.sin(np.sqrt(np.abs(x)))**2))
        return (t1 + t2) * t3    

class Zackarov:
    name = 'Zackarov'
    optimal = 0
    bounds = (-5,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        return np.sum(np.square(x)) + (np.sum(0.5 * np.arange(1, n+1) * x))**2 + (np.sum(0.5 * np.arange(1, n+1) * x))**4 
    
class ZeroSum:
    name = 'Zero Sum'
    optimal = 0
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        if np.abs(np.sum(x)) < 3e-16:
            return 0.0
        return 1.0 + (10000.0 * np.abs(np.sum(x))) ** 0.5  
    
class Zirilli:
    name = 'Zirilli'
    optimal = -0.3523
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.25*x[0]**4 - 0.5*x[0]**2 + 0.1*x[0] +0.5*x[1]**2  
    
class Zimmerman:
    name = 'Zimmerman'
    optimal = 0
    bounds = (0,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        Zh1 = lambda x: 9.0 - x[0] - x[1]
        Zh2 = lambda x: (x[0] - 3.0) ** 2.0 + (x[1] - 2.0) ** 2.0 - 16.0
        Zh3 = lambda x: x[0] * x[1] - 14.0
        Zp = lambda x: 100.0 * (1.0 + x)
        return max(Zh1(x), Zp(Zh2(x)) * np.sign(Zh2(x)), Zp(Zh3(x)) * np.sign(Zh3(x)), Zp(-x[0]) * np.sign(x[0]),Zp(-x[1]) * np.sign(x[1]))      

## Rotated and shifted functions

class ShiftedRotatedCigar:  ## OKK
    name = 'Shifted Rotated Cigar'
    optimal = 789.123
    dimension = 25
    type = 'unimodal'
    bounds = [
        (-10, 10) for _ in range(25)
    ]
    
    shift = np.array([4.2, -3.8, 5.1, -4.7, 3.5, -5.3, 4.8, -3.4, 5.5, -4.1,
                     3.7, -5.6, 4.4, -3.9, 5.2, -4.5, 3.6, -5.4, 4.9, -3.3,
                     4.2, -3.8, 5.1, -4.7, 3.5])
    
    # Matriz de rotación 25D usando composición de rotaciones
    rotation_matrix = np.eye(25)
    for i in range(24):
        for j in range(i+1, 25):
            theta = np.pi/4 + (i*j*np.pi/100)
            R = np.eye(25)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedCigar.rotation_matrix @ (x - ShiftedRotatedCigar.shift)
        
        # Función Cigar modificada
        first_term = shifted_rotated_x[0]**2
        other_terms = 1e6 * np.sum(shifted_rotated_x[1:]**2)
        
        # Añadir término oscilatorio para hacer la función más desafiante
        oscillation = np.sum(np.sin(0.1 * np.pi * shifted_rotated_x))
        
        return first_term + other_terms + oscillation + 789.123

class ShiftedRotatedAckley01: ## OKK
    name = 'Shifted Rotated Ackley 01'
    optimal = -123.456
    dimension = 30
    type = 'multimodal'
    bounds = [
        (-32.768, 32.768) for _ in range(30)  # Límites estándar de Ackley para cada dimensión
    ]
    
    shift = np.array([2.1, -1.7, 3.2, -2.8, 1.5, -3.4, 2.6, -1.9, 3.8, -2.3,
                     1.8, -3.5, 2.9, -1.6, 3.7, -2.4, 1.4, -3.6, 2.7, -1.8,
                     3.3, -2.5, 1.6, -3.3, 2.8, -1.5, 3.9, -2.2, 1.7, -3.7])
    
    # Matriz de rotación 30D usando una matriz ortogonal
    rotation_matrix = np.random.randn(30, 30)
    q, r = np.linalg.qr(rotation_matrix)
    rotation_matrix = q

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedAckley01.rotation_matrix @ (x - ShiftedRotatedAckley01.shift)
        
        # Términos principales de Ackley
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(shifted_rotated_x**2)))
        term2 = -np.exp(np.mean(np.cos(2*np.pi*shifted_rotated_x)))
        
        # Términos adicionales para aumentar la dificultad
        term3 = np.sum(np.sin(shifted_rotated_x/2)**2) / len(shifted_rotated_x)
        term4 = np.sum(np.cos(shifted_rotated_x/3)**2) / len(shifted_rotated_x)
        
        return term1 + term2 + 20 + np.e + term3 + term4 - 123.456 


class ShiftedRotatedSchaffer: ## OKK
    name = 'Shifted Rotated Schaffer'
    optimal = 423.9229
    dimension = 15
    type = 'multimodal'
    bounds = [
        (-100, 100) for _ in range(15)
    ]
    
    shift = np.array([15.2, -12.8, 18.4, -14.7, 11.5, -19.3, 16.8, -13.4, 17.5, -11.1,
                     13.7, -16.6, 14.4, -18.9, 12.2])
    
    # Matriz de rotación 15D usando rotaciones compuestas
    rotation_matrix = np.eye(15)
    for i in range(14):
        for j in range(i+1, 15):
            theta = np.pi/3 + (i*j*np.pi/30)
            R = np.eye(15)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedSchaffer.rotation_matrix @ (x - ShiftedRotatedSchaffer.shift)
        
        # Versión modificada de Schaffer para alta dimensionalidad
        sum_term = 0
        for i in range(len(shifted_rotated_x)-1):
            x_i = shifted_rotated_x[i]
            x_next = shifted_rotated_x[i+1]
            
            inner_square = x_i**2 + x_next**2
            sin_term = np.sin(np.sqrt(inner_square))**2 - 0.5
            denominator = (1 + 0.001*inner_square)**2
            
            sum_term += 0.5 + sin_term/denominator
        
        # Término adicional para acoplar todas las dimensiones
        coupling_term = np.sum(np.sin(np.sqrt(shifted_rotated_x**2 + np.roll(shifted_rotated_x, 1)**2)))
        
        return sum_term + 0.1*coupling_term + 421.7843

class ShiftedRotatedRastrigin: ## OKK
    name = 'Shifted Rotated Rastrigin'
    optimal = 954.5044
    dimension = 25
    type = 'multimodal'
    bounds = [
        (-5.12, 5.12) for _ in range(25)  # Límites estándar de Rastrigin
    ]
    
    shift = np.array([2.1, -1.7, 2.4, -1.9, 2.2, -1.8, 2.3, -1.6, 2.5, -1.5,
                     2.0, -1.4, 2.6, -1.3, 1.9, -1.2, 2.7, -1.1, 1.8, -1.0,
                     2.8, -0.9, 1.7, -0.8, 2.9])
    
    # Matriz de rotación 25D usando composición de rotaciones
    rotation_matrix = np.eye(25)
    for i in range(24):
        for j in range(i+1, 25):
            theta = np.pi/4 + (i*j*np.pi/50)
            R = np.eye(25)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedRastrigin.rotation_matrix @ (x - ShiftedRotatedRastrigin.shift)
        
        # Versión modificada de Rastrigin
        omega = 2*np.pi  # Frecuencia base
        A = 10.0        # Amplitud
        
        # Término principal de Rastrigin con frecuencia variable
        main_term = np.sum([
            shifted_rotated_x[i]**2 - A*np.cos(omega*(i+1)*shifted_rotated_x[i]) 
            for i in range(len(shifted_rotated_x))
        ])
        
        # Término de acoplamiento entre dimensiones
        coupling = 0.0
        for i in range(len(shifted_rotated_x)-1):
            for j in range(i+1, len(shifted_rotated_x)):
                coupling += 0.2 * shifted_rotated_x[i] * shifted_rotated_x[j] * \
                           np.sin(omega*np.sqrt(shifted_rotated_x[i]**2 + shifted_rotated_x[j]**2))
        
        return A*len(shifted_rotated_x) + main_term + coupling + 894.3215

class ShiftedRotatedZakharov: ## OKK
    name = 'Shifted Rotated Zakharov'
    optimal = 623.8295
    dimension = 20
    type = 'unimodal'
    bounds = [
        (-10, 10) for _ in range(20)
    ]
    
    shift = np.array([2.1, -1.7, 2.4, -1.9, 2.2, -1.8, 2.3, -1.6, 2.5, -1.5,
                     2.0, -1.4, 2.6, -1.3, 1.9, -1.2, 2.7, -1.1, 1.8, -1.0])
    
    # Matriz de rotación 20D usando composición de rotaciones
    rotation_matrix = np.eye(20)
    for i in range(19):
        for j in range(i+1, 20):
            theta = np.pi/4 + (i*j*np.pi/40)
            R = np.eye(20)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedZakharov.rotation_matrix @ (x - ShiftedRotatedZakharov.shift)
        
        # Términos principales de Zakharov
        sum1 = np.sum(shifted_rotated_x**2)
        
        # Término cuadrático modificado con pesos variables
        weighted_sum = np.sum([0.5*(i+1)*shifted_rotated_x[i] for i in range(len(shifted_rotated_x))])
        
        # Término de acoplamiento
        coupling = 0.0
        for i in range(len(shifted_rotated_x)-1):
            coupling += shifted_rotated_x[i] * shifted_rotated_x[i+1]
        
        return sum1 + weighted_sum**2 + weighted_sum**4 + 0.1*coupling + 623.8295

class ShiftedRotatedBrent: ## OKK
    name = 'Shifted Rotated Brent'
    optimal = -650.862696
    dimension = 25
    type = 'unimodal'
    bounds = [
        (-20, 20) for _ in range(25)
    ]
    
    shift = np.array([3.2, -2.8, 3.4, -2.6, 3.6, -2.4, 3.8, -2.2, 4.0, -2.0,
                     4.2, -1.8, 4.4, -1.6, 4.6, -1.4, 4.8, -1.2, 5.0, -1.0,
                     5.2, -0.8, 5.4, -0.6, 5.6])
    
    # Matriz de rotación 25D usando una matriz ortogonal
    rotation_matrix = np.random.randn(25, 25)
    q, r = np.linalg.qr(rotation_matrix)
    rotation_matrix = q

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedBrent.rotation_matrix @ (x - ShiftedRotatedBrent.shift)
        
        # Versión modificada de Brent
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)):
            sum_term += (shifted_rotated_x[i] + 10)**2
        
        # Término exponencial modificado
        exp_term = np.exp(-np.sum(shifted_rotated_x**2)/len(shifted_rotated_x))
        
        # Término de acoplamiento entre dimensiones
        coupling = 0.0
        for i in range(len(shifted_rotated_x)-1):
            coupling += np.abs(shifted_rotated_x[i] * shifted_rotated_x[i+1])
        
        return sum_term + exp_term + 0.1*coupling - 869.4567

class ShiftedRotatedSphere: ## OKK
    name = 'Shifted Rotated Sphere'
    optimal = 742.9183
    dimension = 30
    type = 'unimodal'
    bounds = [
        (-100, 100) for _ in range(30)
    ]
    
    shift = np.array([10.2, -9.8, 10.4, -9.6, 10.6, -9.4, 10.8, -9.2, 11.0, -9.0,
                     11.2, -8.8, 11.4, -8.6, 11.6, -8.4, 11.8, -8.2, 12.0, -8.0,
                     12.2, -7.8, 12.4, -7.6, 12.6, -7.4, 12.8, -7.2, 13.0, -7.0])
    
    # Matriz de rotación 30D
    rotation_matrix = np.random.randn(30, 30)
    q, r = np.linalg.qr(rotation_matrix)
    rotation_matrix = q

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedSphere.rotation_matrix @ (x - ShiftedRotatedSphere.shift)
        
        # Versión modificada de Sphere
        weighted_sum = 0.0
        for i in range(len(shifted_rotated_x)):
            # Peso que aumenta con la dimensión
            weight = 1 + 0.1 * i
            weighted_sum += weight * shifted_rotated_x[i]**2
        
        # Término de interacción entre dimensiones consecutivas
        interaction = 0.0
        for i in range(len(shifted_rotated_x)-1):
            interaction += shifted_rotated_x[i] * shifted_rotated_x[i+1]
        
        return weighted_sum + 0.05*interaction + 742.9183



class ShiftedRotatedAlpine01: ## OKK
    name = 'Shifted Rotated Alpine 01'
    optimal = 478.9324
    dimension = 20
    type = 'multimodal'
    bounds = [
         (-10, 10) for _ in range(20)
    ]
    
    shift = np.array([2.1, -1.7, 2.4, -1.9, 2.2, -1.8, 2.3, -1.6, 2.5, -1.5,
                     2.0, -1.4, 2.6, -1.3, 1.9, -1.2, 2.7, -1.1, 1.8, -1.0])
    
    # Matriz de rotación 20D usando composición de rotaciones
    rotation_matrix = np.eye(20)
    for i in range(19):
        for j in range(i+1, 20):
            theta = np.pi/4 + (i*j*np.pi/40)
            R = np.eye(20)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedAlpine01.rotation_matrix @ (x - ShiftedRotatedAlpine01.shift)
        
        # Versión modificada de Alpine01
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)):
            # Peso que aumenta con la dimensión
            weight = 1 + 0.1 * i
            sum_term += weight * np.abs(shifted_rotated_x[i] * np.sin(shifted_rotated_x[i]) + 0.1 * shifted_rotated_x[i])
        
        # Término de acoplamiento entre dimensiones
        coupling = 0.0
        for i in range(len(shifted_rotated_x)-1):
            coupling += np.abs(shifted_rotated_x[i] * shifted_rotated_x[i+1] * 
                             np.sin(shifted_rotated_x[i] + shifted_rotated_x[i+1]))
        
        return sum_term + 0.1*coupling + 478.9324

class ShiftedRotatedEggCrate: ## OKK
    name = 'Shifted Rotated Egg Crate'
    optimal = -892.4567
    dimension = 25
    type = 'multimodal'
    bounds = [
        (-5, 5) for _ in range(25)
    ]
    
    shift = np.array([1.2, -0.8, 1.4, -0.6, 1.6, -0.4, 1.8, -0.2, 2.0,
                     -0.1, 2.2, 0.1, 2.4, 0.3, 2.6, 0.5, 2.8, 0.7, 3.0,
                     0.9, 3.2, 1.1, 3.4, 1.3, 3.6])
    
    # Matriz de rotación 25D
    rotation_matrix = np.random.randn(25, 25)
    q, r = np.linalg.qr(rotation_matrix)
    rotation_matrix = q

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedEggCrate.rotation_matrix @ (x - ShiftedRotatedEggCrate.shift)
        
        # Versión modificada de Egg Crate para alta dimensionalidad
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)):
            sum_term += shifted_rotated_x[i]**2 + 2 * (np.sin(2*np.pi*shifted_rotated_x[i]))**2
        
        # Término de acoplamiento no lineal
        coupling = 0.0
        for i in range(len(shifted_rotated_x)-1):
            coupling += np.sin(np.pi*(shifted_rotated_x[i] + shifted_rotated_x[i+1])) * \
                       np.cos(np.pi*(shifted_rotated_x[i] - shifted_rotated_x[i+1]))
        
        return sum_term + coupling - 892.4567

class ShiftedRotatedCrossInTray: ## OKK
    name = 'Shifted Rotated Cross In Tray'
    optimal = 537
    dimension = 15
    type = 'multimodal'
    bounds = [
         (-10, 10) for _ in range(15)
    ]
    
    shift = np.array([2.1, -1.7, 2.4, -1.9, 2.2, -1.8, 2.3, -1.6, 2.5, -1.5,
                     2.0, -1.4, 2.6, -1.3, 1.9])
    
    # Matriz de rotación 15D
    rotation_matrix = np.eye(15)
    for i in range(14):
        for j in range(i+1, 15):
            theta = np.pi/3 + (i*j*np.pi/30)
            R = np.eye(15)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedCrossInTray.rotation_matrix @ (x - ShiftedRotatedCrossInTray.shift)
        
        # Versión modificada de Cross in Tray para alta dimensionalidad
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)-1):
            term1 = np.abs(np.sin(shifted_rotated_x[i]) * np.sin(shifted_rotated_x[i+1]))
            term2 = np.exp(np.abs(100 - np.sqrt(shifted_rotated_x[i]**2 + shifted_rotated_x[i+1]**2)/np.pi))
            sum_term += -0.0001 * (np.abs(term1 * term2) + 1)**0.1
        
        # Término adicional de acoplamiento
        coupling = np.sum([
            np.sin(shifted_rotated_x[i] * shifted_rotated_x[i+1]) *
            np.cos(shifted_rotated_x[i] + shifted_rotated_x[i+1])
            for i in range(len(shifted_rotated_x)-1)
        ])
        
        return sum_term + 0.1*coupling + 567.8912

class ShiftedRotatedLevy01: ## OKK
    name = 'Shifted Rotated Levy 01'
    optimal = -345.6789
    dimension = 30
    type = 'multimodal'
    bounds = [
        (-10, 10) for _ in range(30)
    ]
    
    shift = np.array([1.1, -0.9, 1.3, -0.7, 1.5, -0.5, 1.7, -0.3, 1.9, -0.1,
                     2.1, 0.1, 2.3, 0.3, 2.5, 0.5, 2.7, 0.7, 2.9, 0.9,
                     3.1, 1.1, 3.3, 1.3, 3.5, 1.5, 3.7, 1.7, 3.9, 1.9])
    
    # Matriz de rotación 30D
    rotation_matrix = np.random.randn(30, 30)
    q, r = np.linalg.qr(rotation_matrix)
    rotation_matrix = q

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedLevy01.rotation_matrix @ (x - ShiftedRotatedLevy01.shift)
        
        w = 1 + (shifted_rotated_x - 1) / 4
        
        # Término principal de Levy modificado
        sum_term = np.sin(np.pi * w[0])**2
        
        # Términos intermedios con pesos variables
        for i in range(len(shifted_rotated_x)-1):
            weight = 1 + 0.1 * i  # Peso que aumenta con la dimensión
            sum_term += weight * ((w[i]-1)**2 * (1 + 10*np.sin(np.pi*w[i] + 1)**2))
        
        # Término final
        sum_term += (w[-1]-1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
        
        # Término de acoplamiento entre dimensiones
        coupling = 0.0
        for i in range(len(shifted_rotated_x)-1):
            coupling += np.sin(np.pi * (w[i] + w[i+1])) * \
                       np.cos(np.pi * (w[i] - w[i+1]))
        
        return sum_term + 0.1*coupling - 345.6789     


class ShiftedRotatedBooth: ## OKK
    name = 'Shifted Rotated Booth'
    optimal = 859.886
    dimension = 20
    type = 'multimodal'
    bounds = [
         (-10, 10) for _ in range(20)
    ]
    
    shift = np.array([2.1, -1.7, 2.4, -1.9, 2.2, -1.8, 2.3, -1.6, 2.5, -1.5,
                     2.0, -1.4, 2.6, -1.3, 1.9, -1.2, 2.7, -1.1, 1.8, -1.0])
    
    # Matriz de rotación 20D usando composición de rotaciones
    rotation_matrix = np.eye(20)
    for i in range(19):
        for j in range(i+1, 20):
            theta = np.pi/4 + (i*j*np.pi/40)
            R = np.eye(20)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedBooth.rotation_matrix @ (x - ShiftedRotatedBooth.shift)
        
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)-1):
            # Términos de Booth modificados
            term1 = (shifted_rotated_x[i] + 2*shifted_rotated_x[i+1] - 7)**2
            term2 = (2*shifted_rotated_x[i] + shifted_rotated_x[i+1] - 5)**2
            # Factor de peso que aumenta con la dimensión
            weight = 1 + 0.1*i
            sum_term += weight * (term1 + term2)
        
        # Término de acoplamiento no lineal
        coupling = 0.0
        for i in range(len(shifted_rotated_x)-2):
            coupling += np.sin(np.pi*(shifted_rotated_x[i] + shifted_rotated_x[i+1] + shifted_rotated_x[i+2]))
        
        return sum_term + 0.1*coupling + 789.1234

class ShiftedRotatedTrid: ## OKK
    name = 'Shifted Rotated Trid'
    optimal = -560.2633
    dimension = 25
    type = 'unimodal'
    bounds = [
        (-25, 25) for _ in range(25)
    ]
    
    shift = np.array([3.2, -2.8, 3.4, -2.6, 3.6, -2.4, 3.8, -2.2, 4.0, -2.0,
                     4.2, -1.8, 4.4, -1.6, 4.6, -1.4, 4.8, -1.2, 5.0, -1.0,
                     5.2, -0.8, 5.4, -0.6, 5.6])
    
    # Matriz de rotación 25D
    rotation_matrix = np.random.randn(25, 25)
    q, r = np.linalg.qr(rotation_matrix)
    rotation_matrix = q

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedTrid.rotation_matrix @ (x - ShiftedRotatedTrid.shift)
        
        # Primer término modificado con pesos
        sum_squared = 0.0
        for i in range(len(shifted_rotated_x)):
            weight = 1 + 0.05*i
            sum_squared += weight * (shifted_rotated_x[i] - 1)**2
        
        # Segundo término modificado con acoplamiento
        sum_product = 0.0
        for i in range(1, len(shifted_rotated_x)):
            sum_product += shifted_rotated_x[i] * shifted_rotated_x[i-1] * (1 + 0.1*np.sin(np.pi*i/len(shifted_rotated_x)))
        
        return sum_squared - sum_product - 456.7891

class ShiftedRotatedRosenbrock:## OKK
    name = 'Shifted Rotated Rosenbrock'
    optimal = 943.4567
    dimension = 30
    type = 'multimodal'
    bounds = [
        (-30, 30) for _ in range(30)
    ]
    
    shift = np.array([4.1, -3.9, 4.2, -3.8, 4.3, -3.7, 4.4, -3.6, 4.5, -3.5,
                     4.6, -3.4, 4.7, -3.3, 4.8, -3.2, 4.9, -3.1, 5.0, -3.0,
                     5.1, -2.9, 5.2, -2.8, 5.3, -2.7, 5.4, -2.6, 5.5, -2.5])
    
    # Matriz de rotación 30D
    rotation_matrix = np.eye(30)
    for i in range(29):
        for j in range(i+1, 30):
            theta = np.pi/6 + (i*j*np.pi/60)
            R = np.eye(30)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedRosenbrock.rotation_matrix @ (x - ShiftedRotatedRosenbrock.shift)
        
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)-1):
            # Factor de peso que aumenta con la dimensión
            weight = 1 + 0.1*i
            # Términos de Rosenbrock modificados
            term1 = 100 * weight * (shifted_rotated_x[i+1] - shifted_rotated_x[i]**2)**2
            term2 = (shifted_rotated_x[i] - 1)**2
            sum_term += term1 + term2
        
        # Término de acoplamiento adicional
        coupling = np.sum([
            np.sin(np.pi * shifted_rotated_x[i] * shifted_rotated_x[i+1])
            for i in range(len(shifted_rotated_x)-1)
        ])
        
        return sum_term + 0.1*coupling + 923.4567

class ShiftedRotatedPowellSum: ## OKK
    name = 'Shifted Rotated Powell Sum'
    optimal = -234.5678
    dimension = 20
    type = 'unimodal'
    bounds = [
         (-5, 5) for _ in range(20)
    ]
    
    shift = np.array([1.1, -0.9, 1.2, -0.8, 1.3, -0.7, 1.4, -0.6, 1.5, -0.5,
                     1.6, -0.4, 1.7, -0.3, 1.8, -0.2, 1.9, -0.1, 2.0, 0.0])
    
    # Matriz de rotación 20D
    rotation_matrix = np.random.randn(20, 20)
    q, r = np.linalg.qr(rotation_matrix)
    rotation_matrix = q

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedPowellSum.rotation_matrix @ (x - ShiftedRotatedPowellSum.shift)
        
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)):
            # Exponente que varía con la dimensión
            exponent = 2 + i/5
            # Factor de peso que aumenta con la dimensión
            weight = 1 + 0.05*i
            sum_term += weight * np.abs(shifted_rotated_x[i])**exponent
        
        # Término de acoplamiento no lineal
        coupling = 0.0
        for i in range(len(shifted_rotated_x)-1):
            coupling += np.abs(shifted_rotated_x[i])**(1.5) * np.abs(shifted_rotated_x[i+1])**(1.5)
        
        return sum_term + 0.1*coupling - 234.5678   


class ShiftedRotatedDolan: ## OKK
    name = 'Shifted Rotated Dolan'
    optimal = -331
    dimension = 20
    type = 'multimodal'
    bounds = [
        (-100, 100) for _ in range(20)
    ]
    
    shift = np.array([5.1, -4.7, 5.4, -4.4, 5.7, -4.1, 6.0, -3.8, 6.3, -3.5,
                     6.6, -3.2, 6.9, -2.9, 7.2, -2.6, 7.5, -2.3, 7.8, -2.0])
    
    # Matriz de rotación 20D
    rotation_matrix = np.random.randn(20, 20)
    q, r = np.linalg.qr(rotation_matrix)
    rotation_matrix = q

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedDolan.rotation_matrix @ (x - ShiftedRotatedDolan.shift)
        
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)-1):
            # Factor de peso que aumenta con la dimensión
            weight = 1 + 0.05*i
            # Términos base de Dolan modificados
            x_i = shifted_rotated_x[i]
            x_next = shifted_rotated_x[i+1]
            
            term1 = (x_i + 1.5*x_next - 2)**2
            term2 = (x_i + x_next - 1)**2
            term3 = np.abs(x_i * x_next + 1)
            
            sum_term += weight * (term1 + term2 + term3)
        
        # Término de acoplamiento adicional
        coupling = np.sum([
            np.sin(np.pi * shifted_rotated_x[i] / (i+1)) *
            np.cos(np.pi * shifted_rotated_x[i+1] / (i+1))
            for i in range(len(shifted_rotated_x)-1)
        ])
        
        return sum_term + 0.1*coupling - 345.6789

class ShiftedRotatedPrice02: ## OKK
    name = 'Shifted Rotated Price 02'
    optimal = 948.35
    dimension = 25
    type = 'multimodal'
    bounds = [
         (-10, 10) for _ in range(25)
    ]
    
    shift = np.array([2.1, -1.7, 2.4, -1.9, 2.2, -1.8, 2.3, -1.6, 2.5, -1.5,
                     2.0, -1.4, 2.6, -1.3, 1.9, -1.2, 2.7, -1.1, 1.8, -1.0,
                     2.8, -0.9, 1.7, -0.8, 2.9])
    
    # Matriz de rotación 25D
    rotation_matrix = np.eye(25)
    for i in range(24):
        for j in range(i+1, 25):
            theta = np.pi/4 + (i*j*np.pi/50)
            R = np.eye(25)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedPrice02.rotation_matrix @ (x - ShiftedRotatedPrice02.shift)
        
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)-1):
            # Factor de peso que aumenta con la dimensión
            weight = 1 + 0.1*i
            # Términos base de Price 02 modificados
            term = 1 + np.sin(np.sin(shifted_rotated_x[i]))**2 + \
                  0.1 * np.exp(((shifted_rotated_x[i]**2 + shifted_rotated_x[i+1]**2)/50))
            sum_term += weight * term
        
        # Término de acoplamiento no lineal
        coupling = np.sum([
            np.sin(shifted_rotated_x[i]**2) * np.cos(shifted_rotated_x[i+1]**2)
            for i in range(len(shifted_rotated_x)-1)
        ])
        
        return sum_term + 0.1*coupling + 891.2345
    
    
class ShiftedRotatedCorana: ## OKK
    name = 'Shifted Rotated Corana'
    optimal = 456.7891
    dimension = 20
    type = 'multimodal'
    bounds = [
         (-500, 500) for _ in range(20)
    ]
    
    shift = np.array([20.1, -19.7, 20.4, -19.4, 20.7, -19.1, 21.0, -18.8, 21.3, -18.5,
                     21.6, -18.2, 21.9, -17.9, 22.2, -17.6, 22.5, -17.3, 22.8, -17.0])
    
    # Matriz de rotación 20D
    rotation_matrix = np.eye(20)
    for i in range(19):
        for j in range(i+1, 20):
            theta = np.pi/4 + (i*j*np.pi/40)
            R = np.eye(20)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedCorana.rotation_matrix @ (x - ShiftedRotatedCorana.shift)
        
        d = np.array([1, 1000, 10, 100] * 5)  # Extendido para 20D
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)):
            z = 0.2 * np.floor(np.abs(shifted_rotated_x[i]/0.2) + 0.5) * np.sign(shifted_rotated_x[i])
            if np.abs(shifted_rotated_x[i] - z) < 0.2:
                sum_term += 0.15 * (z - 0.05*np.sign(z))**2 * d[i]
            else:
                sum_term += d[i] * shifted_rotated_x[i]**2
        
        # Término de acoplamiento
        coupling = np.sum([
            np.sin(shifted_rotated_x[i] * shifted_rotated_x[i+1] / 100)
            for i in range(len(shifted_rotated_x)-1)
        ])
        
        return sum_term + 0.1*coupling + 456.7891

class ShiftedRotatedCsendes: ## OKK
    name = 'Shifted Rotated Csendes'
    optimal = -789.1234
    dimension = 25
    type = 'multimodal'
    bounds = [
        (-2, 2) for _ in range(25)
    ]
    
    shift = np.array([0.21, -0.17, 0.24, -0.19, 0.22, -0.18, 0.23, -0.16, 0.25, -0.15,
                     0.20, -0.14, 0.26, -0.13, 0.19, -0.12, 0.27, -0.11, 0.18, -0.10,
                     0.28, -0.09, 0.17, -0.08, 0.29])
    
    # Matriz de rotación 25D
    rotation_matrix = np.random.randn(25, 25)
    q, r = np.linalg.qr(rotation_matrix)
    rotation_matrix = q

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedCsendes.rotation_matrix @ (x - ShiftedRotatedCsendes.shift)
        
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)):
            # Factor de peso que aumenta con la dimensión
            weight = 1 + 0.05*i
            if shifted_rotated_x[i] != 0:
                term = weight * (shifted_rotated_x[i]**6 * (2 + np.sin(1/shifted_rotated_x[i])))
                sum_term += term
        
        # Término de acoplamiento no lineal
        coupling = np.sum([
            np.sin(shifted_rotated_x[i]**2 + shifted_rotated_x[i+1]**2)
            for i in range(len(shifted_rotated_x)-1)
        ])
        
        return sum_term + 0.1*coupling - 789.1234

class ShiftedRotatedDamavandi: ## OKK
    name = 'Shifted Rotated Damavandi'
    optimal = 266.4165
    dimension = 15
    type = 'multimodal'
    bounds = [
        (-10, 10) for _ in range(15)
    ]
    
    shift = np.array([2.1, -1.7, 2.4, -1.9, 2.2, -1.8, 2.3, -1.6, 2.5, -1.5,
                     2.0, -1.4, 2.6, -1.3, 1.9])
    
    # Matriz de rotación 15D
    rotation_matrix = np.eye(15)
    for i in range(14):
        for j in range(i+1, 15):
            theta = np.pi/3 + (i*j*np.pi/30)
            R = np.eye(15)
            R[i,i] = np.cos(theta)
            R[i,j] = -np.sin(theta)
            R[j,i] = np.sin(theta)
            R[j,j] = np.cos(theta)
            rotation_matrix = R @ rotation_matrix

    @staticmethod
    def function(x):
        shifted_rotated_x = ShiftedRotatedDamavandi.rotation_matrix @ (x - ShiftedRotatedDamavandi.shift)
        
        sum_term = 0.0
        for i in range(len(shifted_rotated_x)-1):
            # Términos base de Damavandi modificados
            x_i = shifted_rotated_x[i]
            x_next = shifted_rotated_x[i+1]
            
            num = np.sin(np.pi*(x_i - 2))*np.sin(np.pi*(x_next - 2))
            den = (np.pi**2)*(x_i - 2)*(x_next - 2)
            
            term1 = (1 - np.abs(num/den if den != 0 else 1))**5
            term2 = 2 + (x_i - 7)**2 + 2*(x_next - 7)**2
            
            sum_term += term1 * term2
        
        # Término de acoplamiento trigonométrico
        coupling = np.sum([
            np.cos(shifted_rotated_x[i] * np.pi/2) * np.sin(shifted_rotated_x[i+1] * np.pi/2)
            for i in range(len(shifted_rotated_x)-1)
        ])
        
        return sum_term + 0.1*coupling + 234.5678


