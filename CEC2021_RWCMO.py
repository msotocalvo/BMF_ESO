import numpy as np

class CEC2021_RWCMO_00:
    name = "Three-bar Truss Design Problem"
    optimal =  2.6389584338 * 10e+02
    bounds = [(0, 1), (0, 1)]  # Bounds for x1 and x2
    sigma = 2  # Assuming a stress limit for the sake of example

    @staticmethod
    def function(x):
        x1, x2 = x

        # Function objective: minimize the total length assumed to be weight
        f = 100 * (x2 + 2*np.sqrt(2*x1))

        # Constraints
        g1 = (x2 / (2 * x1*x1) + np.sqrt(2*x1**2))*2 - CEC2021_RWCMO_00.sigma
        g2 = (x2 + np.sqrt(2*x1) / (2 * x1 * x2 + np.sqrt(2*x1**2)))*2 - CEC2021_RWCMO_00.sigma
        g3 = 1 / (x1 + np.sqrt(2*x2))*2 - CEC2021_RWCMO_00.sigma

        # Penalize constraint violations
        penalty = 0
        for g in [g1, g2, g3]:
            if g > 0:
                penalty += 1e6 * g**2

        return f + penalty

class CEC2021_RWCMO_0:
    ### Implementation cheked#########
    name = "Tension/Compression Spring Design"
    optimal =  1.2665232788 * 10e-02  # El valor óptimo no se especifica
    bounds = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]

    @staticmethod
    def function(x):
        x1, x2, x3 = x

        # Función objetivo
        f = (x3 + 2) * x2 * x1**2

        # Penalizaciones por las restricciones
        penalty = 0
        g1 = 1 - (x2**3 * x3) / (71785 * x1**4)
        g2 = (4 * x2**2 - x1 * x2) / (12566 * (x2 * x1**3 - x1**4)) + 1 / (5108 * x1**2) - 1
        g3 = 1 - 140.45 * x1 / (x2**2 * x3)
        g4 = (x1 + x2) / 1.5 - 1

        # Agregando penalizaciones si las restricciones no se cumplen
        for g in [g1, g2, g3, g4]:
            if g > 0:
                penalty += 1e6 * g**2

        return f + penalty

class CEC2021_RWCMO_1:
    ### Implementation Checked #####
    name = "Pressure Vessel Problem"
    optimal = 5.8853327736 * 10e+03
    bounds = [(1, 99), (1, 99), (10, 200), (10, 200)]  # Límites para x1, x2, x3, x4

    @staticmethod
    def function(x):
        # Redondeo y cálculo de z1 y z2 según x1 y x2
        x1, x2, x3, x4 = np.round(x[0]), np.round(x[1]), x[2], x[3]
        z1 = 0.0625 * x1
        z2 = 0.0625 * x2

        # Función objetivo
        f1 = 1.7781 * z2 * x3**2 + 0.6224 * z1 * x3 * x4 + 3.1661 * z1**2 * x4 + 19.84 * z1**2 * x3
        objective = f1

        # Penalizaciones por no cumplir restricciones
        penalty = 0
        if not (0.00954 * x3 <= z2):
            penalty += 1e6 * (0.00954 * x3 - z2)
        if not (0.0193 * x3 <= z1):
            penalty += 1e6 * (0.0193 * x3 - z1)
        if not (x4 <= 240):
            penalty += 1e6 * (x4 - 240)
        if not (-np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3 <= -1296000):
            penalty += 1e6 * (-1296000 + np.pi * x3**2 * x4 + (4/3) * np.pi * x3**3)

        # Sumar la penalización a la función objetivo
        return objective + penalty


class CEC2021_RWCMO_2:
    name = "Vibrating Platform"
    optimal = 0 
    bounds = [(0.05, 0.5), (0.2, 0.5), (0.2, 0.6), (0.35, 0.5), (3, 6)]
    
    @staticmethod
    def function(x):
        rho1 = 100
        rho2 = 2770
        rho3 = 7780
        E1 = 1.6
        E2 = 70
        E3 = 200
        c1 = 500
        c2 = 1500
        c3 = 800
        
        d1, d2, d3, b, L = x
        
        mu = 2 * b * (rho1 * d1 + rho2 * (d2 - d1) + rho3 * (d3 - d2))
        EI = (2 * b / 3) * (E1 * d1**3 + E2 * (d2**3 - d1**3) + E3 * (d3**3 - d2**3))
        
        f1 = (-np.pi) / (2 * L)**2 * (np.abs(EI / mu))**0.5
        f2 = 2 * b * L * (c1 * d1 + c2 * (d2 - d1) + c3 * (d3 - d2))
        
        objective = f1 + f2 # Asumiendo que queremos minimizar la suma de f1 y f2
        
        # Calculo de penalizaciones para las restricciones
        penalties = [
            np.maximum(0, mu * L - 2800),
            np.maximum(0, d1 - d2),
            np.maximum(0, d2 - d1 - 0.15),
            np.maximum(0, d2 - d3),
            np.maximum(0, d3 - d2 - 0.01)
        ]
        
        penalty = np.sum(penalties) * 1e60 # Factor de penalización ajustable
        
        return objective + penalty # Incorporando las penalizaciones

class CEC2021_RWCMO_3:
    name = "Two Bar Truss Design Problem"
    optimal = 0  # Si el óptimo es desconocido
    bounds = [(10e-5, 100), (10e-5, 100), (1, 3)]  # Suponiendo límites para x1, x2, x3
    
    @staticmethod
    def function(x):
        x1, x2, x3 = x
        
        f1 = x1 * (16 + x3**2)**0.5 + x2 * (1 + x3**2)**0.5
        f2 = (20 * (16 + x3**2)**0.5) / (x3 * x1)
        
        # Combinación de las funciones objetivo
        objective = f1 + f2
        
        # Calculo de penalizaciones para las restricciones
        penalties = [
            np.maximum(0, f1 - 0.1),
            np.maximum(0, f2 - 1e5),
            np.maximum(0, (80 * (1 + x3**2)**0.5) / (x3 * x2) - 1e5)
        ]
        
        penalty = np.sum(penalties) * 1e60  # Factor de penalización ajustable
        
        return objective + penalty  # Incorporando las penalizaciones  
    
class CEC2021_RWCMO_4:
    #### Implementation checked ######
    name = "Welded Beam Design Problem"
    optimal = 1.6702177263
    bounds = [(0.125, 5), (0.1, 10), (0.1, 10), (0.125, 5)]  # Suponiendo límites para x1, x2, x3, x4
    
    @staticmethod
    def function(x):
        x1, x2, x3, x4 = x
        P = 6000
        L = 14
        E = 30e6
        G = 12e6
        tmax = 13600
        sigmax = 30000
        
        Pc = (4.013 * E * np.sqrt(x3**2 + x4**6) / 36) / L**2 * (1 - x3 / (2 * L) * np.sqrt(E / (4 * G)))
        sigma = (6 * P * L) / (x4 * x3**2)
        J = 2 * np.sqrt(2) * x1 * x2 * (x2**2 / 12 + ((x1 + x3) / 2)**2)
        R = np.sqrt(x2**2 / 4 + ((x1 + x3) / 2)**2)
        M = P * (L + x2 / 2)
        tho1 = P / (np.sqrt(2) * x1 * x2)
        tho2 = M * R / J
        tho = np.sqrt(tho1**2 + 2 * tho1 * tho2 * x2 / (2 * R) + tho2**2)
        
        f1 = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14 + x2)
        # f2 = (4 * P * L**3) / (E * x4 * x3**3)
        
        objective = f1 
        
        penalties = [
            np.maximum(0, tho - tmax),
            np.maximum(0, sigma - sigmax),
            np.maximum(0, x1 - x4),
            np.maximum(0, P - Pc)
        ]
        
        penalty = np.sum(penalties) * 1e60  # Factor de penalización ajustable
        
        return objective + penalty  # Incorporando las penalizaciones   

class CEC2021_RWCMO_5:
    name = "Disc Brake Design Problem"
    optimal = 0
    bounds = [(55,80),(75,110),(1000,3000),(11,20)]
            
    @staticmethod
    def function(x):
        # Desempaquetar variables
        x1, x2, x3, x4 = x

        # Calcular las funciones objetivo
        f1 = 4.9e-5 * (x2**2 - x1**2) * (x4 - 1)
        f2 = 9.82e6 * (x2**2 - x1**2) / (x3 * x4 * (x2**3 - x1**3))

        # Combinamos las funciones objetivo para simplificar
        objective = f1 + f2

        # Calculo de penalizaciones para las restricciones
        penalty = 0
        penalties = [
            np.maximum(0, 20 - (x2 - x1)),
            np.maximum(0, x3 / (3.14 * (x2**2 - x1**2)) - 0.4),
            np.maximum(0, 2.22e-3 * x3 * (x2**3 - x1**3) / ((x2**2 - x1**2)**2) - 1),
            np.maximum(0, 900 - 2.66e-2 * x3 * x4 * (x2**3 - x1**3) / (x2**2 - x1**2))
        ]
        for g in penalties:
            penalty += g * 10e60  # Factor de penalización ajustable
        
        return objective + penalty  # Incorporando las penalizaciones   

class CEC2021_RWCMO_6:
    name = "Speed Reducer Design Problem"
    optimal = 0  # Valor optimo desconocido
    bounds = [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.8, 8.3), (2.9, 3.9), (5, 5.5)]
    
    @staticmethod
    def function(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        # Asegurar que x3 es un entero si es necesario
        x3 = round(x3)

        f1 = 0.7854 * x1 * x2**2 * (10 * x3**2 / 3 + 14.933 * x3 - 43.0934) - 1.508 * x1 * (x6**2 + x7**2) + 7.477 * (x6**3 + x7**3) + 0.7854 * (x4 * x6**2 + x5 * x7**2)
        f2 = np.sqrt((745 * x4 / (x2 * x3))**2 + 1.69e7) / (0.1 * x6**3)

        objective = f1 + f2  # Combinación de los objetivos si es necesario

        # Calculo de penalizaciones para las restricciones
        penalties = [
            np.maximum(0, 1 / (x1 * x2**2 * x3) - 1 / 27),
            np.maximum(0, 1 / (x1 * x2**2 * x3**2) - 1 / 397.5),
            np.maximum(0, x4**3 / (x2 * x3 * x6**4) - 1 / 1.93),
            np.maximum(0, x5**3 / (x2 * x3 * x7**4) - 1 / 1.93),
            np.maximum(0, x2 * x3 - 40),
            np.maximum(0, x1 / x2 - 12),
            np.maximum(0, -x1 / x2 + 5),
            np.maximum(0, 1.9 - x4 + 1.5 * x6),
            np.maximum(0, 1.9 - x5 + 1.1 * x7),
            np.maximum(0, f2 - 1300),
            np.maximum(0, np.sqrt((745 * x5 / (x2 * x3))**2 + 1.575e8) / (0.1 * x7**3) - 850)
        ]

        penalty = np.sum(penalties) * 1e60  # Factor de penalización ajustable

        return objective + penalty  # Incorporando las penalizaciones   

class CEC2021_RWCMO_7:
    name = "Gear Train Design Problem"
    optimal = 0  
    # Asumiendo límites para x1 a x4, ajustar según sea necesario
    bounds = [(12, 60), (12, 60), (12, 60), (12, 60)]
    
    @staticmethod
    def function(x):
        x1, x2, x3, x4 = x
        # Calcula la función objetivo y la restricción
        f1 = np.abs(6.931 - x3 * x4 / (x1 * x2))
        f2 = np.max(x, axis=0)

        objective = f1 + f2  # Aquí, combinamos f1 y f2 para simplificar, ajusta según tu criterio

        # Calculo de penalizaciones para la restricción
        penalty = np.maximum(0, f1 / 6.931 - 0.5) * 1e6  # Factor de penalización ajustable

        return objective + penalty  # Incorporando la penalización     

class CEC2021_RWCMO_8:
    name = "Car Side Impact Design Problem"
    optimal = 0  # Valor optimo desconocido
    bounds = [(0.5, 1.5), (0.45, 1.35), (0.5, 1.5), (0.5, 1.5), (0.875, 2.625), (0.4, 1.2), (0.4, 1.2)]
    
    @staticmethod
    def function(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        # Calcula las métricas VMBP y VFD basadas en las variables de diseño
        VMBP = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        VFD = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6

        # Calcula la función objetivo y las restricciones
        f1 = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1e-5 * x6 + 2.73 * x7
        f2 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        f3 = 0.5 * (VMBP + VFD)

        objective = f1 + f2 + f3  # Aquí, combinamos f1, f2 y f3 para simplificar, ajusta según tu criterio

        # Calculo de penalizaciones para las restricciones
        penalty = 0
        penalties = [
            np.maximum(0, -1 + 1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3),
            # Agrega todas las demás restricciones aquí
            # Por ejemplo:
            np.maximum(0, VMBP - 9.9)
        ]
        for g in penalties:
            penalty += g * 1e60  # Factor de penalización ajustable

        return objective + penalty  # Incorporando la penalización    

class CEC2021_RWCMO_9:
    name = "Four Bar Plane Truss"
    optimal = 0
    
    F = 10  # Carga aplicada
    E = 2e5  # Módulo de elasticidad
    L = 200  # Longitud de las barras
    sig = 10  # Tensión máxima permitida
    bounds = [((F/sig), 3*(F/sig)),(np.sqrt(2)*(F/sig),3*(F/sig)),(np.sqrt(2)*(F/sig),3*(F/sig)),(F/sig, 3*(F/sig))]
    
    
    @staticmethod
    def function(x):
        F = 10  # Carga aplicada
        E = 2e5  # Módulo de elasticidad
        L = 200  # Longitud de las barras
        sig = 10  # Tensión máxima permitida
        
        x1, x2, x3, x4 = x
        f1 = L * (2 * x1 + np.sqrt(2) * x2 + np.sqrt(2) * x3 + x4)
        f2 = F * L / E * (2 / x1 + 2 * np.sqrt(2) / x2 - 2 * np.sqrt(2) / x3 + 2 / x4)
        
        # Combinación de los objetivos en una sola métrica
        objective = f1 + f2

        # Calculo de penalizaciones para las restricciones
        penalty = 0
        penalties = [
            np.abs(L * (2 * x1 + np.sqrt(2) * x2 + np.sqrt(2) * x3 + x4) - (sig * E / F)),
            np.abs(F * L / E * (2 / x1 + 2 * np.sqrt(2) / x2 - 2 * np.sqrt(2) / x3 + 2 / x4) - sig)
        ]
        for g in penalties:
            if g > 0:  # Penalización solo si la restricción no se cumple
                penalty += g * 10e6  # Factor de penalización ajustable
        
        return objective + penalty  # Incorporando las penalizaciones    

class CEC2021_RWCMO_10:
    name = "Two Bar Plane Truss"
    optimal = 0  # Si el valor óptimo es desconocido o no aplicable
    bounds = [(0.1, 2), (0.5, 2.5)]  # Ejemplo de límites, ajustar según sea necesario

    @staticmethod
    def function(x):
        x1, x2 = x
        rho = 0.283
        h = 100
        P = 104
        E = 3e7
        rho0 = 2e4

        f1 = 2 * rho * h * x2 * np.sqrt(1 + x1**2)
        f2 = rho * h * (1 + x1**2)**1.5 * (1 + x1**4)**0.5 / (2 * np.sqrt(2) * E * x1**2 * x2)

        # Calculo de penalizaciones para las restricciones
        penalty = 0
        g1 = P * (1 + x1) * (1 + x1**2)**0.5 / (2 * np.sqrt(2) * x1 * x2) - rho0
        g2 = P * (-x1 + 1) * (1 + x1**2)**0.5 / (2 * np.sqrt(2) * x1 * x2) - rho0
        penalties = [g1, g2]

        for g in penalties:
            if g > 0:  # Penalización solo si la restricción no se cumple
                penalty += g * 1e60  # Factor de penalización ajustable

        return f1 + f2 + penalty  # Incorporando la penalización      

class CEC2021_RWCMO_11:
    name = "Water Resource Management Problem"
    optimal = 0
    bounds = [(0.01, 0.45), (0.01, 0.1), (0.01, 0.1)]
    
    @staticmethod        
    def function(x):
        # Extracción de variables
        x1, x2, x3 = x
        
        # Cálculo de los objetivos individuales
        f1 = 106780.37 * (x2 + x3) + 61704.67
        f2 = 3000 * x1
        f3 = 305700 * 2289 * x2 / (0.06 * 2289)**0.65
        f4 = 250 * 2289 * np.exp(-39.75*x2 + 9.9*x3 + 2.74)
        f5 = 25 * (1.39 / (x1*x2) + 4940*x3 - 80)
        
        # Combinación de los objetivos en una sola métrica
        objective = f1 + f2 + f3 + f4 + f5

        # Calculo de penalizaciones para las restricciones
        penalties = [
            np.maximum(0, 1 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08)),
            np.maximum(0, 1 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986)),
            np.maximum(0, 50000 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02)),
            np.maximum(0, 16000 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71)),
            np.maximum(0, 10000 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04)),
            np.maximum(0, 2000 - (0.417 * x1 * x2 + 1721.26 * x3 - 136.54)),
            np.maximum(0, 550 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48))
        ]
        penalty = np.sum(penalties) * 1e60
        
        return objective + penalty

class CEC2021_RWCMO_12:
    name = "Simply Supported I-beam Design"
    optimal = 0  
    bounds = [(10, 80), (10, 50), (0.9, 5), (0.9, 5)]  # Ejemplo de límites, ajustar según sea necesario

    @staticmethod
    def function(x):
        x1, x2, x3, x4 = x
        P = 600
        L = 200
        E = 2e4

        f1 = 2 * x2 * x4 + x3 * (x1 - 2 * x4)
        f2 = P * L**3 / (48 * E * (x3 * ((x1 - 2 * x4)**3) + 2 * x2 * x4 * (4 * x4**2 + 3 * x1 * (x1 - 2 * x4))) / 12)

        # Calculo de penalizaciones para las restricciones
        penalty = 0
        g1 = -16 + 180000 * x1 / (x3 * ((x1 - 2 * x4)**3) + 2 * x2 * x4 * (4 * x4**2 + 3 * x1 * (x1 - 2 * x4))) + 15000 * x2 / ((x1 - 2 * x4) * x3**3 + 2 * x4 * x2**3)

        if g1 > 0:  # Penalización solo si la restricción no se cumple
            penalty += g1 * 1e60  # Factor de penalización ajustable

        return f1 + f2 + penalty  # Incorporando la penalización      

class CEC2021_RWCMO_13:
    name = "Gear Box Design"
    optimal = 0 
    bounds = [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.8, 8.3), (2.9, 3.9), (5.0, 5.5)]  # Ejemplo de límites

    @staticmethod
    def function(x):
        x1, x2, x3, x4, x5, x6, x7 = map(float, x)  # Asegura que x3 sea tratado como float para operaciones
        x3 = round(x3)  # Redondeo de x3 a entero si es necesario

        # Calcula los objetivos
        f1 = 0.7854 * x1 * x2**2 * (10 * x3**2 / 3 + 14.933 * x3 - 43.0934) - 1.508 * x1 * (x6**2 + x7**2) + 7.477 * (x6**3 + x7**3) + 0.7854 * (x4 * x6**2 + x5 * x7**2)
        f2 = (745 * x4 / (x2 * x3))**2 + 1.69e7
        f2 = np.sqrt(f2) / (0.1 * x6**3)
        f3 = (745 * x5 / (x2 * x3))**2 + 1.575e8
        f3 = np.sqrt(f3) / (0.1 * x7**3)

        # Combinación de objetivos y penalizaciones
        objective = f1 + f2 + f3   # Solo se toma f1 como referencia para la función objetivo simplificada

        # Calculo de penalizaciones para restricciones no satisfechas
        penalties = [
            max(0, 1 / (x1 * x2**2 * x3) - 1 / 27),
            max(0, 1 / (x1 * x2**2 * x3**2) - 1 / 397.5),
            max(0, x4**3 / (x2 * x3 * x6**4) - 1 / 1.93),
            # Añadir el resto de las penalizaciones de restricciones aquí...
        ]
        penalty = sum(penalties) * 1e60  # Ajuste de penalización según sea necesario

        return objective + penalty    
    
class CEC2021_RWCMO_14:
    #### Implementation checked #############
    name = "Multiple-disk clutch brake design"
    optimal = 2.3524245790 * 10e-01
    bounds = [(60,80), (90,110), (1,3), (0,100), (2,9)]
                
    @staticmethod
    def function(x):
        # Parámetros del problema dados
        Mf = 3; Ms = 40; Iz = 55; n = 250; Tmax = 15; s = 1.5; delta = 0.5
        Vsrmax = 10; rho = 0.0000078; pmax = 1; mu = 0.6; Lmax = 30; delR = 20
        
        # Calcula variables derivadas y objetivos
        Rsr = 2./3.*(x[1]**3 - x[0]**3) / (x[1]**2 * x[0]**2)
        Vsr = np.pi * Rsr * n / 30
        A = np.pi * (x[1]**2 - x[0]**2)
        Prz = x[3] / A
        w = np.pi * n / 30
        Mh = 2/3 * mu * x[3] * x[4] * (x[1]**3 - x[0]**3) / (x[1]**2 - x[0]**2)
        T = Iz * w / (Mh + Mf)
        f1 = np.pi * (x[1]**2 - x[0]**2) * x[2] * (x[4] + 1) * rho        
        
        # Combinamos las funciones objetivo para simplificar
        objective = f1 

        # Calculo de penalizaciones para las restricciones
        penalties = [
            np.maximum(0, -x[1] + x[0] + delR),
            np.maximum(0, (x[4] + 1) * (x[2] + delta) - Lmax),
            np.maximum(0, Prz - pmax),
            np.maximum(0, Prz * Vsr - pmax * Vsrmax),
            np.maximum(0, Vsr - Vsrmax),
            np.maximum(0, T - Tmax),
            np.maximum(0, s * Ms - Mh),
            np.maximum(0, -T)
        ]

        penalty = np.sum(penalties) * 10e60  # Factor de penalización ajustable

        return objective + penalty  # Incorporando las penalizaciones    

class CEC2021_RWCMO_15:
    name = "Spring Design Problem"
    optimal = 0 
    bounds = [(1, 70), (0.6, 3), (1, 42)]  # Ejemplo de límites

    @staticmethod
    def function(x):
        x1, x2 = x[:2]
        d_index = min(max(int(round(x[2])), 1), 42) - 1  # Ajuste del índice dentro de los límites permitidos
        d = [
            0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014,
            0.015, 0.0162, 0.0173, 0.018, 0.020, 0.023, 0.025,
            0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063,
            0.072, 0.080, 0.092, 0.0105, 0.120, 0.135, 0.148,
            0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263,
            0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.500
        ]
        x3 = d[d_index]
        cf = (4 * x2 / x3 - 1) / (4 * x2 / x3 - 4) + 0.615 * x3 / x2
        K = (11.5 * 10**6 * x3**4) / (8 * x1 * x2**3)
        lf = 1000 / K + 1.05 * (x1 + 2) * x3
        sigp = 300 / K
        
        # Objetivo y penalizaciones
        f1 = (np.pi**2 * x2 * x3**2 * (x1 + 2)) / 4
        f2 = (8000 * cf * x2) / (np.pi * x3**3)

        objective = f1 + f2 # Para simplificar, solo consideramos f1 como el objetivo aquí

        # Calculo de penalizaciones para restricciones no satisfechas
        penalties = [
            max(0, (8000 * cf * x2) / (np.pi * x3**3) - 189000),
            max(0, lf - 14),
            max(0, 0.2 - x3),
            max(0, x2 - 3),
            max(0, 3 - x2 / x3),
            max(0, sigp - 6),
            max(0, sigp + 700 / K + 1.05 * (x1 + 2) * x3 - lf),
            max(0, 1.25 - 700 / K)
        ]
        penalty = sum(penalties) * 1e60  # Ajuste de penalización según sea necesario

        return objective + penalty    
    
class CEC2021_RWCMO_16:
    name = "Cantilever Beam Design Problem"
    optimal = 0  
    bounds = [(0.01, 0.05), (0.20, 1)]  
    
    @staticmethod
    def function(x):
        # Asegurarse que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)
        
        # Extraer variables
        x1, x2 = x[:, 0], x[:, 1]
        
        # Constantes del problema
        P = 1
        E = 207000000
        Sy = 300000
        delta_max = 0.005
        rho = 7800
        
        # Funciones objetivo
        f1 = 0.25 * rho * np.pi * x2 * x1**2
        f2 = (64 * P * x2**3) / (3 * E * np.pi * x1**4)
        
        # Combinar las funciones objetivo
        objective = f1 + f2
        
        # Calculo de penalizaciones para las restricciones
        penalty = 0
        penalties = [
            np.maximum(0, -Sy + (32 * P * x2) / (np.pi * x1**3)),
            np.maximum(0, -delta_max + (64 * P * x2**3) / (3 * E * np.pi * x1**4))
        ]
        for g in penalties:
            if g > 0:  # Penalización solo si la restricción no se cumple
                penalty += g * 10e6  # Factor de penalización ajustable
        
        return objective + penalty  # Retornar objetivo más penalizaciones


class CEC2021_RWCMO_17:
    name = "Bulk Carriers Design Problem"
    optimal = 0  
    bounds = [(150, 274.32),  
              (20, 32.31),   
              (13, 25),    
              (10, 11.71),     
              (14, 18),    
              (0.63, 0.75)] 
    
    @staticmethod
    def function(x):
        x = np.atleast_2d(x)
        
        # Extraer variables
        L, B, D, T, V_k, C_B = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        
        # Cálculos intermedios
        a = 4977.06 * C_B**2 - 8105.61 * C_B + 4456.51
        b = -10847.2 * C_B**2 + 12817 * C_B - 6960.32
        F_n = 0.5144 / (9.8065 * L)**0.5
        P = ((1.025 * L * B * T * C_B)**(2/3) * V_k**3) / (a + b * F_n)
        
        W_s = 0.034 * L**1.7 * B**0.6 * D**0.4 * C_B**0.5
        W_o = L**0.8 * B**0.6 * D**0.3 * C_B**0.1
        W_m = 0.17 * P**0.9
        ls = W_s + W_o + W_m
        
        D_wt = 1.025 * L * B * T * C_B - ls
        F_c = 0.19 * 24 * P / 1000 + 0.2
        D_cwt = D_wt - F_c * ((5000 * V_k) / 24 + 5) - 2 * D_wt**0.5
        R_trp = 350 / ((5000 * V_k) / 24 + 2 * (D_cwt / 8000 + 0.5))
        ac = D_cwt * R_trp
        S_d = 5000 * V_k / 24
        
        C_c = 0.2 * 1.3 * (2000 * W_s**0.85 + 3500 * W_o + 2400 * P**0.8)
        C_r = 40000 * D_wt**0.3
        C_v = (1.05 * 100 * F_c * S_d + 6.3 * D_wt**0.8) * R_trp
        
        # Funciones objetivo
        f1 = (C_c + C_r + C_v) / ac
        f2 = ls
        f3 = -ac
        
        # Combinar las funciones objetivo
        objective = f1 + f2 + f3
        
        # Penalizaciones por restricciones no cumplidas
        penalties = np.maximum(0, np.column_stack([
            L/B - 6,
            15 - L/D,
            19 - L/T,
            0.45 * D_wt**0.31 - T,
            0.7 * D + 0.7 - T,
            0.32 - F_n,
            0.53 * T + ((0.085 * C_B - 0.002) * B**2) / (T * C_B) - (1 + 0.52 * D) - 0.07 * B,
            D_wt - 3000,
            500000 - D_wt
        ]))
        penalty = np.sum(penalties, axis=1) * 10e6  # Factor de penalización ajustable
        
        return objective + penalty[:, np.newaxis]  # Incorporando las penalizaciones


class CEC2021_RWCMO_18:
    name = "Front Rail Design Problem"
    optimal = 0 
    bounds = [(136, 146),  
              (58, 68),   
              (1.4, 2.2)]  
    
    @staticmethod
    def function(x):
        x = np.atleast_2d(x)
        
        # Extraer variables
        hh, w, t = x[:, 0], x[:, 1], x[:, 2]
        
        # Cálculos de las funciones
        Ea = 14496.5
        Fa = 234.9
        E = -70973.4 + 958.656 * w + 614.173 * hh - 3.827 * w * hh + 57.023 * w * t + 63.274 * hh * t \
            - 3.582 * w**2 - 1.4842 * hh**2 - 1890.174 * t**2
        F = 111.854 - 20.210 * w + 7.560 * hh - 0.025 * w * hh + 2.731 * w * t - 1.479 * hh * t \
            + 0.165 * w**2
        
        # Funciones objetivo
        f1 = Ea / E
        f2 = F / Fa
        
        # Combinar las funciones objetivo
        objective = f1 +f2
        
        # Penalizaciones por restricciones no cumplidas
        penalties = np.maximum(0, np.column_stack([
            (hh - 136) * (146 - hh),
            (w - 58) * (66 - w),
            (t - 1.4) * (2.2 - t)
        ]))
        penalty = np.sum(penalties, axis=1) * 10e6  # Factor de penalización ajustable
        
        return objective + penalty[:, np.newaxis]  # Incorporando las penalizaciones


class CEC2021_RWCMO_20:
    name = "Hydro-static Thrust Bearing Design Problem"
    optimal =  1.6254428092 * 10e+03
    bounds = [(1, 16), (1, 16), (10e-6, 16*10e-6), (1, 16)]  

    @staticmethod
    def function(x):
        # Asegurarse que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)
        
        # Extraer variables
        R, Ro, mu, Q = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        # Constantes del problema
        gamma = 0.0307; C = 0.5; n = -3.55; C1 = 10.04
        Ws = 101000; Pmax = 1000; delTmax = 50; hmin = 0.001
        gg = 386.4; N = 750

        # Cálculos de la función objetivo y restricciones
        P = (np.log10(np.log10(8.122*1e6*mu + 0.8)) - C1) / n
        delT = 2 * (10**P - 560)
        Ef = 9336 * Q * gamma * C * delT
        h = ((2 * np.pi * N / 60)**2 * 2 * np.pi * mu / Ef * (R**4 / 4 - Ro**4 / 4) - 1e-5)
        Po = (6 * mu * Q / (np.pi * h**3)) * np.log(R / Ro)
        W = np.pi * Po / 2 * (R**2 - Ro**2) / (np.log(R / Ro) - 1e-5)

        # Funciones objetivo
        f1 = (Q * Po / 0.7) + Ef
        # f2 = gamma / (gg * Po) * (Q / (2 * np.pi * R * h))
        
        # Combinar las funciones objetivo
        objective = f1 

        # Penalizaciones por restricciones no cumplidas
        penalties = np.column_stack([
            np.maximum(0, 1000 - Po),
            np.maximum(0, W - Ws),
            np.maximum(0, 5000 - (W / (np.pi * (R**2 - Ro**2)))),
            np.maximum(0, 50 - Po),
            np.maximum(0, 0.001 - (0.0307 / 386.4*Po) * (Q / 2*np.pi*R*h)),
            np.maximum(0, R - Ro),
            np.maximum(0, h - 0.001)
        ])
        penalty = np.sum(penalties, axis=1) * 10e60  # Factor de penalización ajustable

        return objective + penalty[:, np.newaxis]  # Incorporando las penalizaciones

class CEC2021_RWCMO_21:
    name = "Crash Energy Management for High-Speed Train"
    optimal = 0  
    bounds = [(1.3, 1.7), (2.5, 3.5), (1.3, 1.7), (1.3, 1.7), (1.3, 1.7), (1.3, 1.7)]  

    @staticmethod
    def function(x):
        # Asegurarse que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)

        # Constantes y cálculos de la función objetivo
        x1, x2, x3, x4, x5, x6 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        # Función objetivo 1
        f1 = 1.3667145844797 - 0.00904459793976106 * x1 - 0.0016193573938033 * x2 - \
             0.00758531275221425 * x3 - 0.00440727360327102 * x4 - 0.00572216860791644 * x5 - \
             0.00936039926190721 * x6 + np.sum([
                 2.62510221107328e-6 * (x1**2),
                 4.92982681358861e-7 * (x2**2),
                 2.25524989067108e-6 * (x3**2),
                 1.84605439400301e-6 * (x4**2),
                 2.17175358243416e-6 * (x5**2),
                 3.90158043948054e-6 * (x6**2),
                 # Pares interactivos y términos de interacción
             ], axis=0)

        # Función objetivo 2
        f2 = -1.19896668942683 + 3.04107017009774 * x1 + 1.23535701600191 * x2 + \
             2.13882039381528 * x3 + 2.33495178382303 * x4 + 2.68632494801975 * x5 + \
             3.43918953617606 * x6 - np.sum([
                 7.89144544980703e-4 * (x1**2),
                 2.06085185698215e-4 * (x2**2),
                 7.15269900037858e-4 * (x3**2),
                 7.8449237573837e-4 * (x4**2),
                 9.31396896237177e-4 * (x5**2),
                 1.40826531972195e-3 * (x6**2),
                 # Pares interactivos y términos de interacción
             ], axis=0)

        # Combinar las funciones objetivo
        objective = f1 + f2

        # Penalizaciones por restricciones no cumplidas
        penalties = np.column_stack([
            np.maximum(0, f1 - 5),
            np.maximum(0, -f1),
            np.maximum(0, f2 - 28),
            np.maximum(0, -f2)
        ])
        penalty = np.sum(penalties, axis=1) * 10e6  # Factor de penalización ajustable

        return objective + penalty[:, np.newaxis]  # Incorporando las penalizaciones
   
        
####### Chemical problems

class CEC2021_RWCMO_22:
    ####Implementation checked####
    name = "Haverly's Pooling Problem"
    optimal = -4.0000560000 * 10e+02
    bounds = [(0, 100), (0, 200), (0, 100), (0, 100), (0, 100), (0, 100), (0, 200), (0, 100), (0, 200)]

    @staticmethod
    def function(x):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        
        # Minimize the negative of the objective since we need to maximize it
        f = -(9*x1 + 15*x2 - 6*x3 - 16*x4 - 10*(x5 + x6))

        # Penalization for constraints
        penalty = 0
        h1 = x7 + x8 - x4 - x3
        h2 = x1 - x5 - x7
        h3 = x2 - x6 - x8
        h4 = x9*x7 + x9*x8 - 3*x3 - x4
        g1 = x9*x7 + 2*x5 - 2.5*x1
        g2 = x9*x8 + 2*x6 - 1.5*x2

        # Adding penalties if constraints are violated
        for h in [h1, h2, h3, h4]:
            if abs(h) > 0.001:  # Small tolerance for equality constraints
                penalty += 1e6 * abs(h)**2
        for g in [g1, g2]:
            if g > 0:
                penalty += 1e6 * g**2

        return f + penalty

class CEC2021_RWCMO_23:
    #### Implementation cheked####
    name = "Reactor Network Design"
    optimal = -3.8826043623 * 10e-01
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0.00001, 16), (0.00001, 16)]
    k1 = 0.09755988
    k2 = 0.99 * k1
    k3 = 0.0391908
    k4 = 0.9 * k3

    @staticmethod
    def function(x):
        x1, x2, x3, x4, x5, x6 = x
        # Minimize the negative of the objective since we need to maximize it
        f = -x4  # Negating because scipy minimizes
        
        # Define constraints and their penalizations
        h1 = abs(CEC2021_RWCMO_23.k1 * x5 * x2 + x1 - 1)
        h2 = abs(CEC2021_RWCMO_23.k3 * x5 * x3 + x3 + x1 - 1)
        h3 = abs(CEC2021_RWCMO_23.k2 * x6 * x2 - x1 + x2)
        h4 = abs(CEC2021_RWCMO_23.k4 * x6 * x4 + x2 - x1 + x4 - x3)
        g1 = x5**0.5 + x6**0.5 - 4

        # Penalization for constraint violations
        penalty = 0
        for h in [h1, h2, h3, h4]:
            if h > 0.001:  # Tolerance
                penalty += 1e6 * h**2
        if g1 > 0:
            penalty += 1e6 * g1**2

        return f + penalty
    
class CEC2021_RWCMO_24:
    name = "Heat Exchanger Network Design (case 1)"
    optimal =  1.8931162966 * 10e+02
    bounds = [(0, 10), (0, 200), (0, 100), (0, 200), (1000, 2000000), (0, 600), (100, 600), (100, 600), (100, 900)]

    @staticmethod
    def function(x):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        # Función objetivo según la definición del problema
        f1 = 35 * x1**0.6 + 35 * x2**0.6
        
        # Restricciones transformadas en penalizaciones
        h1 = np.maximum(0, np.abs(200 * x1 * x4 - x3)) * 1e6
        h2 = np.maximum(0, np.abs(200 * x2 * x6 - x5)) * 1e6
        h3 = np.maximum(0, np.abs(x3 - 10000 * (x7 - 100))) * 1e6
        h4 = np.maximum(0, np.abs(x5 - 10000 * (300 - x7))) * 1e6
        h5 = np.maximum(0, np.abs(x3 - 10000 * (600 - x8))) * 1e6
        h6 = np.maximum(0, np.abs(x5 - 10000 * (900 - x9))) * 1e6
        h7 = np.maximum(0, np.abs(x4 * np.log(x8 - 100) - x4 * np.log(600 - x7) - x8 + x7 + 500)) * 1e6
        h8 = np.maximum(0, np.abs(x6 * np.log(x9 - x7) - x6 * np.log(600) - x9 + x7 + 600)) * 1e6
        
        # Total de penalizaciones
        penalty = h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8
        
        return f1 + penalty  # Función objetivo total con penalizaciones

class CEC2021_RWCMO_24_1:
    name = "Heat Exchanger Network Design (case 2)"
    optimal =  7.0490369540 * 10e+03
    bounds = [(104, 81900), (104, 113100), (104, 205000), (0, 0.05074), (0, 0.05074), (0, 0.05074), (100, 300), (100, 300), (100, 300), (100, 300), (100, 300)]

    @staticmethod
    def function(x):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
        # Función objetivo
        f1 = (x1 / (120 * x4))**0.6 + (x2 / (80 * x5))**0.6 + (x3 / (40 * x6))**0.6
        
        # Penalizaciones por las restricciones de igualdad
        h1 = np.maximum(0, np.abs(x1 - 104 * (x7 - 100))) * 1e6
        h2 = np.maximum(0, np.abs(x2 - 104 * (x8 - x7))) * 1e6
        h3 = np.maximum(0, np.abs(x3 - 104 * (500 - x8))) * 1e6
        h4 = np.maximum(0, np.abs(x1 - 104 * (300 - x9))) * 1e6
        h5 = np.maximum(0, np.abs(x2 - 104 * (400 - x10))) * 1e6
        h6 = np.maximum(0, np.abs(x3 - 104 * (600 - x11))) * 1e6
        h7 = np.maximum(0, np.abs(x4 * np.log(x9 - 100) - x4 * np.log(300 - x7) - x9 + x7 + 400)) * 1e6
        h8 = np.maximum(0, np.abs(x5 * np.log(x10 - x7) - x5 * np.log(400 - x8) - x10 + x7 - x8 + 400)) * 1e6
        h9 = np.maximum(0, np.abs(x6 * np.log(x11 - x8) - x6 * np.log(100) - x11 + x8 + 100)) * 1e6

        # Total de penalizaciones
        penalty = h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8 + h9
        
        return f1 + penalty  # Función objetivo total con penalizaciones
        
##### Process design and syntesis problems
class CEC2021_RWCMO_25:
    #### Implementation checked######
    name = "Process Synthesis Problem"
    optimal = 2  
    bounds = [(0, 100),(0, 100),(0, 100),(0, 1),(0, 1),(0, 1),(0, 1)]

    @staticmethod
    def function(x):
        x1, x2, x3, x4, x5, x6, x7 = x

        # Función objetivo
        f = (1-x4)**2 + (1-x5)**2 + (1-x6)**2 + np.log(1+x7) + (1-x1)**2

        # Penalizaciones por las restricciones de desigualdad e igualdad
        penalty = 0
        constraints = [
            (x1 + x2 + x3 + x4 + x5 + x6 - 5, 0),  # g1 >= 13
            (x6**3 + x1**2 + x2**2 +x3**2 - 5.5, 0),  # g2 <= 20
            (x1 + x4 - 1.2, 0),  # g3 = 0
            (x2 +x5 -1.8, 0),  # g4 = 0
            (x3 +x6- 2.5, 0),  # g5 <= 0
            (x1 + x7 -1.2, 0),  # g6 = 0
            (x5**2 + x2**2 - 1.64, 0),  # g7 = 0
            (x6**2 + x3**2 - 4.25, 0),  # g8 = 0
            (x5**2 + x3**2 - 4.64, 0)  # g9 = 0
        ]
        
        for (g, h) in constraints:
            if g < 0:
                penalty += 1e6 * (g**2)
            elif h != 0:
                penalty += 1e6 * (h**2)

        return f + penalty
    
class CEC2021_RWCMO_26:
    name = "Process Synthesis and Design Problems"
    optimal = 2.5576545740
    bounds = [(0.2, 1), (-2.22554, -1),(0, 1)]  

    @staticmethod
    def function(x):
        # Asegurarse que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)
        x = np.column_stack([x[:, 0], x[:, 1], np.round(x[:, 2])])

        # Desempaquetar variables
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        # Calcular la función objetivo
        f1 = -x3 + x2 + 2 * x1
        f2 = -x1**2 - x2 + x1 * x3

        # Combinar las funciones objetivo en un arreglo
        objective = f1 + f2

        # Calcular las restricciones
        h = np.zeros((x.shape[0], 1))
        g = np.zeros((x.shape[0], 1))
        h[:, 0] = -2 * np.exp(-x2) + x1
        g[:, 0] = x2 - x1 + x3

        # Penalizaciones para las restricciones h y g, si son positivas
        penalties_h = np.maximum(0, h)
        penalties_g = np.maximum(0, g)
        penalty = np.sum(penalties_h + penalties_g, axis=1) * 10e6  # Penalización si las restricciones no se cumplen

        return objective + penalty[:, np.newaxis]  # Incorporando penalizaciones

class CEC2021_RWCMO_27:
    name = "Process Flow Sheeting Problem"
    optimal = 1.0765430833
    # Definir los límites apropiados para cada variable basado en el conocimiento del problema
    bounds = [(0.2, 1), (-2.22554, -1), (0, 1)]
   
    @staticmethod
    def function(x):
        # Asegurarse que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)

        # Desempaquetar variables
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        
        # Calcular la función objetivo
        f1 = -0.7 * x3 + 0.8 + 5 * (0.5 - x1)**2
        f2 = x1 - x3
        
        # Combinar las funciones objetivo en un arreglo
        objective = f1 + f2

        # Calcular las restricciones
        g = np.zeros((x.shape[0], 3))
        g[:, 0] = -(np.exp(x1 - 0.2) + x2)
        g[:, 1] = x2 + 1.1 * x3 - 1
        g[:, 2] = x1 - x3 - 0.2
        
        # Penalizaciones para las restricciones g, si g > 0
        penalties = np.maximum(0, g)
        penalty = np.sum(penalties, axis=1) * 10e6  # Penalización si las restricciones no se cumplen

        return objective + penalty[:, np.newaxis]  # Incorporando penalizaciones


class CEC2021_RWCMO_28:
    name = "Two Reactor Problem"
    optimal = 9.9238463653 * 10e+01
    bounds = [(0, 100), (0, 100),(0, 100),(0, 100),(0, 100),(0, 100),(0, 1), (0, 1)]  
    
    @staticmethod
    def function(x):
        # Asegurarse que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)
        
        # Extraer y redondear variables donde sea necesario
        x1, x2, v1, v2, y1, y2, x_ = x[:, 0], x[:, 1], x[:, 2], x[:, 3], np.round(x[:, 4]), np.round(x[:, 5]), x[:, 6]
        
        # Calcular funciones intermediarias
        z1 = 0.9 * (1 - np.exp(-0.5 * v1)) * x1
        z2 = 0.8 * (1 - np.exp(-0.4 * v2)) * x2
        
        # Funciones objetivo
        f1 = 7.5 * y1 + 5.5 * y2 + 7 * v1 + 6 * v2 + 5 * x_
        f2 = x1 + x2
        
        # Combinar las funciones objetivo
        objective = f1 + f2
        
        # Calculo de penalizaciones para las restricciones
        penalty = 0
        penalties = [
            np.maximum(0, y1 + y2 - 1),
            np.maximum(0, z1 + z2 - 10),
            np.maximum(0, x1 + x2 - x_),
            np.maximum(0, z1 * y1 + z2 * y2 - 10),
            np.maximum(0, v1 - 10 * y1 - 1e-6),
            np.maximum(0, v2 - 10 * y2),
            np.maximum(0, x1 - 20 * y1),
            np.maximum(0, x2 - 20 * y2)
        ]
        for g in penalties:
            if g > 0:  # Penalización solo si la restricción no se cumple
                penalty += g * 10e60  # Factor de penalización ajustable
        
        return objective + penalty  # Retornar objetivo más penalizaciones
    
class CEC2021_RWCMO_29:
    name = "Process Synthesis Problem"
    optimal = 2.9248305537  
    bounds = [(0, 100), (0, 100),(0, 100),
              (0, 1),(0, 1),(0, 1),(0, 1)]   
    
    @staticmethod
    def function(x):
        # Asegurar que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)
        
        # Extraer variables, redondear donde sea necesario
        x1, x2, x3, x4, x5, x6, x7 = x[:, 0], x[:, 1], x[:, 2], np.round(x[:, 3]), np.round(x[:, 4]), np.round(x[:, 5]), np.round(x[:, 6])
        
        # Funciones objetivo
        f1 = (1 - x4) ** 2 + (1 - x5) ** 2 + (1 - x6) ** 2 - np.log(np.abs(1 + x7) + 1e-6)
        f2 = (1 - x1) ** 2 + (2 - x2) ** 2 + (3 - x3) ** 2
        
        # Combinar las dos funciones objetivo
        objective = f1 + f2
        
        # Calculo de penalizaciones para las restricciones
        penalty = 0
        penalties = [
            np.maximum(0, x1 + x2 + x3 + x4 + x5 + x6 - 5),
            np.maximum(0, x6 ** 3 + x1 ** 2 + x2 ** 2 + x3 ** 2 - 5.5),
            np.maximum(0, x1 + x4 - 1.2),
            np.maximum(0, x2 + x5 - 1.8),
            np.maximum(0, x3 + x6 - 2.5),
            np.maximum(0, x1 + x7 - 1.2),
            np.maximum(0, x5 ** 2 + x2 ** 2 - 1.64),
            np.maximum(0, x6 ** 2 + x3 ** 2 - 4.25),
            np.maximum(0, x5 ** 2 + x3 ** 2 - 4.64)
        ]
        for g in penalties:
            if g > 0:  # Penalización solo si la restricción no se cumple
                penalty += g * 10e60 # Factor de penalización ajustable
        
        return objective + penalty  # Retornar objetivo más penalizaciones

#### Power System problems 

class CEC2021_RWCMO_30:
    name = "SOPWM for 3-level Inverters"
    optimal = 3.8029250566 * 10e-02
    bounds = [(0, np.pi / 2)] * 10  

    @staticmethod
    def function(x):
        # Constants
        ks = np.array([5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97], dtype=float)
        N = len(x)
        si = np.array([(-1)**(i + 1) for i in range(N)], dtype=float)
        sum_k = np.sum(ks**-4)

        # Calculate numerator of the objective function
        numerator = 0
        for k in ks:
            sum_cosines = np.sum(si * np.cos(k * x))
            numerator += (k**-4 * sum_cosines)**2
        numerator = np.sqrt(numerator)

        # Objective function
        f = numerator / np.sqrt(sum_k)

        # Constraints
        penalty = 0
        m = 1  # This needs to be specified or calculated
        h1 = m - np.sum(si * np.cos(x))
        if abs(h1) > 1e-5:
            penalty += 1e6 * abs(h1)**2

        for i in range(N - 1):
            if x[i + 1] - x[i] - 1e-5 < 0:
                penalty += 1e6 * (x[i + 1] - x[i] - 1e-5)**2

        return f + penalty

class CEC2021_RWCMO_31:
    name = "SOPWM for 5-level Inverters"
    optimal = 2.1215000000 * 10e-02
    bounds = [(0, np.pi / 2) for _ in range(25)]  

    @staticmethod
    def function(x):
        # Asegurarse que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)

        # Constantes del problema
        m = 0.32
        s = np.array([1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1])
        k = np.array([5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47, 49, 53, 55, 59, 61, 65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97])
        
        x_rad = np.radians(x)  # Convertir grados a radianes
        f1 = np.zeros(x.shape[0])
        
        # Calcular la primera función objetivo
        for j in range(len(k)):
            su2 = np.sum(s * np.cos(k[j] * x_rad), axis=1)
            f1 += su2**2 / k[j]**4
        f1 = np.sqrt(f1) / np.sqrt(np.sum(1 / k**4))
        
        # Calcular la segunda función objetivo
        f2 = np.sum(s * np.cos(x_rad), axis=1) - m
        f2 = f2**2
        
        # Combinar en un arreglo para múltiples objetivos
        objective = f1 + f2

        # Calcular las restricciones
        g = np.zeros((x.shape[0], x.shape[1] - 1))
        for i in range(x.shape[1] - 1):
            g[:, i] = x[:, i] - x[:, i + 1] + 1e-6
        
        penalties = np.maximum(0, g)
        penalty = np.sum(penalties, axis=1) * 10e6  # Penalización si las restricciones no se cumplen

        return objective + penalty[:, np.newaxis]  # Incorporando penalizaciones


class CEC2021_RWCMO_32:
    name = "SOPWM for 7-level Inverters"
    optimal = 1.5164538375 * 10e-02    
    bounds = [(0, np.pi / 2) for _ in range(25)]  

    @staticmethod
    def function(x):
        # Asegurarse que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)

        # Constantes del problema
        m = 0.36
        s = np.array([1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1])
        k = np.array([5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47, 49, 53, 55, 59, 61, 65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97])
        
        x_rad = np.radians(x)  # Convertir grados a radianes
        f1 = np.zeros(x.shape[0])
        
        # Calcular la primera función objetivo
        for j in range(len(k)):
            su2 = np.sum(s * np.cos(k[j] * x_rad), axis=1)
            f1 += su2**2 / k[j]**4
        f1 = np.sqrt(f1) / np.sqrt(np.sum(1 / k**4))
        
        # Calcular la segunda función objetivo
        f2 = np.sum(s * np.cos(x_rad), axis=1) - m
        f2 = f2**2
        
        # Combinar en un arreglo para múltiples objetivos
        objective = f1 + f2

        # Calcular las restricciones
        g = np.zeros((x.shape[0], x.shape[1] - 1))
        for i in range(x.shape[1] - 1):
            g[:, i] = x[:, i] - x[:, i + 1] + 1e-6
        
        penalties = np.maximum(0, g)
        penalty = np.sum(penalties, axis=1) * 10e6  # Penalización si las restricciones no se cumplen

        return objective + penalty[:, np.newaxis]  # Incorporando penalizaciones

class CEC2021_RWCMO_33:
    name = "SOPWM for 9-level Inverters"
    optimal =  1.6787535766 * 10e-02    
    bounds = [(0, np.pi / 2) for _ in range(30)]  

    @staticmethod
    def function(x):
        # Asegurarse que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)

        # Constantes del problema
        m = 0.32
        s = np.array([1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1])
        k = np.array([5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47, 49, 53, 55, 59, 61, 65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97])
        
        x_rad = np.radians(x)  # Convertir grados a radianes
        f1 = np.zeros(x.shape[0])
        
        # Calcular la primera función objetivo
        for j in range(len(k)):
            su2 = np.sum(s * np.cos(k[j] * x_rad), axis=1)
            f1 += su2**2 / k[j]**4
        f1 = np.sqrt(f1) / np.sqrt(np.sum(1 / k**4))
        
        # Calcular la segunda función objetivo
        f2 = np.sum(s * np.cos(x_rad), axis=1) - m
        f2 = f2**2
        
        # Combinar en un arreglo para múltiples objetivos
        objective = f1 + f2

        # Calcular las restricciones
        g = np.zeros((x.shape[0], x.shape[1] - 1))
        for i in range(x.shape[1] - 1):
            g[:, i] = x[:, i] - x[:, i + 1] + 1e-6
        
        penalties = np.maximum(0, g)
        penalty = np.sum(penalties, axis=1) * 10e6  # Penalización si las restricciones no se cumplen

        return objective + penalty[:, np.newaxis]  # Incorporando penalizaciones

class CEC2021_RWCMO_34:
    name = "SOPWM for 11-level Inverters"
    optimal = 9.3118741800 * 10e-03   
    bounds = [(0, np.pi / 2) for _ in range(30)] 

    @staticmethod
    def function(x):
        # Asegurarse que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)

        # Constantes del problema
        m = 0.3333
        s = np.array([1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1])
        k = np.array([5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47, 49, 53, 55, 59, 61, 65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97])
        
        x_rad = np.radians(x)  # Convertir grados a radianes
        f1 = np.zeros(x.shape[0])
        
        # Calcular la primera función objetivo
        for j in range(len(k)):
            su2 = np.sum(s * np.cos(k[j] * x_rad), axis=1)
            f1 += su2**2 / k[j]**4
        f1 = np.sqrt(f1) / np.sqrt(np.sum(1 / k**4))
        
        # Calcular la segunda función objetivo
        f2 = np.sum(s * np.cos(x_rad), axis=1) - m
        f2 = f2**2
        
        # Combinar en un arreglo para múltiples objetivos
        objective = f1 + f2

        # Calcular las restricciones
        g = np.zeros((x.shape[0], x.shape[1] - 1))
        for i in range(x.shape[1] - 1):
            g[:, i] = x[:, i] - x[:, i + 1] + 1e-6
        
        penalties = np.maximum(0, g)
        penalty = np.sum(penalties, axis=1) * 10e6  # Penalización si las restricciones no se cumplen

        return objective + penalty[:, np.newaxis]  # Incorporando penalizaciones  

class CEC2021_RWCMO_35:
    name = "SOPWM for 13-level Inverters"
    optimal = 1.5096451396 * 10e-02   
    bounds = [(0, np.pi / 2)] * 30 

    @staticmethod
    def function(x):
        # Convertir x a array de numpy si aún no lo es
        x = np.atleast_2d(x) 
        
        m = 0.32
        s = np.array([1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1])
        k = np.array([5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47, 49, 53, 55, 59, 61, 65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97])
        
        x_rad = x * np.pi / 180  # Convertir grados a radianes
        su = np.zeros(x.shape[0])
        for j in range(len(k)):
            su2 = np.sum(s * np.cos(k[j] * x_rad), axis=1)
            su += su2 ** 2 / k[j] ** 4
        f1 = np.sqrt(su) / np.sqrt(np.sum(1 / k ** 4))
        f2 = np.sum(s * np.cos(x_rad), axis=1) - m
        f2 = f2 ** 2
        
        return f1 + f2  # Suma de las dos funciones objetivo

class CEC2021_RWCMO_36:
    name = "Optimal Sizing of Single Phase Distributed Generation"
    optimal = 0  # Valor óptimo desconocido
    bounds = [(0, 1)] * 24 + [(-1, 1)] * 4 

    @staticmethod
    def function(x):
        # Matriz de admitancia
        G = np.array([
            [2.0336507971433414e+02, -5.3956475914255392e+01, -2.7088978087868369e+01, -2.0336507971375815e+02, 5.3956475914255392e+01, 2.7088978087868369e+01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-5.3956475914255392e+01, 2.3398757254504045e+02, -8.6283054943635207e+01, 5.3956475914255392e+01, -2.3398757253928045e+02, 8.6283054943635207e+01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-2.7088978087868366e+01, -8.6283054943635207e+01, 2.2201347979167249e+02, 2.7088978087868366e+01, 8.6283054943635207e+01, -2.2201347979109650e+02, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-2.0336507971375815e+02, 5.3956475914255392e+01, 2.7088978087868369e+01, 9.6523494572590857e+02, -2.1701526742738102e+02, -1.1958512736147634e+02, -4.0673015942751630e+02, 1.0791295182851078e+02, 5.4177956175736739e+01, 0, 0, 0, -3.5513970658463421e+02, 5.5145839684614856e+01, 3.8318193097871237e+01],
            [5.3956475914255392e+01, -2.3398757253928045e+02, 8.6283054943635207e+01, -2.1701526742738102e+02, 1.0650133220777063e+03, -3.3574578008216434e+02, 1.0791295182851078e+02, -4.6797514507856090e+02, 1.7256610988727041e+02, 0, 0, 0, 5.5145839684614856e+01, -3.6305060445986504e+02, 7.6896615251258694e+01],
            [2.7088978087868366e+01, 8.6283054943635207e+01, -2.2201347979109650e+02, -1.1958512736147634e+02, -3.3574578008216429e+02, 1.0272843110747488e+03, 5.4177956175736732e+01, 1.7256610988727041e+02, -4.4402695958219300e+02, 0, 0, 0, 3.8318193097871237e+01, 7.6896615251258680e+01, -3.6124387170145928e+02],
            [0, 0, 0, -4.0673015942751630e+02, 1.0791295182851078e+02, 5.4177956175736739e+01, 8.1346031885503260e+02, -2.1582590365702157e+02, -1.0835591235147348e+02, -4.0673015942751630e+02, 1.0791295182851078e+02, 5.4177956175736739e+01, 0, 0, 0],
            [0, 0, 0, 1.0791295182851078e+02, -4.6797514507856090e+02, 1.7256610988727041e+02, -2.1582590365702157e+02, 9.3595029015712180e+02, -3.4513221977454083e+02, 1.0791295182851078e+02, -4.6797514507856090e+02, 1.7256610988727041e+02, 0, 0, 0],
            [0, 0, 0, 5.4177956175736732e+01, 1.7256610988727041e+02, -4.4402695958219300e+02, -1.0835591235147346e+02, -3.4513221977454083e+02, 8.8805391916438600e+02, 5.4177956175736732e+01, 1.7256610988727041e+02, -4.4402695958219300e+02, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -4.0673015942751630e+02, 1.0791295182851078e+02, 5.4177956175736739e+01, 4.0673015942751630e+02, -1.0791295182851078e+02, -5.4177956175736739e+01, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1.0791295182851078e+02, -4.6797514507856090e+02, 1.7256610988727041e+02, -1.0791295182851078e+02, 4.6797514507856090e+02, -1.7256610988727041e+02, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5.4177956175736732e+01, 1.7256610988727041e+02, -4.4402695958219300e+02, -5.4177956175736732e+01, -1.7256610988727041e+02, 4.4402695958219300e+02, 0, 0, 0],
            [0, 0, 0, -3.5513970658463421e+02, 5.5145839684614856e+01, 3.8318193097871237e+01, 0, 0, 0, 0, 0, 0, 3.5513970658463421e+02, -5.5145839684614856e+01, -3.8318193097871237e+01],
            [0, 0, 0, 5.5145839684614856e+01, -3.6305060445986504e+02, 7.6896615251258694e+01, 0, 0, 0, 0, 0, 0, -5.5145839684614856e+01, 3.6305060445986504e+02, -7.6896615251258694e+01],
            [0, 0, 0, 3.8318193097871237e+01, 7.6896615251258680e+01, -3.6124387170145928e+02, 0, 0, 0, 0, 0, 0, -3.8318193097871237e+01, -7.6896615251258680e+01, 3.6124387170145928e+02]
        ])

        B = np.array([
            [-3.5113004928555353e+02, 4.3573978441589354e+01, 3.1816722979531573e+01, 3.5113004928555353e+02, -4.3573978441589354e+01, -3.1816722979531573e+01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4.3573978441589354e+01, -3.5377814658607412e+02, 5.8917529889855743e+01, -4.3573978441589354e+01, 3.5377814658607412e+02, -5.8917529889855743e+01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3.1816722979531573e+01, 5.8917529889855743e+01, -3.5272127849600997e+02, -3.1816722979531573e+01, -5.8917529889855743e+01, 3.5272127849600997e+02, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3.5113004928555353e+02, -4.3573978441589354e+01, -3.1816722979531573e+01, -1.3556671476971078e+03, 1.3185933449564033e+02, 1.0092116565621727e+02, 7.0226009857110705e+02, -8.7147956883178708e+01, -6.3633445959063145e+01, 0, 0, 0, 3.0227699984044705e+02, -1.1373991708722637e+00, -5.4709967176225547e+00],
            [-4.3573978441589354e+01, 3.5377814658607412e+02, -5.8917529889855743e+01, 1.3185933449564033e+02, -1.3531674012165354e+03, 1.7296810353963031e+02, -8.7147956883178708e+01, 7.0755629317214823e+02, -1.1783505977971149e+02, 0, 0, 0, -1.1373991708722739e+00, 2.9183296145831315e+02, 3.7844861299369055e+00],
            [-3.1816722979531573e+01, -5.8917529889855743e+01, 3.5272127849600997e+02, 1.0092116565621727e+02, 1.7296810353963031e+02, -1.3538611528970828e+03, -6.3633445959063145e+01, -1.1783505977971149e+02, 7.0544255699201995e+02, 0, 0, 0, -5.4709967176225582e+00, 3.7844861299369024e+00, 2.9569731740905274e+02],
            [0, 0, 0, 7.0226009857110705e+02, -8.7147956883178708e+01, -6.3633445959063145e+01, -1.4045201971422141e+03, 1.7429591376635742e+02, 1.2726689191812629e+02, 7.0226009857110705e+02, -8.7147956883178708e+01, -6.3633445959063145e+01, 0, 0, 0],
            [0, 0, 0, -8.7147956883178708e+01, 7.0755629317214823e+02, -1.1783505977971149e+02, 1.7429591376635742e+02, -1.4151125863442965e+03, 2.3567011955942297e+02, -8.7147956883178708e+01, 7.0755629317214823e+02, -1.1783505977971149e+02, 0, 0, 0],
            [0, 0, 0, -6.3633445959063145e+01, -1.1783505977971149e+02, 7.0544255699201995e+02, 1.2726689191812629e+02, 2.3567011955942297e+02, -1.4108851139840399e+03, -6.3633445959063145e+01, -1.1783505977971149e+02, 7.0544255699201995e+02, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 7.0226009857110705e+02, -8.7147956883178708e+01, -6.3633445959063145e+01, -7.0226009857110705e+02, 8.7147956883178708e+01, 6.3633445959063145e+01, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -8.7147956883178708e+01, 7.0755629317214823e+02, -1.1783505977971149e+02, 8.7147956883178708e+01, -7.0755629317214823e+02, 1.1783505977971149e+02, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -6.3633445959063145e+01, -1.1783505977971149e+02, 7.0544255699201995e+02, -6.3633445959063145e+01, -1.1783505977971149e+02, 7.0544255699201995e+02, 0, 0, 0],
            [0, 0, 0, 3.0227699984044705e+02, -1.1373991708722637e+00, -5.4709967176225547e+00, 0, 0, 0, 0, 0, 0, -3.0227699984044705e+02, 1.1373991708722637e+00, 5.4709967176225547e+00],
            [0, 0, 0, -1.1373991708722739e+00, 2.9183296145831315e+02, 3.7844861299369055e+00, 0, 0, 0, 0, 0, 0, 1.1373991708722739e+00, -2.9183296145831315e+02, -3.7844861299369055e+00],
            [0, 0, 0, -5.4709967176225582e+00, 3.7844861299369024e+00, 2.9569731740905274e+02, 0, 0, 0, 0, 0, 0, 5.4709967176225582e+00, -3.7844861299369024e+00, -2.9569731740905274e+02]
        ])

        P = np.array([
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-3.5999999999999999e-01],
            [-2.8799999999999998e-01],
            [-4.1999999999999998e-01],
            [-5.7599999999999996e-01],
            [-4.8000000000000001e-02],
            [-4.7999999999999998e-01],
            [-4.3200000000000000e-01],
            [-2.8799999999999998e-01],
            [-3.5999999999999999e-01]
        ])

        Q = np.array([
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-2.1600000000000000e-01],
            [-1.9200000000000000e-01],
            [-2.6400000000000001e-01],
            [-4.3200000000000000e-01],
            [-3.3599999999999998e-02],
            [-2.9999999999999999e-01],
            [-2.8799999999999998e-01],
            [-1.9200000000000000e-01],
            [-2.3999999999999999e-01]
        ])

        Y = G + 1j * B

        # Inicialización de voltajes y cargas de generación
        V = np.zeros(15, dtype=complex)
        V[0:3] = [1, 1 * np.exp(1j * 4 * np.pi / 3), 1 * np.exp(1j * 2 * np.pi / 3)]
        Pdg = np.zeros(15)
        Qdg = np.zeros(15)

        # Reshape y asignación de variables de decisión
        V[3:15] = x[:12] + 1j * x[12:24]
        Pdg[[6, 14]] = x[24:26]  # Asumiendo índices 7 y 15 para Pdg
        Qdg[[6, 14]] = x[26:28]  # Asumiendo índices 7 y 15 para Qdg

        # Cálculo de corrientes y potencias
        I = Y @ V
        S = V * np.conj(I)
        Psp = np.real(S)
        Qsp = np.imag(S)
        delP = Psp - Pdg - P
        delQ = Qsp - Qdg - Q

        # Cálculo de funciones objetivo y restricciones
        f1 = np.abs(I[0] + I[1] + I[2]) + np.abs(I[0] + np.exp(1j * 4 * np.pi / 3) * I[1] + np.exp(1j * 2 * np.pi / 3) * I[2])
        f2 = np.sum(Psp)
        h = np.concatenate([delP[3:], delQ[3:]])

        # Agregando funciones objetivo
        objective = f1 + f2

        # Penalizaciones para las restricciones h, si son diferentes de cero
        penalty = np.sum(np.abs(h)) * 10e6  # Penalización si las restricciones no se cumplen

        return objective + penalty

class CEC2021_RWCMO_37:
    name = "Optimal Sizing of Single Phase Distributed Generation with reactive power support for Phase Balancing at Main Transformer/Grid and reactive Power loss"
    optimal = 0  # Valor óptimo desconocido
    bounds = [(0, 1)] * 24 + [(-1, 1)] * 4  # Límites para las variables de voltaje y generación

    @staticmethod
    def function(x):
        # Asegurar que x tiene la forma adecuada para las operaciones
        x = np.atleast_2d(x)
        G = np.array([
            [2.0336507971433414e+02, -5.3956475914255392e+01, -2.7088978087868369e+01, -2.0336507971375815e+02, 5.3956475914255392e+01, 2.7088978087868369e+01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-5.3956475914255392e+01, 2.3398757254504045e+02, -8.6283054943635207e+01, 5.3956475914255392e+01, -2.3398757253928045e+02, 8.6283054943635207e+01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-2.7088978087868366e+01, -8.6283054943635207e+01, 2.2201347979167249e+02, 2.7088978087868366e+01, 8.6283054943635207e+01, -2.2201347979109650e+02, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-2.0336507971375815e+02, 5.3956475914255392e+01, 2.7088978087868369e+01, 9.6523494572590857e+02, -2.1701526742738102e+02, -1.1958512736147634e+02, -4.0673015942751630e+02, 1.0791295182851078e+02, 5.4177956175736739e+01, 0, 0, 0, -3.5513970658463421e+02, 5.5145839684614856e+01, 3.8318193097871237e+01],
            [5.3956475914255392e+01, -2.3398757253928045e+02, 8.6283054943635207e+01, -2.1701526742738102e+02, 1.0650133220777063e+03, -3.3574578008216434e+02, 1.0791295182851078e+02, -4.6797514507856090e+02, 1.7256610988727041e+02, 0, 0, 0, 5.5145839684614856e+01, -3.6305060445986504e+02, 7.6896615251258694e+01],
            [2.7088978087868366e+01, 8.6283054943635207e+01, -2.2201347979109650e+02, -1.1958512736147634e+02, -3.3574578008216429e+02, 1.0272843110747488e+03, 5.4177956175736732e+01, 1.7256610988727041e+02, -4.4402695958219300e+02, 0, 0, 0, 3.8318193097871237e+01, 7.6896615251258680e+01, -3.6124387170145928e+02],
            [0, 0, 0, -4.0673015942751630e+02, 1.0791295182851078e+02, 5.4177956175736739e+01, 8.1346031885503260e+02, -2.1582590365702157e+02, -1.0835591235147348e+02, -4.0673015942751630e+02, 1.0791295182851078e+02, 5.4177956175736739e+01, 0, 0, 0],
            [0, 0, 0, 1.0791295182851078e+02, -4.6797514507856090e+02, 1.7256610988727041e+02, -2.1582590365702157e+02, 9.3595029015712180e+02, -3.4513221977454083e+02, 1.0791295182851078e+02, -4.6797514507856090e+02, 1.7256610988727041e+02, 0, 0, 0],
            [0, 0, 0, 5.4177956175736732e+01, 1.7256610988727041e+02, -4.4402695958219300e+02, -1.0835591235147346e+02, -3.4513221977454083e+02, 8.8805391916438600e+02, 5.4177956175736732e+01, 1.7256610988727041e+02, -4.4402695958219300e+02, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -4.0673015942751630e+02, 1.0791295182851078e+02, 5.4177956175736739e+01, 4.0673015942751630e+02, -1.0791295182851078e+02, -5.4177956175736739e+01, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1.0791295182851078e+02, -4.6797514507856090e+02, 1.7256610988727041e+02, -1.0791295182851078e+02, 4.6797514507856090e+02, -1.7256610988727041e+02, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5.4177956175736732e+01, 1.7256610988727041e+02, -4.4402695958219300e+02, -5.4177956175736732e+01, -1.7256610988727041e+02, 4.4402695958219300e+02, 0, 0, 0],
            [0, 0, 0, -3.5513970658463421e+02, 5.5145839684614856e+01, 3.8318193097871237e+01, 0, 0, 0, 0, 0, 0, 3.5513970658463421e+02, -5.5145839684614856e+01, -3.8318193097871237e+01],
            [0, 0, 0, 5.5145839684614856e+01, -3.6305060445986504e+02, 7.6896615251258694e+01, 0, 0, 0, 0, 0, 0, -5.5145839684614856e+01, 3.6305060445986504e+02, -7.6896615251258694e+01],
            [0, 0, 0, 3.8318193097871237e+01, 7.6896615251258680e+01, -3.6124387170145928e+02, 0, 0, 0, 0, 0, 0, -3.8318193097871237e+01, -7.6896615251258680e+01, 3.6124387170145928e+02]
        ])

        B = np.array([
            [-3.5113004928555353e+02, 4.3573978441589354e+01, 3.1816722979531573e+01, 3.5113004928555353e+02, -4.3573978441589354e+01, -3.1816722979531573e+01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4.3573978441589354e+01, -3.5377814658607412e+02, 5.8917529889855743e+01, -4.3573978441589354e+01, 3.5377814658607412e+02, -5.8917529889855743e+01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3.1816722979531573e+01, 5.8917529889855743e+01, -3.5272127849600997e+02, -3.1816722979531573e+01, -5.8917529889855743e+01, 3.5272127849600997e+02, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3.5113004928555353e+02, -4.3573978441589354e+01, -3.1816722979531573e+01, -1.3556671476971078e+03, 1.3185933449564033e+02, 1.0092116565621727e+02, 7.0226009857110705e+02, -8.7147956883178708e+01, -6.3633445959063145e+01, 0, 0, 0, 3.0227699984044705e+02, -1.1373991708722637e+00, -5.4709967176225547e+00],
            [-4.3573978441589354e+01, 3.5377814658607412e+02, -5.8917529889855743e+01, 1.3185933449564033e+02, -1.3531674012165354e+03, 1.7296810353963031e+02, -8.7147956883178708e+01, 7.0755629317214823e+02, -1.1783505977971149e+02, 0, 0, 0, -1.1373991708722739e+00, 2.9183296145831315e+02, 3.7844861299369055e+00],
            [-3.1816722979531573e+01, -5.8917529889855743e+01, 3.5272127849600997e+02, 1.0092116565621727e+02, 1.7296810353963031e+02, -1.3538611528970828e+03, -6.3633445959063145e+01, -1.1783505977971149e+02, 7.0544255699201995e+02, 0, 0, 0, -5.4709967176225582e+00, 3.7844861299369024e+00, 2.9569731740905274e+02],
            [0, 0, 0, 7.0226009857110705e+02, -8.7147956883178708e+01, -6.3633445959063145e+01, -1.4045201971422141e+03, 1.7429591376635742e+02, 1.2726689191812629e+02, 7.0226009857110705e+02, -8.7147956883178708e+01, -6.3633445959063145e+01, 0, 0, 0],
            [0, 0, 0, -8.7147956883178708e+01, 7.0755629317214823e+02, -1.1783505977971149e+02, 1.7429591376635742e+02, -1.4151125863442965e+03, 2.3567011955942297e+02, -8.7147956883178708e+01, 7.0755629317214823e+02, -1.1783505977971149e+02, 0, 0, 0],
            [0, 0, 0, -6.3633445959063145e+01, -1.1783505977971149e+02, 7.0544255699201995e+02, 1.2726689191812629e+02, 2.3567011955942297e+02, -1.4108851139840399e+03, -6.3633445959063145e+01, -1.1783505977971149e+02, 7.0544255699201995e+02, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 7.0226009857110705e+02, -8.7147956883178708e+01, -6.3633445959063145e+01, -7.0226009857110705e+02, 8.7147956883178708e+01, 6.3633445959063145e+01, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -8.7147956883178708e+01, 7.0755629317214823e+02, -1.1783505977971149e+02, 8.7147956883178708e+01, -7.0755629317214823e+02, 1.1783505977971149e+02, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -6.3633445959063145e+01, -1.1783505977971149e+02, 7.0544255699201995e+02, -6.3633445959063145e+01, -1.1783505977971149e+02, 7.0544255699201995e+02, 0, 0, 0],
            [0, 0, 0, 3.0227699984044705e+02, -1.1373991708722637e+00, -5.4709967176225547e+00, 0, 0, 0, 0, 0, 0, -3.0227699984044705e+02, 1.1373991708722637e+00, 5.4709967176225547e+00],
            [0, 0, 0, -1.1373991708722739e+00, 2.9183296145831315e+02, 3.7844861299369055e+00, 0, 0, 0, 0, 0, 0, 1.1373991708722739e+00, -2.9183296145831315e+02, -3.7844861299369055e+00],
            [0, 0, 0, -5.4709967176225582e+00, 3.7844861299369024e+00, 2.9569731740905274e+02, 0, 0, 0, 0, 0, 0, 5.4709967176225582e+00, -3.7844861299369024e+00, -2.9569731740905274e+02]
        ])

        P = np.array([
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-3.5999999999999999e-01],
            [-2.8799999999999998e-01],
            [-4.1999999999999998e-01],
            [-5.7599999999999996e-01],
            [-4.8000000000000001e-02],
            [-4.7999999999999998e-01],
            [-4.3200000000000000e-01],
            [-2.8799999999999998e-01],
            [-3.5999999999999999e-01]
        ])

        Q = np.array([
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-0.0000000000000000e+00],
            [-2.1600000000000000e-01],
            [-1.9200000000000000e-01],
            [-2.6400000000000001e-01],
            [-4.3200000000000000e-01],
            [-3.3599999999999998e-02],
            [-2.9999999999999999e-01],
            [-2.8799999999999998e-01],
            [-1.9200000000000000e-01],
            [-2.3999999999999999e-01]
        ])
        
        # Extraer variables
        voltage_real = x[:, :12]
        voltage_imag = x[:, 12:24]
        pg = x[:, 24:26]  # Generación activa en puntos específicos
        qg = x[:, 26:28]  # Generación reactiva en puntos específicos
                
        # Crear la matriz de admitancia compleja
        Y = G + 1j * B
        
        # Inicializar los voltajes en el sistema
        V = np.zeros(15, dtype=complex)
        V[0] = 1
        V[1] = np.exp(1j * -2 * np.pi / 3)
        V[2] = np.exp(1j * 2 * np.pi / 3)
        V[3:] = voltage_real + 1j * voltage_imag

        # Cálculo de corrientes y potencias
        I = Y.dot(V)
        S = V * np.conj(I)
        Psp = S.real
        Qsp = S.imag        

        # Desbalance de potencia
        delP = Psp - pg - P
        delQ = Qsp - qg - Q

        # Funciones objetivo y restricciones
        f1 = np.sum(np.abs(delP[3:]))  # Suma del desbalance de potencia activa
        f2 = np.sum(np.abs(delQ[3:]))  # Suma del desbalance de potencia reactiva

        # Combinar las funciones objetivo
        objective = f1 + f2

        # Retornar el objetivo más las penalizaciones por restricciones si las hubiera
        return objective

class CEC2021_RWCMO_46:
    name = "Optimal Power flow (Minimization of Fuel Cost, voltage deviation, active and reactive power loss)"
    optimal = 0  # Valor óptimo desconocido    
    bounds = [(-1, 1)] * 26 + [(0, 10)] * 4  # Ajustar según las necesidades reales de los límites de voltaje y generación

    @staticmethod
    def function(x):
        # Matrices de admitancia
        G = np.array([
            [6.0250290557682238, -4.9991316007980346, 0.0, 0.0, -1.0258974549701889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-4.9991316007980346, 9.521323610814779, -1.1350191923073958, -1.6860331506149431, -1.7011396670944048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.1350191923073958, 3.1209949022329564, -1.9859757099255606, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.6860331506149431, -1.9859757099255606, 10.512989522036175, -6.840980661495671, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0258974549701889, -1.7011396670944048, 0.0, -6.840980661495671, 9.568017783560265, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.579923407466223, 0.0, 0.0, 0.0, 0.0, -1.9550285631772606, -1.525967440450974, -3.0989274038379877, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.3260550394673585, -3.9020495524474277, 0.0, 0.0, 0.0, -1.424005487019931],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.9020495524474277, 5.782934306147827, -1.8808847537003996, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.9550285631772606, 0.0, 0.0, 0.0, -1.8808847537003996, 3.8359133168776602, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.525967440450974, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0149920272728927, -2.4890245868219187, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -3.0989274038379877, 0.0, 0.0, 0.0, 0.0, 0.0, -2.4890245868219187, 6.7249461484662332, -1.1369941578063267],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.424005487019931, 0.0, 0.0, 0.0, -1.1369941578063267, 2.560999644826258]
        ])
        
        B = np.array([
            [-19.447070205514382, 15.263086523179553, 0.0, 0.0, 4.234983682334831, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [15.263086523179553, -30.272115398779064, 4.781863151757718, 5.115838325872083, 5.193927397969713, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.781863151757718, -9.82238012935164, 5.068816977593921, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 5.115838325872083, 5.068816977593921, -38.6541712076078, 21.57855398169159, 0.0, 4.889512660317341, 0.0, 1.8554995578159004, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.234983682334831, 5.193927397969713, 0.0, 21.57855398169159, -35.533639456044824, 4.257445335253384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.257445335253384, -17.34073280991911, 0.0, 0.0, 0.0, 0.0, 4.094074344240442, 3.1759639650294, 6.102755448193116, 0.0],
            [0.0, 0.0, 0.0, 4.889512660317341, 0.0, 0.0, -19.549005948264654, 5.676979846721544, 9.09008271975275, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.676979846721544, -5.676979846721544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.8554995578159004, 0.0, 0.0, 9.09008271975275, 0.0, -24.092506375267877, 10.365394127060915, 0.0, 0.0, 0.0, 3.0290504569306034],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.365394127060915, -14.768337876521436, 4.402943749460521, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 4.094074344240442, 0.0, 0.0, 0.0, 4.402943749460521, -8.497018093700962, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.1759639650294, 0.0, 0.0, 0.0, 0.0, 0.0, -5.427938591201612, 2.251974626172212, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.102755448193116, 0.0, 0.0, 0.0, 0.0, 0.0, 2.251974626172212, -10.66969354947068, 2.314963475105352],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0290504569306034, 0.0, 0.0, 0.0, 2.314963475105352, -5.344013932035955]
        ])

        P = np.array([
            [0.0],
            [0.217],
            [0.942],
            [0.478],
            [0.076],
            [0.112],
            [0.0],
            [0.0],
            [0.295],
            [0.09],
            [0.035],
            [0.061],
            [0.135],
            [0.149]
        ])

        Q = np.array([
            [0.0],
            [0.127],
            [0.19],
            [-0.039],
            [0.016],
            [0.075],
            [0.0],
            [0.0],
            [0.166],
            [0.058],
            [0.018],
            [0.016],
            [0.058],
            [0.05]
        ])

        # Inicialización de voltajes
        V = np.zeros(14, dtype=complex)
        V[0] = 1  # Nodo slack con voltaje establecido en 1 + j0

        # Configurar voltajes según las variables de decisión
        V[1:] = x[:13] + 1j * x[13:26]

        # Generación de potencia en nodos específicos
        Pg = np.zeros(14)
        Pg_indices = [1, 2, 5, 7]  # nodos con generación activa
        Pg[Pg_indices] = x[26:30]

        Qg = np.zeros(14)
        Qg_indices = [1, 2, 5, 7]  # nodos con generación reactiva
        Qg[Qg_indices] = x[30:34]

        # Matriz de admitancia compleja
        Y = G + 1j * B

        # Cálculo de corrientes y potencias
        I = Y.dot(V)
        S = V * np.conj(I)
        Psp = np.real(S)
        Qsp = np.imag(S)

        delP = Psp - Pg + P
        delQ = Qsp - Qg + Q

        # Funciones de costo de combustible y otros objetivos
        fuel_cost = np.sum([0.02 * p**2 + 1.75 * p + 2 for p in Pg[Pg_indices]])  # Costo cuadrático de generación
        voltage_deviation = np.sum((1 - np.abs(V))**2)  # Desviación cuadrática del voltaje del ideal |V|=1
        power_loss = np.sum(Psp) + np.sum(Qsp)  # Pérdidas totales de potencia

        # Penalización por balance de potencia no satisfecho
        penalty = np.sum(np.abs(delP) + np.abs(delQ)) * 1000  # Penalización por desbalance de potencia

        return fuel_cost + voltage_deviation + power_loss + penalty          
class CEC2021_RWCMO_50:
    name = "Power Distribution System Planning"
    optimal = 0
    bounds = [(0, 1200) for _ in range(6)]
    
    @staticmethod
    def function(x):
        # Power Distribution System Planning
        PD = 1200  # Demanda de potencia total
        B = np.array([
            [140, 17, 15, 19, 26, 22],
            [17, 60, 13, 16, 15, 20],
            [15, 13, 65, 17, 24, 19],
            [19, 16, 17, 71, 30, 25],
            [26, 15, 24, 30, 69, 32],
            [22, 20, 19, 25, 32, 85]
        ]) * 10**-6
        a = np.array([756.7988, 451.3251, 1243.5311, 1049.9977, 1356.6592, 1658.5696])
        b = np.array([38.5390, 46.1591, 38.3055, 40.3965, 38.2704, 36.3278])
        c = np.array([0.15247, 0.10587, 0.03546, 0.02803, 0.01799, 0.02111])
        alpha = np.array([13.8593, 13.8593, 40.2669, 40.2669, 42.8955, 42.8955])
        beta = np.array([0.32767, 0.32767, -0.54551, -0.54551, -0.51116, -0.51116])
        gamma = np.array([0.00419, 0.00419, 0.00683, 0.00683, 0.00461, 0.00461])
        P = x.reshape(-1, 6)  
        PL = np.zeros(P.shape[0])
        for i in range(6):
            for j in range(6):
                PL += P[:, i] * B[i, j] * P[:, j]
        cost = np.sum(a + b * P + c * P**2, axis=1)
        emissions = np.sum(alpha + beta * P + gamma * P**2, axis=1)
        
        # Cálculo de la penalización por la restricción de demanda de potencia total
        total_power_violation = np.maximum(0, np.sum(P, axis=1) - PD - PL)
        penalty = np.sum(total_power_violation) * 10e60  # El factor de penalización puede ajustarse según sea necesario

        # Combinando coste, emisiones y penalización
        objective_with_penalty = cost + emissions + penalty
        return objective_with_penalty    
        
class CEC2021_RWCMO_51:    
    name = "Microgrid generation dispatch"
    optimal = 0
    bounds = [(10, 200) for _ in range(96)]
        
    def function(position):
                
        # Convertir la posición (vector de decisiones) en la potencia de cada generador
        power = position.reshape((24, 4))  # 24 horas, 4 generadores en un nodo

        # Cargar la demanda horaria del sistema
        demand = np.array([
            50, 45, 40, 35, 35, 40, 45, 50,  # Off-peak hours (1-8)
            55, 60, 65, 70, 200, 300, 390, 380,  # Mixed hours with peak demand (9-16)
            370, 300, 250, 200, 180, 160, 140, 120  # Decreasing demand towards off-peak (17-24)
        ])

        # Reserva requerida, 20% sobre la demanda
        required_reserve = demand * 0.2

        # Costo de generación (simplificado a un costo lineal por MW)
        cost_per_mw = np.array([20, 21, 19, 22] * 24).reshape((24, 4))

        # Costos de combustible por MWh para cada generador
        fuel_costs = np.array([5, 4.5, 6, 5.5] * 24).reshape((24, 4))

        # Costos de mantenimiento por MWh para cada generador
        maintenance_costs = np.array([2, 1.5, 2.5, 1.8] * 24).reshape((24, 4))

        # Cálculo del costo total incluyendo costos de combustible y mantenimiento
        total_cost = np.sum(power * (cost_per_mw + fuel_costs + maintenance_costs))

        # Restricciones
        # 1. Balance de energía con reserva estática
        balance_penalty = np.sum((np.sum(power, axis=1) - (demand + required_reserve))**2)

        # 2. Límites de potencia de generación
        power_limits = [(10, 100)] * 4  # Cada generador tiene un mínimo y un máximo de generación en MW
        limit_penalty = np.sum(np.clip(power - np.array(power_limits)[:, 1], 0, None) + np.clip(np.array(power_limits)[:, 0] - power, 0, None))

        # 3. Ramp-up y Ramp-down (20 MW/h como ejemplo)
        ramp_up_down_penalty = np.sum(np.clip(np.diff(power, axis=0) - 20, 0, None) + np.clip(-20 - np.diff(power, axis=0), 0, None))

        # 4. Tiempo mínimo de operación (asumiendo 3 horas como mínimo)
        min_operation_penalty = 0  # Simplificación, asumiendo manejo en la lógica de asignación de la variable "position"

        # Función de penalización agregada
        penalty = balance_penalty + limit_penalty + ramp_up_down_penalty + min_operation_penalty

        # Función objetivo: minimizar el costo total y las penalizaciones
        return total_cost + penalty
    
class CEC2021_RWCMO_52:    
    name = "Knapsack Problem"
    optimal = 0  # Valor óptimo desconocido
    bounds = [(0, 1)] * 50  
       
        
    @staticmethod
    def __init__(self, values, weights, capacity):
        self.values = values
        self.weights = weights
        self.capacity = capacity

    def function(self, x):
        total_value = np.dot(self.values, x)
        total_weight = np.dot(self.weights, x)
        if total_weight <= self.capacity:
            return total_value  # Buscamos maximizar el valor
        else:
            # Penalización por exceder la capacidad
            return total_value - (total_weight - self.capacity) * 10  # Penalización proporcional al exceso de peso
    
class BeamOptimization:
    name = "40-Bar Beam Optimization"
    optimal = 0  # Valor óptimo desconocido, se busca minimizar la deflexión   
    bounds = [(0.05, 0.5)] * 40 + [(1, 5)] * 40  # 40 diámetros y 40 longitudes

    @staticmethod
    def function(x):
        # Propiedades de los materiales (se pueden extender o ajustar)
        densities = [7800, 2770, 8900]  # acero, aluminio, cobre
        elastic_moduli = [200, 70, 110]  # E para acero, aluminio, cobre
        cost_factors = [0.5, 0.3, 0.7]  # factores de costo relativo

        # Unpacking de variables, la mitad son diámetros y la otra mitad longitudes
        diameters = x[:20]
        lengths = x[20:]

        total_deflection = 0
        total_weight = 0
        total_cost = 0

        for i in range(20):
            diameter = diameters[i]
            length = lengths[i]
            material_index = i % 3  # Asignar material de manera cíclica

            area = np.pi * (diameter / 2) ** 2
            inertia = np.pi * (diameter / 2) ** 4 / 4
            density = densities[material_index]
            E = elastic_moduli[material_index]
            weight = density * area * length
            cost = cost_factors[material_index] * weight

            # Deflexión para una carga distribuida w, usando la fórmula simplificada
            w = 100  # carga por unidad de longitud
            deflection = 5 * w * length**4 / (384 * E * inertia)
            total_deflection += deflection
            total_weight += weight
            total_cost += cost

        # Penalizaciones por peso y coste excesivos
        weight_penalty = np.maximum(0, total_weight - 3000) * 1e60
        cost_penalty = np.maximum(0, total_cost - 1000) * 1e60

        return total_deflection + weight_penalty + cost_penalty  # Función objetivo   

class WaterResourceManagement:
    name = "Dam System Optimization"
    optimal = 0  
    bounds = [(15, 90), (20, 85), (18, 80)]  # Volúmenes de agua almacenados en cada represa (en millones de m^3)
    
    min_levels = [15, 20, 18]
    max_levels = [90, 85, 80]
    seasonal_demand = np.array([[30, 45, 25], [35, 50, 30], [40, 55, 35]])  # Demanda estacional
    seasonal_evaporation_rate = np.array([0.1, 0.05, 0.03])  # Tasa de evaporación estacional
    inflows = np.array([[5, 7, 6], [4, 6, 5], [3, 5, 4]])  # Afluentes mensuales a cada represa

    @staticmethod
    def function(x, month=0):
        cost_per_m3 = 30  # Costo base por metro cúbico de tratamiento y bombeo de agua
        penalty_for_shortage = 10e6  # Costo de la penalización por no satisfacer la demanda
        value_per_m3_saved = 10  # Valor por metro cúbico de agua ahorrado
        penalty_for_level_violation = 1e6  # Penalización por violar los límites de niveles de agua
        
        volumes = np.array(x) + WaterResourceManagement.inflows[month]  # Incluir afluentes
        total_cost = 0
        total_value_saved = 0

        for i in range(3):
            evaporated_volume = volumes[i] * WaterResourceManagement.seasonal_evaporation_rate[month]
            volumes[i] -= evaporated_volume
            
            water_released = min(volumes[i], WaterResourceManagement.seasonal_demand[month][i])
            volumes[i] -= water_released
            
            shortage = WaterResourceManagement.seasonal_demand[month][i] - water_released
            total_cost += shortage * penalty_for_shortage
            total_cost += water_released * cost_per_m3
            total_value_saved += volumes[i] * value_per_m3_saved
            
            if volumes[i] < WaterResourceManagement.min_levels[i]:
                total_cost += penalty_for_level_violation
            if volumes[i] > WaterResourceManagement.max_levels[i]:
                total_cost += penalty_for_level_violation

        return total_cost - total_value_saved

class CEC2021_RWCMO_54:
    name = "Wind Farm Layout Problem"
    optimal = -6260.7  # Specified in the document, corrected for magnitude
    bounds = [(0, 100), (0, 100)] * 5  # Example for 5 turbines

    @staticmethod
    def energy_output(xi, yi, wind_speed, wind_direction, frequency):
        # Placeholder values for the constants
        vr = 10  # Rated wind speed
        c1 = 0.5  # Coefficient for the wind speed distribution
        k = 2     # Shape factor for the Weibull distribution
        theta_n = np.pi / 4  # Wind direction in radians

        # Energy calculation based on the provided formula (simplified)
        part1 = np.exp(-np.power((wind_speed - vr) / c1, k))
        part2 = np.exp(-np.power(vr / (c1 * ((theta_n + np.pi / 2) / 2)), k))

        # Calculate energy considering frequency of the wind
        energy = frequency * (part1 - part2)

        return energy

    @staticmethod
    def function(x):
        x = np.atleast_2d(x)
        positions = x[:, :10].reshape(-1, 2)  # Positions of the turbines

        # Example wind parameters for each turbine
        wind_speeds = [12, 12, 12, 12, 12]
        wind_directions = [np.pi / 4] * 5
        frequency_intervals = [1] * 5

        # Calculate energy for each turbine
        energies = [
            CEC2021_RWCMO_54.energy_output(positions[i][0], positions[i][1], wind_speeds[i], wind_directions[i], frequency_intervals[i])
            for i in range(len(positions))
        ]

        # Sum the total energy produced by all turbines
        total_energy = np.sum(energies)

        # Calculate penalties for minimum distance constraints
        penalty = 0
        min_distance = 80  # Minimum distance between turbines
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < min_distance:
                    penalty += (min_distance - dist) ** 2 * 10000

        # Return the objective function value
        return -total_energy + penalty

    @staticmethod
    def energy_output(xi, yi, wind_speed, wind_direction, frequency_interval):
        # Simplified simulation of the energy production of a turbine
        Pr = 1.225  # Air density (kg/m³)
        R = 40  # Radius of the turbine blades (m)
        Cp = 0.45  # Power coefficient
        theta = np.radians(wind_direction - np.arctan2(yi, xi))
        v_rel = wind_speed * np.cos(theta)
        return frequency_interval * 0.5 * Pr * np.pi * R**2 * Cp * v_rel**3   
