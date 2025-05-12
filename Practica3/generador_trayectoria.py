#!/usr/bin/env python3 
from sympy import *
import matplotlib
import matplotlib.pyplot as plt

class GeneradorTrayectoria():
  def __init__(self, dim = (0.3, 0.3, 0.3)):
    self.dim = dim
  def trans_homo(self, x, y, z, gamma, beta, alpha):
    R_z = Matrix([ [cos(alpha), -sin(alpha), 0], [sin(alpha), cos(alpha), 0],[0, 0, 1]])
    R_y = Matrix([ [cos(beta), 0, -sin(beta)], [0, 1, 0],[sin(beta), 0, cos(beta)]])
    R_x = Matrix([ [1, 0, 0], [0, cos(gamma), -sin(gamma)],[0, sin(gamma), cos(gamma)]])
    R = R_x * R_y *R_z
    p = Matrix([[x],[y],[z]])
    T = Matrix.vstack(Matrix.hstack(R, p), Matrix([[0,0,0,1]]))
    return T 
  def cinematica_directa(self):
    print("Generando cinematica directa")
    self.theta_0_1, self.theta_1_2, self.theta_2_3 = symbols("theta_0_1 theta_1_2 theta_2_3")
    #Matrices de transformación
    self.T_0_1 = self.trans_homo(0, 0, 0, pi/2, 0, self.theta_0_1)
    self.T_1_2 = self.trans_homo(self.dim[0], 0, 0, 0, 0, self.theta_1_2)
    self.T_2_3 = self.trans_homo(self.dim[1], 0, 0, 0, 0, self.theta_2_3)
    self.T_3_P = self.trans_homo(self.dim[2], 0, 0, 0, 0, 0)
    self.T_0_P = simplify(self.T_0_1 * self.T_1_2 * self.T_2_3 * self.T_3_P)
    #Vector de postura xi = [x z th]
    self.xi_0_P = Matrix([[self.T_0_P[0, 3]],
                          [self.T_0_P[2, 3]],
                          [self.theta_0_1 + self.theta_1_2 + self.theta_2_3]])
  def generar_trayectoria(self, q_in = (pi/4, -pi/2, 3*pi/8), xi_fn = (0.5, 0.2, 0), tie = (0, 2), frec = 60):
    print("Creando trayectoria")
    #Variables para polinomio lambda
    self.t, self.a_0, self.a_1, self.a_2, self.a_3, self.a_4, self.a_5 = symbols(
    "t a_0 a_1 a_2 a_3 a_4 a_5")
    #Polinomio lambda lam = a0 + a1t + a2t^2 + a3t^3 + a4t^4 + a5t^5
    self.lam = self.a_0 + self.a_1 * self.t + self.a_2 * (self.t)**2 + self.a_3 * (self.t)**3 + self.a_4 * (self.t)**4+ self.a_5 * (self.t)**5
    #Primera y segunda derivada de lambda
    self.lam_dot = diff(self.lam, self.t)
    self.lam_dot_dot = diff(self.lam_dot, self.t)
    # Cálculo de parámetros de lambda. 
    # Planteando ecuaciones igualadas a cero
    # lam(t=ti) = 0
    # lam(t=tf) = 1   ==>  lam(t=tf) -1 = 0
    # lam'(t=ti) = 0
    # lam'(t=tf) = 0
    # lam''(t=ti) = 0
    # lam''(t=tf) = 0
    ec_1 = self.lam.subs(self.t, tie[0])
    ec_2 = self.lam.subs(self.t, tie[1]) - 1
    ec_3 = self.lam_dot.subs(self.t, tie[0])
    ec_4 = self.lam_dot.subs(self.t, tie[1])
    ec_5 = self.lam_dot_dot.subs(self.t, tie[0])
    ec_6 = self.lam_dot_dot.subs(self.t, tie[1])
    # Resolviendo sistema para las variables a0-a5
    terminos = solve([ec_1, ec_2, ec_3, ec_4, ec_5, ec_6], [self.a_0, self.a_1, self.a_2, self.a_3, self.a_4, self.a_5], dict = True)
    # Tomando la primera solución devuelta y sustituyéndola en el polinomio
    self.lam_s          = self.lam.subs(terminos[0])
    self.lam_dot_s      = self.lam_dot.subs(terminos[0])
    self.lam_dot_dot_s  = self.lam_dot_dot.subs(terminos[0])

    # Calculo de la posicion inicial del efector final a partir del vector de postura
    xi_in = self.xi_0_P.subs({
      self.theta_0_1: q_in[0],
      self.theta_1_2: q_in[1],
      self.theta_2_3: q_in[2]
    })
    # Posiciones de espacio de trabajo
    # xi = xi_in + lam(t) * (xi_fn - xi_in) 
    self.xi = xi_in + Matrix([
      [self.lam_s * (xi_fn[0] - xi_in[0])],
      [self.lam_s * (xi_fn[1] - xi_in[1])],
      [self.lam_s * (xi_fn[2] - xi_in[2])]
    ])
    # Velocidades de espacio de trabajo
    # xi' = lam'(t) * (xi_fn - xi_in)
    self.xi_dot = Matrix([
      [self.lam_dot_s * (xi_fn[0] - xi_in[0])],
      [self.lam_dot_s * (xi_fn[1] - xi_in[1])],
      [self.lam_dot_s * (xi_fn[2] - xi_in[2])]
    ])
    # Aceleraciones de espacio de trabajo
    # xi'' = lam''(t) * (xi_fn - xi_in)
    self.xi_dot_dot = Matrix([
      [self.lam_dot_dot_s * (xi_fn[0] - xi_in[0])],
      [self.lam_dot_dot_s * (xi_fn[1] - xi_in[1])],
      [self.lam_dot_dot_s * (xi_fn[2] - xi_in[2])]
    ])
    print("Vector segunda derivada")
    print(self.xi)
    print(self.xi_dot)
    print(self.xi_dot_dot)

    # Muestreo del espacio de trabajo
    print("Muestreando trayectoria")
    # Número de muestras e incremento de tiempo
    self.muestras = int(frec * (tie[1] - tie[0]) + 1)
    self.dt = 1.0 / frec

    # Muestreo de tiempo entre tf y ti
    self.t_m = Matrix.zeros(1, self.muestras)
    self.t_m[0,0] = tie[0]
    for a in range(self.muestras - 1):
      self.t_m[0, a + 1] = self.t_m[0, a] + self.dt

    # Matrices vacías para guardar valores del espacio de trabajo
    # 3 filas, n columnas (cada columna es una posición/velocidad/aceleración del espacio de trabajo en un instante)
    self.xi_m         = Matrix.zeros(3, self.muestras)
    self.xi_dot_m     = Matrix.zeros(3, self.muestras)
    self.xi_dot_dot_m = Matrix.zeros(3, self.muestras)

    # Generando funciones para evitar usar lenguaje simbólico en las sustituciones
    xi_m_func =         lambdify([self.t], self.xi)
    xi_dot_m_func =     lambdify([self.t], self.xi_dot)
    xi_dot_dot_m_func = lambdify([self.t], self.xi_dot_dot)
    for a in range(self.muestras):
      """Así se sustituiría directo
      self.xi_m[:,a]          = self.xi.subs(self.t, self.t_m[0, a])
      self.xi_dot_m[:,a]      = self.xi_dot.subs(self.t, self.t_m[0, a])
      self.xi_dot_dot_m[:,a]  = self.xi_dot_dot.subs(self.t, self.t_m[0, a])"""
      self.xi_m[:, a]         = xi_m_func(float(self.t_m[0, a]))
      self.xi_dot_m[:, a]     = xi_dot_m_func(float(self.t_m[0, a]))
      self.xi_dot_dot_m[:, a] = xi_dot_dot_m_func(float(self.t_m[0, a]))
      print(a)

    #Agregando posición inicial como variable de la clase
    self.q_in = q_in


  def cinematica_inversa(self):
    print("Calculando cinematica inversa")
    # Variables para los valores de las velocidades del ws
    self.x_0_P_dot, self.z_0_P_dot, self.theta_0_P_dot = symbols(
    "x_0_P_dot z_0_P_dot theta_0_P_dot")
    # Derivada del vector de postura en términos de las velocidades del efector final
    # xi' = [x' z' th'] 
    self.xi_dot = Matrix([[self.x_0_P_dot], [self.z_0_P_dot], [self.theta_0_P_dot]])
    # Jacobiano
    self.J = Matrix.hstack(diff(self.xi_0_P, self.theta_0_1), 
                           diff(self.xi_0_P, self.theta_1_2), 
                           diff(self.xi_0_P, self.theta_2_3))
    self.J_inv = self.J.inv()
    # De la primera derivada, despejamos q': 
    # xi' = J * q'
    # q' = J-1 * xi'
    self.q_dot = self.J_inv * self.xi_dot

    # Crear una función a partir de la expresión simbólica
    q_dot_func = lambdify([self.x_0_P_dot, self.z_0_P_dot, self.theta_0_P_dot, self.theta_0_1, self.theta_1_2, self.theta_2_3], self.q_dot)

    # Matrices vacías para los valores de espacio de las juntas
    self.q_m         = Matrix.zeros(3, self.muestras)
    self.q_dot_m     = Matrix.zeros(3, self.muestras)
    self.q_dot_dot_m = Matrix.zeros(3, self.muestras)
    # Agregando posición inicial (dato)
    self.q_m[:,0] = Matrix([[self.q_in[0]], [self.q_in[1]], [self.q_in[2]]])
    # Agregando velocidad inicial
    # q' = J-1 * xi' => Obtenemos velocidad de las juntas 
    # en el punto actual sustituyendo velocidad actual del efector y 
    # posición actual de las juntas
    """Así se sustituiría directo
      q_dot_m[:,0] = self.q_dot.subs({
      self.x_0_P_dot:     xi_dot_m[0, 0],
      self.z_0_P_dot:     xi_dot_m[1, 0],
      self.theta_O_P_dot: xi_dot_m[2, 0],
      self.theta_O_1:     q_m[0, 0],
      self.theta_1_2:     q_m[1, 0],
      self.theta_2_3:     q_m[2, 0]})"""

    self.q_dot_m[:, 0] = q_dot_func(
        float(self.xi_dot_m[0, 0]),
        float(self.xi_dot_m[1, 0]),
        float(self.xi_dot_m[2, 0]),
        float(self.q_m[0, 0]),
        float(self.q_m[1, 0]),
        float(self.q_m[2, 0])
    )

    print("Calculando puntos")
    for a in range(self.muestras - 1):
      # Posición de las juntas
      # Posición siguiente = posición actual + velocidad actual * dt 
      self.q_m[:,a+1] = self.q_m[:,a] + self.q_dot_m[:,a] * self.dt
      # Velocidad de las juntas 
      # Velocidad siguiente de las juntas a partir de velocidad siguiente del efector y posición siguiente de las juntas
      """self.q_dot_m[:,a+1] = self.q_dot.subs({
                                          self.x_0_P_dot:     xi_dot_m[0, a],
                                          self.z_0_P_dot:     xi_dot_m[1, a],
                                          self.theta_O_P_dot: xi_dot_m[2, a],
                                          self.theta_O_1:     q_m[0, a],
                                          self.theta_1_2:     q_m[1, a],
                                          self.theta_2_3:     q_m[2, a]})"""
      self.q_dot_m[:, a + 1] = q_dot_func(
        float(self.xi_dot_m[0, a + 1]),
        float(self.xi_dot_m[1, a + 1]),
        float(self.xi_dot_m[2, a + 1]),
        float(self.q_m[0, a + 1]),
        float(self.q_m[1, a + 1]),
        float(self.q_m[2, a + 1]))
        
      # Aceleración
      # Aceleración actual = velocidad siguiente - velocidad actual / dt
      self.q_dot_dot_m[:,a] = (self.q_dot_m[:,a+1] - self.q_dot_m[:,a]) / self.dt

      print("Iteración: " + str(a))
    # Aceleración final (cero)
    self.q_dot_dot_m[:, self.muestras - 1] = Matrix.zeros(3, 1)
    
  def graficar(self):
    fig, ((xi_g, xi_dot_g, xi_dot_dot_g),
              (q_g, q_dot_g, q_dot_dot_g)) = plt.subplots(nrows=2, ncols = 3)
    # Posiciones ws
    xi_g.set_title("Posiciones de WS")
    xi_g.plot(self.t_m.T, self.xi_m[0, :].T, color = "RED")
    xi_g.plot(self.t_m.T, self.xi_m[1, :].T, color = "GREEN")
    xi_g.plot(self.t_m.T, self.xi_m[2, :].T, color = "BLUE")

    # Velocidades ws
    xi_dot_g.set_title("Velocidades de WS")
    xi_dot_g.plot(self.t_m.T, self.xi_dot_m[0, :].T, color = "RED")
    xi_dot_g.plot(self.t_m.T, self.xi_dot_m[1, :].T, color = "GREEN")
    xi_dot_g.plot(self.t_m.T, self.xi_dot_m[2, :].T, color = "BLUE")

    # Aceleraciones ws
    xi_dot_dot_g.set_title("Aceleraciones de WS")
    xi_dot_dot_g.plot(self.t_m.T, self.xi_dot_dot_m[0, :].T, color = "RED")
    xi_dot_dot_g.plot(self.t_m.T, self.xi_dot_dot_m[1, :].T, color = "GREEN")
    xi_dot_dot_g.plot(self.t_m.T, self.xi_dot_dot_m[2, :].T, color = "BLUE")

    # Posiciones q
    q_g.set_title("Posiciones de q")
    q_g.plot(self.t_m.T, self.q_m[0, :].T, color = "RED")
    q_g.plot(self.t_m.T, self.q_m[1, :].T, color = "GREEN")
    q_g.plot(self.t_m.T, self.q_m[2, :].T, color = "BLUE")

    # Velocidades q
    q_dot_g.set_title("Velocidades de q")
    q_dot_g.plot(self.t_m.T, self.q_dot_m[0, :].T, color = "RED")
    q_dot_g.plot(self.t_m.T, self.q_dot_m[1, :].T, color = "GREEN")
    q_dot_g.plot(self.t_m.T, self.q_dot_m[2, :].T, color = "BLUE")

    # Aceleraciones q
    q_dot_dot_g.set_title("Aceleraciones de q")
    q_dot_dot_g.plot(self.t_m.T, self.q_dot_dot_m[0, :].T, color = "RED")
    q_dot_dot_g.plot(self.t_m.T, self.q_dot_dot_m[1, :].T, color = "GREEN")
    q_dot_dot_g.plot(self.t_m.T, self.q_dot_dot_m[2, :].T, color = "BLUE")
    plt.show()
  def graficar_ws(self):
    plt.plot(self.xi_m[0, :].T, self.xi_m[1, :].T)
    plt.axis((0, 1, 0, 1))
    plt.show()
  def graficar_trayectoria(self):
    # Cinematica directa de cada sistema de referencia
    self.T_0_2 = self.T_0_1 * self.T_1_2
    self.T_0_3 = self.T_0_2 * self.T_2_3
    
    plt.plot(self.xi_m[0, :].T, self.xi_m[1, :].T)
    plt.axis((0, 1, 0, 1))
    plt.show()
def main():
  generador_trayectoria = GeneradorTrayectoria()
  generador_trayectoria.cinematica_directa()
  generador_trayectoria.generar_trayectoria()
  generador_trayectoria.cinematica_inversa()
  generador_trayectoria.graficar()
  generador_trayectoria.graficar_ws()

if __name__ == "__main__":
  main()
