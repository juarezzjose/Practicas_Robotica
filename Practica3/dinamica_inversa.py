#!/usr/bin/env python3 
from generador_trayectoria import GeneradorTrayectoria
from sympy import *
import matplotlib
import matplotlib.pyplot as plt

class GeneradorDinamica(GeneradorTrayectoria):
  def __init__(self):
    super().__init__()
  def matriz_inercia(self, lx, ly, lz, masa):
    return Matrix([[(masa/12.0)*(ly**2 + lz**2), 0, 0], 
                  [0, (masa/12.0)*(lx**2 + lz**2), 0], 
                  [0, 0, (masa/12.0)*(lx**2 + ly**2)]])
  def definir_inercia(self, masas = [0.25, 0.25, 0.25]):
    # Cinematica directa de cada sistema de referencia
    self.T_0_2 = self.T_0_1 * self.T_1_2
    self.T_0_3 = self.T_0_2 * self.T_2_3
    # Transformaciones de centros de masa
    self.T_1_C1 = self.trans_homo(self.dim[0] / 2, 0, 0, 0, 0, 0)
    self.T_2_C2 = self.trans_homo(self.dim[1] / 2, 0, 0, 0, 0, 0)
    self.T_3_C3 = self.trans_homo(self.dim[2] / 2, 0, 0, 0, 0, 0)
    self.T_0_C1 = simplify(self.T_0_1 * self.T_1_C1)
    self.T_0_C2 = simplify(self.T_0_2 * self.T_2_C2)
    self.T_0_C3 = simplify(self.T_0_3 * self.T_3_C3)
    #Vectores de posición de sistemas de referencia
    self.p_0_1 = self.T_0_1[:3, 3]
    self.p_1_2 = self.T_1_2[:3, 3]
    self.p_2_3 = self.T_2_3[:3, 3]
    self.p_0_1 = self.T_0_1[:3, 3]
    self.p_0_2 = self.T_0_2[:3, 3]
    self.p_0_3 = self.T_0_3[:3, 3]
    #Vectores de posición de centros de masa
    self.p_1_C1 = self.T_1_C1[:3, 3]
    self.p_2_C2 = self.T_2_C2[:3, 3]
    self.p_3_C3 = self.T_3_C3[:3, 3]
    self.p_0_C1 = self.T_0_C1[:3, 3]
    self.p_0_C2 = self.T_0_C2[:3, 3]
    self.p_0_C3 = self.T_0_C3[:3, 3]
    #Rotaciones
    self.R_0_1 = self.T_0_1[:3, :3]
    self.R_1_2 = self.T_1_2[:3, :3]
    self.R_2_3 = self.T_2_3[:3, :3]
    self.Id = Matrix([[1,0,0], [0,1,0], [0,0,1]]) 

    # Variables de velocidad angular
    self.theta_0_1_dot = Symbol('theta_0_1_dot')
    self.theta_1_2_dot = Symbol('theta_1_2_dot')
    self.theta_2_3_dot = Symbol('theta_2_3_dot')
    # Variables de aceleración angular
    self.theta_0_1_dot_dot = Symbol('theta_0_1_dot_dot')
    self.theta_1_2_dot_dot = Symbol('theta_1_2_dot_dot')
    self.theta_2_3_dot_dot = Symbol('theta_2_3_dot_dot')
    # Masas
    self.m1 = masas[0]
    self.m2 = masas[1]
    self.m3 = masas[2]
    # Matrices de inercia
    self.Ic1 = self.matriz_inercia(self.dim[0], 0.03, 0.03, self.m1)
    self.Ic2 = self.matriz_inercia(self.dim[0], 0.03, 0.03, self.m2)
    self.Ic3 = self.matriz_inercia(self.dim[0], 0.03, 0.03, self.m3)
    #Gravedad
    self.g = -9.81
    
  def generar_esfuerzos(self):
    #Velocidades angulares de sistemas
    omega_1_1 = Matrix([0, 0, self.theta_0_1_dot])
    omega_2_2 = self.R_1_2.transpose() @ omega_1_1 + Matrix([0, 0, self.theta_1_2_dot])
    omega_3_3 = self.R_2_3.transpose() @ omega_2_2 + Matrix([0, 0, self.theta_2_3_dot]) 
    #Velocidades angulares de centros de masa
    omega_1_C1 = omega_1_1
    omega_2_C2 = omega_2_2
    omega_3_C3 = omega_3_3
    omega_1_C1_f = lambdify([self.theta_0_1_dot], omega_1_C1)
    omega_2_C2_f = lambdify([self.theta_0_1_dot, self.theta_1_2_dot], omega_2_C2)
    omega_3_C3_f = lambdify([self.theta_0_1_dot, self.theta_1_2_dot, self.theta_2_3], omega_3_C3)
    #Velocidades lineales de sistemas
    v_1_1 = Matrix([0, 0, 0])
    v_2_2 = self.R_1_2.transpose() @ (v_1_1 + omega_1_1.cross(self.p_1_2))
    v_3_3 = self.R_2_3.transpose() @ (v_2_2 + omega_2_2.cross(self.p_2_3))
    #Velocidades lineales de centros de masa
    v_1_C1 = v_1_1 + omega_1_C1.cross(self.p_1_C1)
    v_2_C2 = v_2_2 + omega_2_C2.cross(self.p_2_C2)
    v_3_C3 = v_3_3 + omega_3_C3.cross(self.p_3_C3)
    v_1_C1_f = lambdify([self.theta_0_1, self.theta_0_1_dot], v_1_C1)
    v_2_C2_f = lambdify([self.theta_0_1, self.theta_0_1_dot, self.theta_1_2, self.theta_1_2_dot], v_2_C2)
    v_3_C3_f = lambdify([self.theta_0_1, self.theta_0_1_dot, self.theta_1_2, self.theta_1_2_dot, self.theta_2_3, self.theta_2_3_dot], v_3_C3)

    #Arreglos para guardar velocidades angulares
    # 3 filas, n columnas (cada columna es una velocidad angular del centro de masa en un instante)
    self.omega_1_C1_m = Matrix.zeros(3, self.muestras)
    self.omega_2_C2_m = Matrix.zeros(3, self.muestras)
    self.omega_3_C3_m = Matrix.zeros(3, self.muestras)
    #Arreglos para guardar velocidades lineales
    # 3 filas, n columnas (cada columna es una velocidad lineal del centro de masa en un instante)
    self.vel_1_C1_m = Matrix.zeros(3, self.muestras)
    self.vel_2_C2_m = Matrix.zeros(3, self.muestras)
    self.vel_3_C3_m = Matrix.zeros(3, self.muestras)
    #Ciclo para todas las muestras
    print("Calculando velocidades")
    for i in range(self.muestras):
      #Velocidades angulares
      self.omega_1_C1_m[:, i] = omega_1_C1_f(self.q_dot_m[0, i])
      self.omega_2_C2_m[:, i] = omega_2_C2_f(self.q_dot_m[0, i], self.q_dot_m[1, i])
      self.omega_3_C3_m[:, i] = omega_3_C3_f(self.q_dot_m[0, i], self.q_dot_m[1, i], self.q_dot_m[2, i])
      
      #Velocidades lineales
      """self.vel_1_C1_m[:, i] = v_1_C1_f(self.q_m[0, i], self.q_dot_m[0, i])
      self.vel_2_C2_m[:, i] = v_2_C2_f(self.q_m[0, i], self.q_dot_m[0, i], self.q_m[1, i], self.q_dot_m[1, i])
      self.vel_3_C3_m[:, i] = v_3_C3_f(self.q_m[0, i], self.q_dot_m[0, i], self.q_m[1, i], self.q_dot_m[1, i], self.q_m[2, i], self.q_dot_m[2, i])
"""
      self.vel_1_C1_m[:, i] = v_1_C1.subs({
        self.theta_0_1:     self.q_m[0, i],
        self.theta_0_1_dot: self.q_dot_m[0, i]
      })
      self.vel_2_C2_m[:, i] = v_2_C2.subs({
        self.theta_0_1:     self.q_m[0, i],
        self.theta_0_1_dot: self.q_dot_m[0, i],
        self.theta_1_2:     self.q_m[1, i],
        self.theta_1_2_dot: self.q_dot_m[1, i]
      })
      self.vel_3_C3_m[:, i] = v_2_C2.subs({
        self.theta_0_1:     self.q_m[0, i],
        self.theta_0_1_dot: self.q_dot_m[0, i],
        self.theta_1_2:     self.q_m[1, i],
        self.theta_1_2_dot: self.q_dot_m[1, i],
        self.theta_2_3:     self.q_m[2, i],
        self.theta_2_3_dot: self.q_dot_m[2, i]
      })
      print(i)
    #Euler-Lagrange
    #Energía cinética
    k1 = 0.5 * self.m1 * v_1_C1.dot(v_1_C1) + 0.5 * omega_1_C1.dot(self.Ic1@omega_1_C1)
    k2 = 0.5 * self.m2 * v_2_C2.dot(v_2_C2) + 0.5 * omega_2_C2.dot(self.Ic2@omega_2_C2)
    k3 = 0.5 * self.m3 * v_3_C3.dot(v_3_C3) + 0.5 * omega_3_C3.dot(self.Ic3@omega_3_C3)
    k = k1 + k2 + k3
    #Energía potencial
    u1 = - self.m1 * Matrix([0, 0, self.g]).dot(self.p_0_C1)
    u2 = - self.m2 * Matrix([0, 0, self.g]).dot(self.p_0_C2)
    u3 = - self.m3 * Matrix([0, 0, self.g]).dot(self.p_0_C3)
    u = u1 + u2 + u3
    #Lagrangiano
    La = k - u
    #Derivadas respecto al espacio de trabajo
    La_dot_q = Matrix([diff(La, self.theta_0_1), 
                       diff(La, self.theta_1_2), 
                       diff(La, self.theta_2_3)])
    #Derivadas respecto a la derivada del espacio de trabajo
    La_dot_q_dot = Matrix([diff(La, self.theta_0_1_dot), 
                            diff(La, self.theta_1_2_dot), 
                            diff(La, self.theta_2_3_dot)])
    #Derivada total
    La_dot_q_dot_dt = diff(La_dot_q_dot, self.theta_0_1) * self.theta_0_1_dot + diff(La_dot_q_dot, self.theta_1_2) * self.theta_1_2_dot + diff(La_dot_q_dot, self.theta_2_3) * self.theta_2_3_dot     + diff(La_dot_q_dot, self.theta_0_1_dot) * self.theta_0_1_dot_dot + diff(La_dot_q_dot, self.theta_1_2_dot) * self.theta_1_2_dot_dot + diff(La_dot_q_dot, self.theta_2_3_dot) * self.theta_2_3_dot_dot
    #Pares en las juntas
    tau = La_dot_q_dot_dt - La_dot_q
    tau_f = lambdify([self.theta_0_1, self.theta_1_2, self.theta_2_3, 
                      self.theta_0_1_dot, self.theta_1_2_dot, self.theta_2_3_dot, 
                      self.theta_0_1_dot_dot, self.theta_1_2_dot_dot, self.theta_2_3_dot_dot], tau)
    #Generar valores numéricos
    self.tau_val = Matrix.zeros(3, self.muestras)
    #Ciclo para todas las muestras
    print("Calculando pares")
    for i in range(self.muestras):
      """self.tau_val[:, i] = tau_f(self.q_m[0, i], self.q_m[1, i], self.q_m[2, i],
                            self.q_dot_m[0, i], self.q_dot_m[1, i], self.q_dot_m[2, i], 
                            self.q_dot_dot_m[0, i], self.q_dot_dot_m[1, i], self.q_dot_dot_m[2, i])"""
      self.tau_val[:, i] = tau.subs({self.theta_0_1 : self.q_m[0, i], self.theta_1_2: self.q_m[1, i], self.theta_2_3 : self.q_m[2, i],
                            self.theta_0_1_dot:self.q_dot_m[0, i], self.theta_1_2_dot:self.q_dot_m[1, i], self.theta_2_3_dot:self.q_dot_m[2, i], 
                            self.theta_0_1_dot_dot:self.q_dot_dot_m[0, i], self.theta_1_2_dot_dot:self.q_dot_dot_m[1, i], self.theta_2_3_dot_dot:self.q_dot_dot_m[2, i]})
      print(i)  
  def graficar_esfuerzos(self):
    fig, ((tau_1_g, tau_2_g, tau_3_g)) = plt.subplots(nrows=1, ncols = 3)
    # Posiciones ws
    tau_1_g.set_title("Esfuerzo junta 1")
    tau_1_g.plot(self.t_m.T, self.tau_val[0, :].T, color = "RED")

    # Velocidades ws
    tau_2_g.set_title("Esfuerzo junta 2")
    tau_2_g.plot(self.t_m.T, self.tau_val[1, :].T, color = "GREEN")

    # Aceleraciones ws
    tau_3_g.set_title("Esfuerzo junta 3")
    tau_3_g.plot(self.t_m.T, self.tau_val[2, :].T, color = "BLUE")
    plt.show()
def main():
  generador_trayectoria = GeneradorDinamica()
  generador_trayectoria.cinematica_directa()
  "generador_trayectoria.generar_trayectoria()"
  generador_trayectoria.generar_trayectoria(q_in=(pi/4, pi/4, pi/4),xi_fn = (0.5, 0.4, pi/2), frec=10)
  generador_trayectoria.cinematica_inversa()
  generador_trayectoria.graficar()
  generador_trayectoria.graficar_ws()
  generador_trayectoria.definir_inercia()
  generador_trayectoria.generar_esfuerzos()
  generador_trayectoria.graficar_esfuerzos()

if __name__ == "__main__":
  main()