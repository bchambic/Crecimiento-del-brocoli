import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, interp1d
from scipy.optimize import curve_fit

# Datos proporcionados
dias = np.array([10, 20, 30, 40, 50, 60])
alturas = np.array([5.2, 11.6, 20.4, 29.7, 38.9, 44.5])

# Función para interpolación de Newton (diferencias divididas)
def newton_interpolation(x, x_data, y_data):
    n = len(x_data)
    coef = np.zeros([n, n])
    coef[:,0] = y_data
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x_data[i+j] - x_data[i])
    
    poly = coef[0,0]
    temp = 1
    for j in range(1,n):
        temp *= (x - x_data[j-1])
        poly += coef[0,j] * temp
    
    return poly

# Interpolación de Lagrange
poly_lag = lagrange(dias, alturas)
lagrange_15 = poly_lag(15)

# Interpolación por Splines Cúbicos
spline = interp1d(dias, alturas, kind='cubic', fill_value='extrapolate')
spline_15 = spline(15)

# Regresión Lineal
def linear_func(x, a, b):
    return a * x + b

popt, pcov = curve_fit(linear_func, dias, alturas)
regresion_15 = linear_func(15, *popt)

# Interpolación de Newton
newton_15 = newton_interpolation(15, dias, alturas)  # Eliminamos el [0] ya que devuelve escalar

# Resultados
print(f"Predicción a 15 días usando:")
print(f"- Lagrange: {lagrange_15:.2f} cm")
print(f"- Splines cúbicos: {spline_15:.2f} cm")
print(f"- Regresión lineal: {regresion_15:.2f} cm")
print(f"- Newton: {newton_15:.2f} cm")

# Gráfico
plt.figure(figsize=(10, 6))
plt.scatter(dias, alturas, label='Datos reales', color='red')
x_vals = np.linspace(10, 60, 100)

# Graficar las interpolaciones
plt.plot(x_vals, poly_lag(x_vals), label='Lagrange', linestyle='--')
plt.plot(x_vals, spline(x_vals), label='Splines cúbicos', linestyle='-.')
plt.plot(x_vals, linear_func(x_vals, *popt), label='Regresión lineal')
plt.plot(x_vals, [newton_interpolation(x, dias, alturas) for x in x_vals], 
         label='Newton', linestyle=':')

plt.axvline(x=15, color='gray', linestyle=':', alpha=0.5)
plt.title('Crecimiento de la planta de brócoli y predicciones')
plt.xlabel('Días')
plt.ylabel('Altura (cm)')
plt.legend()
plt.grid(True)
plt.show()
