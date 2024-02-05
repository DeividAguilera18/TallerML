import sys
import pickle
import numpy as np

def cargar_modelo(nombre_archivo):
    with open(nombre_archivo, 'rb') as archivo:
        modelo = pickle.load(archivo)
    return modelo

def main():
    if len(sys.argv) != 3:
        print("Uso: python main.py model.pkl datos_a_predecir.csv")
        sys.exit(1)

    nombre_modelo = sys.argv[1]
    nombre_datos = sys.argv[2]

    modelo = cargar_modelo(nombre_modelo)

    # Cargar los datos a predecir
    datos_a_predecir = np.loadtxt(nombre_datos, delimiter=',')

    # Realizar predicciones
    predicciones = modelo.predict(datos_a_predecir)

    print("Predicciones:")
    print(predicciones)

if __name__ == "__main__":
    main()
