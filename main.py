"""Punto de entrada: captura configuración, genera partículas y abre la gráfica."""

from clasificador_particulas import ejecutar_pipeline_simulacion
from definicion_particulas import solicitar_configuracion
from grafica_3d import visualizar_particulas


ARCHIVO_CONFIGURACION = "config_simulacion.json"


def main():
    try:
        configuracion = solicitar_configuracion()
        configuracion.exportar_json(ARCHIVO_CONFIGURACION)

        particulas, dimensiones = ejecutar_pipeline_simulacion(ARCHIVO_CONFIGURACION)
        resumen = {}
        for particula in particulas:
            clase = particula["clase"]
            resumen[clase] = resumen.get(clase, 0) + 1

        print("\n--- Partículas generadas ---")
        for clase, cantidad in resumen.items():
            print(f"{clase}: {cantidad}")

        visualizar_particulas(particulas, dimensiones)
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        print(f"No se pudo completar la generación: {error}")


if __name__ == "__main__":
    main()
