"""Funciones comunes para interpretar los porcentajes de aparición."""

import math


def repartir_cantidades(total_particulas, fracciones):
    """Convierte fracciones de aparición en conteos enteros cuya suma es el total."""
    if isinstance(total_particulas, bool) or not isinstance(total_particulas, int):
        raise ValueError("El total de partículas debe ser un número entero.")
    if total_particulas < 1:
        raise ValueError("El total de partículas debe ser mayor que cero.")
    if not fracciones or any(not math.isfinite(float(f)) or float(f) <= 0 for f in fracciones):
        raise ValueError("Cada clase debe tener un porcentaje de aparición positivo y válido.")

    suma = sum(float(f) for f in fracciones)
    cuotas = [total_particulas * float(f) / suma for f in fracciones]
    conteos = [math.floor(cuota) for cuota in cuotas]
    sobrantes = total_particulas - sum(conteos)
    orden = sorted(
        range(len(cuotas)),
        key=lambda indice: cuotas[indice] - conteos[indice],
        reverse=True,
    )
    for indice in orden[:sobrantes]:
        conteos[indice] += 1
    return conteos
