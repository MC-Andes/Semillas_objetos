import json
import math
from dataclasses import asdict, dataclass, field
from typing import List, Tuple
from utilidades_particulas import repartir_cantidades


COLORES_DISPONIBLES = {
    "azul": "#3498db",
    "naranja": "#e67e22",
    "verde": "#2ecc71",
    "morado": "#9b59b6",
    "rojo": "#e74c3c",
    "amarillo": "#f1c40f",
    "gris": "#95a5a6",
}


@dataclass
class ClaseParticula:
    nombre: str
    fraccion_aparicion: float
    rango_caras: Tuple[int, int]
    volumen_objetivo: float
    tolerancia_tamano_porcentaje: float = 0.0
    rango_angulos: Tuple[float, float] = (0.0, 180.0)
    color: str = "#3498db"

    def __post_init__(self):
        if not (0.0 < self.fraccion_aparicion <= 1.0):
            raise ValueError("La fracción debe estar entre 0 y 1.")
        if not 0 <= self.tolerancia_tamano_porcentaje < 100:
            raise ValueError("La tolerancia de tamaño debe estar entre 0 y menos de 100 %.")
        if self.rango_caras[0] < 3 or self.rango_caras[1] < self.rango_caras[0]:
            raise ValueError("El rango de caras debe ser válido y empezar en 3 o más.")
        if not (0 <= self.rango_angulos[0] <= self.rango_angulos[1] <= 180):
            raise ValueError("El rango de ángulos debe estar entre 0 y 180 grados.")


@dataclass
class EspecificacionEspacio:
    ancho: float  # mm
    alto: float  # mm
    profundo: float  # mm
    total_particulas: int
    clases: List[ClaseParticula] = field(default_factory=list)

    @property
    def volumen_total(self) -> float:
        return self.ancho * self.alto * self.profundo

    def validar_configuracion(self):
        if min(self.ancho, self.alto, self.profundo) <= 0:
            raise ValueError("Las dimensiones del volumen deben ser mayores que cero.")
        if isinstance(self.total_particulas, bool) or not isinstance(self.total_particulas, int):
            raise ValueError("El total de partículas debe ser un número entero.")
        if not self.clases:
            raise ValueError("Debe definirse al menos una clase de partículas.")
        suma = sum(c.fraccion_aparicion for c in self.clases)
        if not math.isclose(suma, 1.0, rel_tol=1e-3):
            raise ValueError(
                f"La suma de porcentajes de aparición debe ser 1.0 (actual: {suma:.2f})"
            )
        repartir_cantidades(self.total_particulas, [c.fraccion_aparicion for c in self.clases])
        nombres = [c.nombre.strip() for c in self.clases]
        if len(set(nombres)) != len(nombres):
            raise ValueError("Cada clase debe tener un nombre único.")
        volumen_ponderado_por_particula = sum(
            (c.fraccion_aparicion / suma) * c.volumen_objetivo for c in self.clases
        )
        volumen_ponderado = self.total_particulas * volumen_ponderado_por_particula
        volumen_con_margen = volumen_ponderado * 1.15
        if volumen_con_margen > self.volumen_total + 1e-6:
            raise ValueError(
                "El volumen ponderado de partículas con margen de seguridad del "
                f"15 % ({volumen_con_margen:.6g} mm³) supera el volumen de la caja "
                f"({self.volumen_total:.6g} mm³)."
            )

    def calcular_estimacion_particulas(self) -> dict:
        self.validar_configuracion()
        resumen = {}
        total_particulas = self.total_particulas
        suma_fracciones = sum(c.fraccion_aparicion for c in self.clases)
        conteos_por_clase = repartir_cantidades(
            total_particulas, [c.fraccion_aparicion for c in self.clases]
        )
        volumen_ponderado_por_particula = sum(
            (c.fraccion_aparicion / suma_fracciones) * c.volumen_objetivo
            for c in self.clases
        )
        volumen_ponderado = total_particulas * volumen_ponderado_por_particula
        volumen_particulas_solicitado = sum(
            cantidad * clase.volumen_objetivo
            for clase, cantidad in zip(self.clases, conteos_por_clase)
        )

        for c, cantidad_asignada in zip(self.clases, conteos_por_clase):
            vol_asignado = (
                total_particulas
                * (c.fraccion_aparicion / suma_fracciones)
                * c.volumen_objetivo
            )
            volumen_clase_solicitado = cantidad_asignada * c.volumen_objetivo
            resumen[c.nombre] = {
                "cantidad_asignada": cantidad_asignada,
                "volumen_objetivo_mm3": c.volumen_objetivo,
                "volumen_particulas_solicitado_mm3": volumen_clase_solicitado,
                "volumen_ponderado_por_aparicion_mm3": vol_asignado,
                "tolerancia_tamano_porcentaje": c.tolerancia_tamano_porcentaje,
                "rango_caras": c.rango_caras,
            }
        return {
            "volumen_dominio_mm3": self.volumen_total,
            "total_particulas_estimadas": total_particulas,
            "volumen_particulas_solicitado_mm3": volumen_particulas_solicitado,
            "volumen_ponderado_por_aparicion_mm3": volumen_ponderado,
            "volumen_con_margen_15_por_ciento_mm3": volumen_ponderado * 1.15,
            "volumen_libre_estimado_mm3": max(0.0, self.volumen_total - volumen_particulas_solicitado),
            "detalle_clases": resumen,
        }

    def exportar_json(self, ruta_archivo: str = "config_simulacion.json"):
        self.validar_configuracion()
        data = {
            "dimensiones": {
                "ancho": self.ancho,
                "alto": self.alto,
                "profundo": self.profundo,
                "volumen_total": self.volumen_total,
            },
            "total_particulas": self.total_particulas,
            "clases": [asdict(c) for c in self.clases],
            "estimaciones": self.calcular_estimacion_particulas(),
        }
        with open(ruta_archivo, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"✓ Configuración guardada en '{ruta_archivo}'.")


def _leer_numero(mensaje: str, minimo: float = 0, maximo: float = None) -> float:
    while True:
        try:
            valor = float(input(mensaje).replace(",", "."))
            if valor <= minimo or (maximo is not None and valor > maximo):
                limite = f"mayor que {minimo}" if maximo is None else f"entre {minimo} y {maximo}"
                print(f"Ingresa un valor {limite}.")
                continue
            return valor
        except ValueError:
            print("Ingresa un número válido.")


def _leer_entero(mensaje: str, minimo: int = 1) -> int:
    while True:
        try:
            valor = int(input(mensaje))
            if valor < minimo:
                print(f"Ingresa un entero mayor o igual a {minimo}.")
                continue
            return valor
        except ValueError:
            print("Ingresa un número entero válido.")


def solicitar_configuracion() -> EspecificacionEspacio:
    """Solicita las dimensiones y las características de cada clase."""
    print("=== Definición del volumen y las partículas (dimensiones en mm) ===")
    espacio = EspecificacionEspacio(
        ancho=_leer_numero("Ancho del volumen: "),
        alto=_leer_numero("Alto del volumen: "),
        profundo=_leer_numero("Profundidad del volumen: "),
        total_particulas=_leer_entero("Número total de partículas para todo el volumen: "),
    )
    cantidad = _leer_entero("Cantidad de clases de partículas: ")

    for indice in range(1, cantidad + 1):
        print(f"\n--- Clase {indice} ---")
        nombre = input("Nombre de la clase: ").strip()
        while not nombre:
            nombre = input("El nombre no puede estar vacío. Nombre: ").strip()
        porcentaje = _leer_numero("Porcentaje de aparición de esta clase (0-100): ", maximo=100)
        caras_min = _leer_entero("Mínimo de caras (3 o más): ", minimo=3)
        caras_max = _leer_entero("Máximo de caras: ", minimo=caras_min)
        angulo_min = _leer_numero("Ángulo mínimo entre caras (grados): ", minimo=-0.000001, maximo=180)
        angulo_max = _leer_numero("Ángulo máximo entre caras (grados): ", minimo=angulo_min - 0.000001, maximo=180)
        print("Colores disponibles: " + ", ".join(COLORES_DISPONIBLES))
        color_nombre = input("Color: ").strip().lower()
        while color_nombre not in COLORES_DISPONIBLES:
            color_nombre = input("Elige uno de los colores disponibles: ").strip().lower()

        volumen_objetivo = _leer_numero("Volumen necesario por partícula (mm³): ")
        tolerancia = float("nan")
        while not math.isfinite(tolerancia) or not 0 <= tolerancia < 100:
            try:
                tolerancia = float(input("Tolerancia de tamaño por partícula (± %, entre 0 y menos de 100): ").replace(",", "."))
                if not 0 <= tolerancia < 100:
                    print("La tolerancia debe ser desde 0 hasta menos de 100 %.")
            except ValueError:
                print("Ingresa un porcentaje numérico válido.")
        espacio.clases.append(
            ClaseParticula(
                nombre=nombre,
                fraccion_aparicion=porcentaje / 100,
                rango_caras=(caras_min, caras_max),
                volumen_objetivo=volumen_objetivo,
                tolerancia_tamano_porcentaje=tolerancia,
                rango_angulos=(angulo_min, angulo_max),
                color=COLORES_DISPONIBLES[color_nombre],
            )
        )

    espacio.validar_configuracion()
    return espacio


if __name__ == "__main__":
    while True:
        try:
            espacio = solicitar_configuracion()
            espacio.exportar_json("config_simulacion.json")
            break
        except ValueError as error:
            print(f"Configuración inválida: {error}")
            print("Vuelve a ejecutar el programa para ingresar los datos nuevamente.")
            break
