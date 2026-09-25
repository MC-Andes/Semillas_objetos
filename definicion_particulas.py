import json
import math
from dataclasses import asdict, dataclass, field
from typing import List, Tuple


@dataclass
class ClaseParticula:
    nombre: str
    fraccion_volumen: float
    rango_caras: Tuple[int, int]
    volumen_objetivo: float
    volumen_minimo: float

    def __post_init__(self):
        if not (0.0 < self.fraccion_volumen <= 1.0):
            raise ValueError("La fracción debe estar entre 0 y 1.")
        if self.volumen_objetivo < self.volumen_minimo:
            raise ValueError(
                "El volumen objetivo no puede ser menor al mínimo."
            )


@dataclass
class EspecificacionEspacio:
    ancho: float  # mm
    alto: float  # mm
    profundo: float  # mm
    clases: List[ClaseParticula] = field(default_factory=list)

    @property
    def volumen_total(self) -> float:
        return self.ancho * self.alto * self.profundo

    def validar_configuracion(self):
        suma = sum(c.fraccion_volumen for c in self.clases)
        if not math.isclose(suma, 1.0, rel_tol=1e-3):
            raise ValueError(
                f"La suma de fracciones debe ser 1.0 (actual: {suma:.2f})"
            )

    def calcular_estimacion_particulas(self) -> dict:
        self.validar_configuracion()
        resumen = {}
        total_estimado = 0

        for c in self.clases:
            vol_asignado = self.volumen_total * c.fraccion_volumen
            num_particulas = round(vol_asignado / c.volumen_objetivo)
            resumen[c.nombre] = {
                "num_particulas": num_particulas,
                "volumen_objetivo_mm3": c.volumen_objetivo,
                "volumen_total_asignado_mm3": vol_asignado,
                "rango_caras": c.rango_caras,
            }
            total_estimado += num_particulas

        return {
            "volumen_dominio_mm3": self.volumen_total,
            "total_particulas_estimadas": total_estimado,
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
            "clases": [asdict(c) for c in self.clases],
            "estimaciones": self.calcular_estimacion_particulas(),
        }
        with open(ruta_archivo, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✓ Configuración guardada en '{ruta_archivo}'.")


if __name__ == "__main__":
    # Dominio compacto de 20x20x20 mm = 8,000 mm³
    espacio = EspecificacionEspacio(ancho=20.0, alto=20.0, profundo=20.0)

    # Clase A: 60% vol (4,800 mm³) / vol_obj = 200 mm³ -> 24 partículas
    clase_a = ClaseParticula(
        nombre="Clase_A",
        fraccion_volumen=0.60,
        rango_caras=(6, 14),
        volumen_objetivo=200.0,
        volumen_minimo=100.0,
    )

    # Clase B: 40% vol (3,200 mm³) / vol_obj = 320 mm³ -> 10 partículas
    clase_b = ClaseParticula(
        nombre="Clase_B",
        fraccion_volumen=0.40,
        rango_caras=(4, 10),
        volumen_objetivo=320.0,
        volumen_minimo=150.0,
    )

    espacio.clases.extend([clase_a, clase_b])
    espacio.exportar_json("config_simulacion.json")