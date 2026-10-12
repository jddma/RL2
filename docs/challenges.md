# Reflexión de retos, dificultades y aprendizajes

## Contexto y evidencia

La implementación DQN de `src/dqn.py` y `src/train.py` se entrenó por 500 episodios. La evidencia original está en `results/logs.csv`, `results/config.json`, los checkpoints y `docs/resultados_actividad5.md`. Las evaluaciones independientes se guardaron en las carpetas `results/evaluation_*_v2`.

## Reto 1: dependencias nativas en Windows

**Problema documentado:** la instalación de `noise` y `box2d-py` exigió herramientas de compilación (C++ y SWIG). **Solución reportada:** preparar dichas dependencias antes de validar el entorno. **Aprendizaje:** reproducibilidad incluye el sistema operativo, las herramientas nativas y las versiones de paquetes, no solamente Python y `requirements.txt`.

## Reto 2: consistencia de las métricas del ambiente

**Problema documentado:** la posición horizontal absoluta no coincidía con la puntuación relativa al inicio; `info["dead"]` tampoco coincidía siempre con el estado de muerte usado internamente. **Solución reportada:** inspección del código del entorno y corrección del registro en `src/evaluate.py`, que obtiene `info["score"]` y consulta `raw_env.agent.dead`. **Aprendizaje:** no deben interpretarse las métricas sin verificar su semántica.

## Reto 3: mejora del entrenamiento sin éxito en evaluación

La curva de recompensa del entrenamiento presenta una tendencia favorable, pero las políticas greedy `best` y `final` obtuvieron **0 % de éxito**, murieron en **100 %** de los 20 episodios y alcanzaron puntuaciones medias de **2** y **0**, respectivamente. La recompensa media fue aproximadamente **-98.03** y **-97.06**, frente a **-657.99** de la política aleatoria. Sin embargo, el baseline alcanzó puntuación media **10.45**. **Aprendizaje:** optimizar el retorno registrado no equivale a maximizar progreso ni a resolver el entorno; se necesitan varias métricas.

## Reto 4: selección del checkpoint

El checkpoint `best` se seleccionó por la mejor media móvil de recompensa durante entrenamiento (20 episodios), no por desempeño greedy de validación. El `final` presenta una recompensa media levemente mayor en evaluación, pero puntuación media cero. **Aprendizaje:** el criterio de selección del modelo debe alinearse con el objetivo real y validarse fuera del entrenamiento.

## Reto 5: comportamiento de las acciones

En los pasos de evaluación, el modelo `best` aceleró en aproximadamente **80.02 %**; el `final` eligió reversa en **88.68 %** y aceleró en **0 %**. La concentración de acciones es un indicio de una política poco diversa, **no una prueba de la causa física del fracaso**. Para confirmar causas se requieren videos o trayectorias completas.

## Limitaciones e hipótesis para mejorar

- El estado de siete elementos no incorpora explícitamente el terreno por delante; esto podría limitar la anticipación, pero no se ha demostrado experimentalmente.
- No existe un estudio controlado de hiperparámetros: no puede atribuirse el fallo a epsilon, gamma o learning rate en particular.
- Los dos checkpoints proceden de una sola corrida; no permiten estimar la variabilidad entre entrenamientos.
- Una tasa observada de éxito del 0 % en 20 episodios no implica que la probabilidad de éxito sea universalmente nula.

## Propuestas de mejora priorizadas

1. Instrumentar trayectorias y videos para identificar causas de terminación y secuencias de acciones.
2. Incorporar evaluación greedy periódica en semillas de validación y seleccionar checkpoint por progreso y éxito.
3. Repetir entrenamientos con varias semillas y reportar dispersión.
4. Experimentar de forma controlada con recompensa, exploración y observaciones adicionales.

## Conclusión académica

El taller permitió implementar y ejecutar el ciclo completo de DQN y construir un procedimiento reproducible de evaluación. Su hallazgo central es que **la mejora de recompensa durante entrenamiento no garantiza una política de conducción exitosa**. La comparación entre retorno, progreso, duración, éxito y acciones evita conclusiones engañosas y orienta experimentos futuros.

**Fuentes internas:** `docs/resultados_actividad5.md`, `docs/evidencia_ejecucion_actividad5.md`, `results/config.json` y `results/evaluation_*_v2/metrics.json`.
