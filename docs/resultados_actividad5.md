# Actividad 5 — Resultados, evaluación y reflexión

Estado: borrador pendiente de validación con Actividad 4.

## 1. Metodología de evaluación

Se evaluaron dos checkpoints del agente DQN y un baseline aleatorio en
Hill Climb Racing Env:

- `checkpoint_best.pt`: checkpoint seleccionado durante el entrenamiento.
- `checkpoint_final.pt`: checkpoint al finalizar el entrenamiento.
- Política aleatoria: selección uniforme entre las tres acciones.

Cada política se ejecutó durante 20 episodios, utilizando las semillas
1000 a 1019. Los episodios de evaluación se ejecutaron separados del
entrenamiento, sin actualizar los pesos de la red.

Los DQN utilizaron una política greedy, con ε=0: en cada paso se eligió
la acción con mayor valor Q. El baseline utilizó un generador aleatorio
inicializado con la semilla de cada episodio.

Se mantuvieron el preprocesamiento de observaciones y la configuración
del entorno utilizados por el proyecto: tres acciones discretas,
recompensa `distance`, modalidad `aggressive` y `max_steps=1200`.
Este último parámetro limita los pasos sin progreso según la lógica
del entorno; no es un límite general de duración del episodio.

Se registraron recompensa acumulada sin descuento, puntuación máxima,
duración, terminación, truncamiento y frecuencia de acciones.

La puntuación del entorno corresponde al máximo avance relativo a la
posición inicial, convertido a entero. Se consideró éxito alcanzar
`MAX_SCORE=1000` con terminación y sin muerte.

La muerte se registró mediante `env.unwrapped.agent.dead`, que es el
indicador utilizado por `step()` para terminar el episodio. Durante la
auditoría se observó que `info["dead"]`, procedente del vehículo, no
siempre coincidía con este indicador.

## 2. Resultados del entrenamiento

El registro contiene 500 episodios. La curva muestra una mejora del
retorno durante el entrenamiento: al inicio predominan recompensas
negativas y posteriormente aparecen retornos positivos más frecuentes.

La media móvil de 20 episodios facilita la lectura de la tendencia.
La curva de episodios individuales conserva la variabilidad, incluidos
fallos pronunciados en etapas avanzadas.

![Curva de entrenamiento y media móvil de 20 episodios](../assets/training_curve.png)

Esta evolución constituye evidencia de mejora del retorno durante el
entrenamiento. Sin embargo, no demuestra por sí sola que la política
greedy resuelva el entorno, porque el entrenamiento incluye exploración
y actualizaciones de la red.

## 3. Resultados de evaluación independiente

| Métrica | DQN best | DQN final | Aleatorio |
|---|---:|---:|---:|
| Episodios | 20 | 20 | 20 |
| Recompensa media | -98.026 | -97.064 | -657.990 |
| Recompensa mediana | -98.043 | -97.025 | -272.078 |
| Desviación estándar de recompensa | 0.250 | 0.085 | 917.081 |
| Puntuación máxima por episodio: media | 2.00 | 0.00 | 10.45 |
| Puntuación máxima observada | 2 | 0 | 40 |
| Duración media en pasos | 50.05 | 53.00 | 449.70 |
| Tasa de éxito | 0 % | 0 % | 0 % |
| Tasa de muerte | 100 % | 100 % | 95 % |
| Tasa de truncamiento | 0 % | 0 % | 5 % |

La desviación estándar se calculó con `ddof=0` sobre los 20 retornos
registrados. Describe su dispersión y no representa un intervalo de
confianza.

![Comparación de recompensa, puntuación y duración](../assets/evaluation_comparison.png)

Los dos checkpoints DQN obtuvieron una recompensa media superior al
baseline aleatorio. Sin embargo, alcanzaron menor puntuación media y
terminaron por muerte en todos los episodios, aproximadamente a los
50 y 53 pasos.

El baseline logró mayor progreso y ejecutó más pasos, aunque sus retornos
fueron muy variables. La separación entre su media y mediana refleja
la influencia de varios episodios con penalizaciones acumuladas grandes.

Ninguna política alcanzó el objetivo. Por tanto, estos resultados no
permiten afirmar que el DQN haya resuelto el entorno ni que supere al
baseline de manera general: la comparación depende de la métrica.

## 4. Distribución de acciones

Los porcentajes se calcularon sobre el total de pasos de cada política.
Los episodios más largos contribuyen con más acciones al cálculo.

| Acción | DQN best | DQN final | Aleatorio |
|---|---:|---:|---:|
| Sin acelerar (0) | 8.89 % | 11.32 % | 33.64 % |
| Acelerar (1) | 80.02 % | 0.00 % | 33.09 % |
| Reversa (2) | 11.09 % | 88.68 % | 33.27 % |

![Distribución de acciones por política](../assets/evaluation_actions.png)

El checkpoint best presenta una preferencia por acelerar. El final
presenta una preferencia por reversa y no selecciona aceleración en
estos episodios. El baseline distribuye sus acciones aproximadamente
por igual, como corresponde a su selección uniforme.

Las frecuencias describen preferencias de acción, pero no muestran
su secuencia ni prueban qué maniobra produjo la muerte del vehículo.

## 5. Interpretación de la recompensa

En la configuración `distance` y `aggressive`, la recompensa depende de
la diferencia entre la posición horizontal y `prev_max_distance`:

- Si la diferencia es negativa, la recompensa es `-1 + diferencia`.
- Si está entre 0 y 0.001, sin incluir 0.001, la recompensa es -0.5.
- Si es al menos 0.001, la recompensa es `1 + diferencia`.
- La muerte o el atasco producen una penalización terminal de -100.

`prev_max_distance` se actualiza únicamente cuando el vehículo alcanza
un nuevo máximo horizontal y guarda el máximo anterior a ese avance.
Cuando no se establece un nuevo máximo, esta referencia permanece fija.

Por ello, la recompensa no equivale al incremento de puntuación.
Permanecer detrás de la referencia puede acumular penalizaciones durante
muchos pasos. Estar por encima de ella puede producir recompensa positiva
incluso en un paso que no establece un nuevo récord.

Como ambos DQN murieron en todos sus episodios, recibieron la penalización
terminal de -100. Antes de ella, acumularon en promedio aproximadamente
+1.974 para best y +2.936 para final. Esto explica aritméticamente su
pequeña diferencia de retorno, pero no identifica la trayectoria que
la produjo.

La terminación temprana también reduce la cantidad de pasos durante los
cuales pueden acumularse recompensas o penalizaciones. Su contribución
exacta a la comparación con el baseline requiere registros por paso.

## 6. Ejemplos concretos

### Ejemplo 1: comparación de checkpoints con semilla 1000

El checkpoint best obtuvo aproximadamente -98.04 de recompensa,
puntuación 2 y murió a los 50 pasos. El final obtuvo aproximadamente
-97.05, puntuación 0 y murió a los 53 pasos.

Este ejemplo muestra que una recompensa ligeramente mayor puede coexistir
con menor progreso. Ninguno de los checkpoints completó el objetivo.

### Ejemplo 2: baseline aleatorio con semilla 1006

El baseline obtuvo aproximadamente +205.10 de recompensa, alcanzó
puntuación 40 y terminó por muerte después de 592 pasos.

Fue el episodio con mayor puntuación del baseline. Ilustra que un retorno
positivo y un avance superior a los DQN evaluados tampoco implican éxito:
la puntuación permaneció lejos del umbral de 1000.

Estos ejemplos fueron seleccionados para ilustrar diferencias concretas;
las conclusiones generales se apoyan en los 20 episodios por política.

## 7. Reflexión sobre los resultados

El entrenamiento produjo una tendencia favorable en el retorno, pero
esa mejora no se trasladó a una política greedy capaz de completar el
entorno en las semillas evaluadas. Los checkpoints muestran preferencias
de acción claras, aunque los datos no permiten afirmar que hayan aprendido
una estrategia eficaz de conducción.

La baja dispersión de los DQN representa consistencia del fallo temprano.
No debe confundirse con estabilidad de una solución exitosa. La diferencia
de acciones entre best y final también indica que continuar entrenando
no garantizó conservar el comportamiento del checkpoint anterior.

Seleccionar un checkpoint mediante recompensa de entrenamiento puede
favorecer un modelo que no ofrece el mejor progreso durante evaluación.
En futuras ejecuciones conviene seleccionar checkpoints mediante
evaluaciones periódicas greedy en semillas de validación y reservar
otras semillas para la evaluación final.

El diseño de observaciones también plantea limitaciones. El vector de
siete componentes incluye posición, ángulo, velocidades de las ruedas y
contacto con el suelo, pero no representa explícitamente el terreno por
delante ni `prev_max_distance`, aunque esta última variable interviene
en la recompensa. Una MLP sin memoria puede tener dificultades para
distinguir situaciones con observaciones similares e historias distintas.
Esta es una hipótesis de diseño que requiere experimentos adicionales.

La sensibilidad a hiperparámetros no se midió mediante un estudio
controlado. No es posible atribuir los fallos a un valor concreto de
tasa de aprendizaje, exploración, descuento o actualización de la red
target. Para investigarlo se requieren varias semillas de entrenamiento
y cambios controlados de configuración.

## 8. Principales dificultades del proceso

La instalación en Windows requirió resolver dependencias de compilación:
`noise` necesitó las herramientas de C++ y `box2d-py` requirió SWIG.

Otra dificultad fue auditar los indicadores del entorno. La posición
horizontal absoluta no corresponde a la puntuación relativa al inicio,
y `info["dead"]` no coincidió siempre con el indicador de muerte utilizado
para terminar el episodio. Fue necesario revisar el código fuente y
corregir el registro de evaluación.

El reto de interpretación principal fue contrastar la mejora de la curva
de entrenamiento con los fallos de la política greedy. Comparar retorno,
progreso, duración y acciones evitó concluir que una recompensa mayor
demostraba una mejor solución.

## 9. Alcance y limitaciones

- Se evaluaron 20 semillas por política y una sola ejecución de
  entrenamiento, representada por dos checkpoints.
- La comparación entre checkpoints no mide variabilidad entre
  entrenamientos independientes.
- El baseline utiliza una secuencia aleatoria por semilla; no se realizaron
  varias repeticiones de acciones para cada terreno.
- El tramo inicial del terreno presenta una estructura similar entre
  semillas, según el código de generación. Cambiar la semilla no garantiza
  condiciones iniciales muy diferentes.
- No se registraron trayectorias completas ni videos. Las causas físicas
  específicas de muerte y las hipótesis sobre el aprendizaje siguen
  pendientes de comprobación.
- Un 0 % de éxito observado describe estos episodios; no prueba una
  probabilidad universal de éxito igual a cero.

## 10. Reproducción

Ejecutar desde la raíz del repositorio, con las dependencias instaladas:

```powershell
python src/evaluate.py --policy dqn --checkpoint results/checkpoint_best.pt --episodes 20 --seed 1000 --output results/evaluation_best_v2

python src/evaluate.py --policy dqn --checkpoint results/checkpoint_final.pt --episodes 20 --seed 1000 --output results/evaluation_final_v2

python src/evaluate.py --policy random --episodes 20 --seed 1000 --output results/evaluation_random_v2

python src/plot_results.py --logs results/logs.csv --output assets/training_curve.png

python src/plot_evaluation.py
```

Cada carpeta de evaluación contiene `evaluation.csv` y `metrics.json`.

Versiones registradas en la máquina de evaluación:

| Componente | Versión |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.14.1 |
| Gymnasium | 1.4.0 |
| NumPy | 2.5.4 |
| hill-climb-racing-env | 0.1.0 |
| box2d-py | 2.3.8 |
| noise | 1.2.2 |

## 11. Traspaso a Actividad 6

El material para integrar incluye:

- `src/evaluate.py`.
- `src/plot_evaluation.py`.
- Las tres figuras en `assets/`.
- Los CSV y JSON de las carpetas `evaluation_*_v2`.
- Este documento con métricas e interpretación.

Antes de integrar, Actividad 4 debe confirmar la correspondencia entre
checkpoints, logs, configuración y preprocesamiento del entrenamiento.
Los resultados corresponden a la evaluación local descrita; esa validación
del equipo todavía está pendiente.