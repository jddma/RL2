# Taller 2 — Agente Deep Q-Network en Hill Climb Racing

## Objetivo y entorno

Se implementa un agente DQN para `hill_racing_env/HillRacing-v0` con `action_space="discrete_3"`, `reward_function="distance"`, `reward_type="aggressive"` y `max_steps=1200`. Referencia del entorno: https://github.com/alexzh3/hillclimbracing . Debe verificarse que el entorno no haya sido utilizado previamente en clase.

## Observaciones, acciones y recompensas

La observación original es un `Dict` con `chassis_position` (`Box`, `(2,)`, `float32`, posición x/y), `chassis_angle` (`Box`, `(1,)`, `float32`, orientación), `wheels_speed` (`Box`, `(2,)`, `float32`, velocidad angular de ambas ruedas) y `on_ground` (`MultiBinary(2)`, habitualmente `int8`, contacto con suelo). El preprocesamiento de `src/env_utils.py` genera un vector `(7,)` `float32`: x/1000, y/700, ángulo/360, ambas velocidades divididas por `13π+0.1` y dos indicadores binarios. Esta representación evita que las diferencias de escala dominen las entradas de la MLP.

Acciones `Discrete(3)`: `0` sin aceleración, `1` acelerar, `2` reversa. La recompensa por distancia favorece superar la mejor posición horizontal previa; la modalidad agresiva penaliza inmovilidad, retroceso y episodios fallidos. El entorno distingue `terminated` (muerte u objetivo) de `truncated` (límite/atasco). El episodio finaliza con cualquiera; el target DQN solo bloquea bootstrap cuando `terminated=True`. La semántica concreta de las señales debe comprobarse en la versión instalada del entorno.

## Particularidades y flujo de entrenamiento

La inercia, las pendientes y los efectos diferidos de acciones justifican explorar trayectorias diversas y usar un descuento alto. El ciclo ejecuta: `reset → estado normalizado → ε-greedy → step → replay (s,a,r,s',terminated) → minibatch → Q_online(s,a) → max Q_target(s',·) → target Bellman → Huber → backpropagation → actualización online → sincronización periódica de target`. La red objetivo se mantiene separada para estabilizar el aprendizaje. El diagrama textual anterior es el pseudoflujo de referencia; su implementación está en `src/dqn.py`.

## Red neuronal e hiperparámetros

Arquitectura MLP `7 → 128 (ReLU) → 128 (ReLU) → 3 (lineal)`, con 17.923 parámetros. La salida representa tres valores Q, no probabilidades. La configuración oficial se carga desde `configs/dqn_main.yaml`; la tabla por capas y justificación de cada parámetro están en [docs/arquitectura.md](docs/arquitectura.md). Los valores son hipótesis de diseño hasta contrastarlos experimentalmente.

## Instalación y ejecución

Desde la raíz del proyecto:

```bash
python -m pip install -r requirements.txt
python src/inspect_env.py
python -m pytest -q tests
python src/train.py --config configs/dqn_main.yaml --output results
python src/evaluate.py --checkpoint results/checkpoint_final.pt --episodes 20 --seed 1000 --output results
python src/plot_results.py --logs results/logs.csv --output assets/training_curve.png
```

Para una prueba breve (no equivalente al experimento oficial):

```bash
python src/train.py --episodes 3 --learning-starts 10 --batch-size 4 --output results/smoke
```

## Reproducibilidad y evidencia

El entrenamiento registra seeds, configuración, recompensas por episodio, duración, epsilon, pérdidas, terminación y distancia estimada. Produce `results/logs.csv`, `results/config.json` y `results/checkpoint_final.pt`. La evaluación greedy separada genera `results/evaluation.csv` y `results/metrics.json`; las gráficas se producen desde logs reales. Las dependencias tienen rangos compatibles propuestos: para máxima reproducibilidad deben registrarse y fijarse versiones exactas del entorno y librerías tras una instalación validada.

## Resultados, reflexión y dificultades

No se incluyen métricas ni conclusiones empíricas sin ejecución real. Para completar la rúbrica deben incorporarse la curva de entrenamiento, métricas de evaluación greedy, comparación con baseline cuando sea posible, ejemplos concretos de conducta y análisis crítico de estabilidad, limitaciones y sensibilidad a hiperparámetros. También debe documentarse una dificultad real con su causa, intentos de solución, resultado y aprendizaje. La evaluación de éxito basada en distancia horizontal es una aproximación que debe cotejarse con el indicador oficial de éxito del entorno.
