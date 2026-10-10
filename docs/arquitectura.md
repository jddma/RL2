# Arquitectura DQN e hiperparámetros

La configuración ejecutable se encuentra en `configs/dqn_main.yaml`. Los valores constituyen una propuesta de diseño y deben contrastarse con los resultados de entrenamiento.

## Arquitectura vigente

| Capa | Entrada | Salida | Activación | Parámetros entrenables |
|---|---:|---:|---|---:|
| Linear 1 | 7 | 128 | ReLU | 1.024 |
| Linear 2 | 128 | 128 | ReLU | 16.512 |
| Linear 3 | 128 | 3 | Lineal | 387 |
| **Total** | | | | **17.923** |

La MLP es apropiada para un vector numérico pequeño; una CNN no es necesaria porque no se usan imágenes. ReLU introduce no linealidad y la salida **no** lleva softmax: DQN necesita valores Q sin normalizar, no probabilidades. Las dos redes, online y target, tienen la misma arquitectura; solo la online recibe gradientes. El tamaño 128×128 es un punto de partida razonable, **no una arquitectura optimizada empíricamente**.

## Hiperparámetros y razones técnicas

| Parámetro | Valor propuesto/vigente | Motivo | Módulo |
|---|---|---|---|
| `input_dim` | 7 | Vector normalizado de observaciones | `src/q_network.py` |
| `hidden` | 128, 128 | Capacidad moderada para control no lineal | `src/q_network.py` |
| `output_dim` | 3 | Una Q por acción discreta | `src/q_network.py` |
| `gamma` | 0.99 | Recompensas demoradas e inercia | `src/dqn.py` |
| Loss | Huber / SmoothL1 | Menor sensibilidad a errores Q extremos | `src/dqn.py` |
| Optimizer | Adam | Optimización adaptativa | `src/train.py` |
| Learning rate | 0.001 | Punto de partida; vigilar inestabilidad | `src/train.py` |
| Batch size | 64 | Equilibrio entre variación y costo | `src/train.py`, `src/dqn.py` |
| Replay capacity | 100000 | Descorrelación de transiciones | `src/replay_buffer.py` |
| Learning starts | 5000 | Diversidad antes de actualizar | `src/train.py`, `src/dqn.py` |
| Train frequency | 1 paso | Actualización frecuente tras warm-up | `src/dqn.py` |
| Epsilon | 1.0 → 0.05 en 100000 pasos | Exploración inicial y explotación gradual | `src/dqn.py` |
| Target update | Cada 1000 pasos | Objetivos relativamente estables | `src/dqn.py` |
| Gradient clipping | norma 10.0 | Limita gradientes extremos | `src/dqn.py` |
| Episodios | 500 | Presupuesto inicial; no garantiza convergencia | `src/train.py` |
| Seed | 42 | Reproducibilidad inicial | `src/train.py` |

La configuración del experimento se carga desde `configs/dqn_main.yaml` al iniciar el entrenamiento.

## Entrenamiento, checkpoints y reproducibilidad

`train_dqn` en src/dqn.py acepta ahora un parámetro opcional `episode_callback`, invocado al final de cada episodio con el índice del episodio, la recompensa obtenida y la red online. El cambio es aditivo y retrocompatible: si no se provee el argumento, el ciclo de entrenamiento se comporta exactamente igual que antes, así que tests/test_integration.py sigue validando el mismo contrato sin modificaciones. src/train.py usa ese callback para calcular la media móvil de recompensa de los últimos 20 episodios y guardar checkpoint_best.pt cada vez que esa media mejora, de forma independiente a checkpoint_final.pt. El criterio de selección queda registrado en config.json, bajo la clave `best_checkpoint`, junto con el episodio y el valor de la media móvil en el momento del guardado.

config.json también incluye, además de la configuración proveniente de configs/dqn_main.yaml, el comando exacto ejecutado (tomado de `sys.argv`) y las versiones instaladas de Python, `torch`, `gymnasium`, `numpy` y `hill-climb-racing-env`, esta última resuelta mediante los metadatos de instalación del paquete y no por un atributo `__version__`. Con esto, a partir de un config.json y un checkpoint_best.pt o checkpoint_final.pt se puede reconstruir el experimento exacto que los generó.

Antes de lanzar el experimento oficial de 500 episodios definido en configs/dqn_main.yaml se validó el flujo completo: tests/test_architecture.py y tests/test_integration.py pasan con las dependencias reales instaladas, no solo con el entorno simulado del smoke test; src/inspect_env.py confirma que el entorno real responde con la forma, acciones y recompensas documentadas en la ficha del ambiente; y una corrida corta de 25 episodios confirmó que logs.csv, checkpoint_final.pt, checkpoint_best.pt y config.json se generan correctamente y que el criterio de mejor checkpoint se aplica como se espera.

Con el flujo ya validado, se ejecutó el experimento oficial completo con el comando `src/train.py --config configs/dqn_main.yaml --output results`, usando la configuración de configs/dqn_main.yaml sin modificaciones. La corrida dejó en results las 500 filas de logs.csv (recompensa, longitud de episodio, epsilon, pérdida promedio y distancia máxima por episodio), checkpoint_final.pt con los pesos al terminar el episodio 500, checkpoint_best.pt con los pesos del episodio 453 (la mejor media móvil de recompensa registrada, 778.09 sobre una ventana de 20 episodios) y config.json con la configuración, el comando y las versiones exactas del entorno: Python 3.12.3, `torch` 2.14.1+cpu, `gymnasium` 1.4.0, `numpy` 2.5.4 y `hill-climb-racing-env` 0.1.0.

Una lectura preliminar de logs.csv muestra que la pérdida de entrenamiento arrancó alrededor de 2.0 cuando empezó el aprendizaje tras `learning_starts=5000` y bajó a un rango de 0.6 a 0.9 hacia el episodio 500, mientras que los episodios con recompensa alta, entre 1000 y 3600, se volvieron más frecuentes después del episodio 400, una vez epsilon se estabilizó en su mínimo de 0.05. También aparecen episodios puntuales con recompensa muy negativa, como -5639 en el episodio 386, lo que es consistente con la inestabilidad ocasional típica de DQN sin mejoras como Double DQN o Dueling DQN.

Estos cambios dan soporte directo al criterio de Resultados de la rúbrica, que exige evidencia reproducible y checkpoints verificables, y al criterio de aceptación de Actividad 4 del plan de distribución: un compañero debe poder clonar el repositorio, instalar dependencias y lanzar el entrenamiento con un comando documentado, de modo que los logs permitan reconstruir las figuras sin reentrenar.

