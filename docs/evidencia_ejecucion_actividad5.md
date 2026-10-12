# Evidencia de ejecución — Actividad 5

## Procedencia y alcance

Este documento organiza el registro de PowerShell aportado por Mauricio para la Actividad 5 de Hill Climb Racing DQN. Las ejecuciones se realizaron localmente en Windows, en la copia del repositorio `jddma/RL2`, rama `actividad5-evaluacion`.

Fuente: `registro_powershell_actividad5.txt`. El anexo conserva el texto recibido, incluidas salidas, errores, repeticiones y líneas cuya copia presenta mezcla de caracteres. Solo se normalizaron los saltos de línea para Markdown. No es una captura automática mediante `Start-Transcript`, ni un registro de ejecución en los servidores de GitHub. No se ejecutaron nuevamente los comandos al elaborar este documento.

El registro no contiene los comandos de clonación, creación del ambiente o instalación gráfica de las herramientas C++, ni una ejecución de entrenamiento. La curva se generó a partir de `results/logs.csv` existente. Las fechas visibles de archivos corresponden al 11 de octubre de 2026; no hay marcas temporales individuales para cada comando.

## 1. Preparación y dependencias

La activación desde `Unidad_4` falló porque `.venv` se encontraba dentro de `hill_climb_racing_dqn`. Cambiar la política de ejecución no resolvió esa ruta incorrecta. La activación funcionó tras entrar a la carpeta del proyecto.

```powershell
cd hill_climb_racing_dqn
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

La instalación de `box2d-py==2.3.8` falló con `swig.exe`. Se instaló SWIG y se repitió la instalación:

```powershell
uv pip install swig
swig -version
uv pip install -r requirements.txt
```

Resultado registrado: SWIG 4.5.1; compilación de `noise==1.2.2` y `box2d-py==2.3.8`; instalación de 34 paquetes. Una línea del segundo comando de instalación está visualmente mezclada en el registro, pero su salida queda conservada en el anexo.

| Componente | Versión registrada |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.14.1 |
| Gymnasium | 1.4.0 |
| NumPy | 2.5.4 |
| Hill Climb Racing Env | 0.1.0 |
| Box2D (`box2d-py`) | 2.3.8 |
| noise | 1.2.2 |

## 2. Inspección y curva de entrenamiento

```powershell
python src/inspect_env.py
python src/plot_results.py --logs results/logs.csv --output assets/training_curve.png
```

La inspección mostró observación procesada de dimensión `(7,)`, tipo `float32`, acciones `Discrete(3)` y cinco transiciones con recompensas y banderas de terminación. La generación de la curva produjo `assets/training_curve.png`. Esto evidencia inspección y graficación; no una nueva ejecución del entrenamiento.

## 3. Evaluaciones preliminares y auditoría

Las primeras evaluaciones se guardaron en `results/evaluation_best` y `results/evaluation_final`. Utilizaban distancia horizontal absoluta y todavía no incluían el registro final de acciones y muerte. No son las carpetas utilizadas para la tabla definitiva.

Se inspeccionó el código instalado con `inspect.getsource`: `reset`, `step`, generación del terreno, `Agent.update`, `_get_reward` y `update_status`. Se verificó `MAX_SCORE=1000` y que la puntuación es el avance máximo relativo a la posición inicial, convertido a entero.

La primera prueba de dos episodios en `check_best_v2` mostraba `muerte=False`. Tras corregir el indicador de muerte de `info["dead"]` a `raw_env.agent.dead`, la repetición mostró `muerte=True` y `death_rate=1.0`, sin cambiar el retorno. El texto del cambio fue realizado en el editor y no aparece como comando en el registro; su efecto sí aparece en ambas salidas.

El intento de importar `env_utils` desde la raíz mediante `python -c` falló. La inspección funcionó al añadir `sys.path.insert(0, 'src')`.

## 4. Evaluaciones finales

Se ejecutaron tres políticas, 20 episodios cada una, con semillas 1000–1019. Los checkpoints DQN se evaluaron greedy (ε=0) con el script de evaluación del proyecto; la salida identifica las políticas, checkpoints, semillas y episodios. El registro de consola debe revisarse junto con `src/evaluate.py` para verificar los detalles de implementación, incluida la ausencia de actualizaciones de pesos.

```powershell
python src/evaluate.py --policy dqn --checkpoint results/checkpoint_best.pt --episodes 20 --seed 1000 --output results/evaluation_best_v2
python src/evaluate.py --policy dqn --checkpoint results/checkpoint_final.pt --episodes 20 --seed 1000 --output results/evaluation_final_v2
python src/evaluate.py --policy random --episodes 20 --seed 1000 --output results/evaluation_random_v2
```

| Métrica | DQN best | DQN final | Aleatorio |
|---|---:|---:|---:|
| Recompensa media | -98.026 | -97.064 | -657.990 |
| Mediana | -98.043 | -97.025 | -272.078 |
| Desviación estándar | 0.250 | 0.085 | 917.081 |
| Puntuación media | 2.00 | 0.00 | 10.45 |
| Puntuación máxima observada | 2 | 0 | 40 |
| Pasos medios | 50.05 | 53.00 | 449.70 |
| Éxito | 0 % | 0 % | 0 % |
| Muerte | 100 % | 100 % | 95 % |
| Truncamiento | 0 % | 0 % | 5 % |

Las tres carpetas finales contienen `evaluation.csv` y `metrics.json`, según el listado registrado. Los DQN obtienen retornos medios mayores, pero menor progreso y duración. Ninguna política alcanza el objetivo en estos episodios. Las causas físicas concretas del fallo no quedan demostradas por las salidas agregadas.

## 5. Figuras y comprobaciones

```powershell
python src/plot_evaluation.py
python -m py_compile src/evaluate.py src/plot_evaluation.py
```

El primer comando informó la creación de `assets/evaluation_comparison.png` y `assets/evaluation_actions.png`, con 20 episodios por política. El segundo volvió al prompt sin errores; comprueba sintaxis, no sustituye una prueba funcional. Las evaluaciones y la generación de figuras aportan la evidencia de ejecución funcional.

## 6. Paquete de entrega

El bloque de Python ejecutado desde PowerShell creó `Actividad5_Resultados_Evaluacion.zip` con 12 archivos y verificó su integridad mediante `ZipFile.testzip()`.

La consulta del ZIP falló mientras la terminal estaba en `Unidad_4` porque se usó `..\`. Funcionó tras regresar a la raíz del proyecto:

```powershell
python -m zipfile -l ..\Actividad5_Resultados_Evaluacion.zip
```

El paquete contiene dos scripts, el documento de resultados, tres figuras y seis archivos de evaluación. No incluye pruebas preliminares, ambiente virtual ni checkpoints.

## 7. Historial Git y estado de publicación

Se verificaron la ruta local, el remoto `https://github.com/jddma/RL2.git` y la rama `actividad5-evaluacion`. Después de `git fetch origin`, ambas ramas locales estaban en `2695aae`, coincidente con `origin/master` en ese momento.

Se seleccionaron los 12 entregables con `git add`. `git diff --cached --check` terminó sin mensajes. El commit se creó correctamente:

```text
[actividad5-evaluacion 7976d3d] Actividad 5: evaluacion greedy, baseline y analisis de resultados
12 files changed, 692 insertions(+), 21 deletions(-)
```

El intento de publicación fue:

```powershell
git push -u origin actividad5-evaluacion
```

```text
remote: Permission to jddma/RL2.git denied to mauricioplascencia.
fatal: unable to access 'https://github.com/jddma/RL2.git/': The requested URL returned error: 403
```

**Estado al cierre de la fuente:** commit local creado; publicación no realizada por falta de permiso. No hay evidencia posterior de aceptación de invitación, push exitoso o integración en `master`. Este nuevo documento tampoco forma parte del commit `7976d3d` ni del ZIP original.

## 8. Ubicación de evidencias en la fuente

Las líneas se cuentan sobre el archivo recibido, comenzando en 1.

| Hito | Línea inicial |
|---|---:|
| Instalación de SWIG | 155 |
| Inspección del entorno | 207 |
| Evaluación final DQN best | 498 |
| Evaluación final DQN final | 541 |
| Evaluación final aleatoria | 584 |
| Figuras comparativas | 649 |
| Verificación de sintaxis | 769 |
| Commit local | 1011 |
| Push rechazado | 1025 |

SHA-256 del archivo fuente original: `0577b6dda307703fcdb3042d04ff94595d777eca2aa1970676f4dfa5e716f5f3`.

## Anexo — Registro completo aportado

El texto siguiente conserva el orden de las ejecuciones y sus salidas. Incluye pruebas previas que no deben confundirse con las evaluaciones finales de las carpetas `evaluation_*_v2`.

````text
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

PS C:\Users\Mauricio> cd..
PS C:\Users> cd..
PS C:\> cd IA
PS C:\IA> dir


    Directorio: C:\IA


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      27/09/2026  1:18 p. m.                Aprendizaje_Maquina_2
d-----       3/09/2026  7:36 p. m.                Gymnasium
d-----      11/10/2026  5:33 p. m.                Simulacion_Aprendizaje_Refuerzo
-a----       2/10/2026  9:16 a. m.          28836 estructura_IA


PS C:\IA> cd Simulacion_Aprendizaje_Refuerzo
PS C:\IA\Simulacion_Aprendizaje_Refuerzo> dir


    Directorio: C:\IA\Simulacion_Aprendizaje_Refuerzo


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      29/09/2026  1:23 p. m.                RL
d-----      13/09/2026  2:05 p. m.                Unidad_2
d-----      28/09/2026  5:22 p. m.                Unidad_3
d-----      11/10/2026  5:35 p. m.                Unidad_4


PS C:\IA\Simulacion_Aprendizaje_Refuerzo> cd Unidad_4
PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> dir


    Directorio: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      11/10/2026  5:38 p. m.                hill_climb_racing_dqn


PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> .\.venv\Scripts\Activate.ps1
.\.venv\Scripts\Activate.ps1 : El término '.\.venv\Scripts\Activate.ps1' no se reconoce como nombre de un cmdlet,
función, archivo de script o programa ejecutable. Compruebe si escribió correctamente el nombre o, si incluyó una ruta
de acceso, compruebe que dicha ruta es correcta e inténtelo de nuevo.
En línea: 1 Carácter: 1
+ .\.venv\Scripts\Activate.ps1
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (.\.venv\Scripts\Activate.ps1:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> Set-ExecutionPolicy –Scope Process –ExecutionPolicy Bypass
PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> .\.venv\Scripts\Activate.ps1
.\.venv\Scripts\Activate.ps1 : El término '.\.venv\Scripts\Activate.ps1' no se reconoce como nombre de un cmdlet,
función, archivo de script o programa ejecutable. Compruebe si escribió correctamente el nombre o, si incluyó una ruta
de acceso, compruebe que dicha ruta es correcta e inténtelo de nuevo.
En línea: 1 Carácter: 1
+ .\.venv\Scripts\Activate.ps1
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (.\.venv\Scripts\Activate.ps1:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> dir


    Directorio: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      11/10/2026  5:38 p. m.                hill_climb_racing_dqn


PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> dir


    Directorio: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      11/10/2026  5:38 p. m.                hill_climb_racing_dqn


PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> cd hill_climb_racing_dqn
PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> dir


    Directorio: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      11/10/2026  5:40 p. m.                .venv
d-----      11/10/2026  5:35 p. m.                assets
d-----      11/10/2026  5:35 p. m.                configs
d-----      11/10/2026  5:35 p. m.                docs
d-----      11/10/2026  5:35 p. m.                results
d-----      11/10/2026  5:35 p. m.                src
d-----      11/10/2026  5:35 p. m.                tests
-a----      11/10/2026  5:35 p. m.             44 .gitignore
-a----      11/10/2026  5:35 p. m.           4468 README.md
-a----      11/10/2026  5:35 p. m.            109 requirements.txt


PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> .\.venv\Scripts\Activate.ps1
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> code .
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> uv pip install -r requirements.txt
Resolved 34 packages in 82ms
  x Failed to build `box2d-py==2.3.8`
  |-> The build backend returned an error
  `-> Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit code: 1)

      [stdout]
      Using setuptools (version 84.0.0).
      running bdist_wheel
      running build
      running build_py
      copying library\Box2D\__init__.py -> build\lib.win-amd64-cpython-312\Box2D
      running build_ext
      building 'Box2D._Box2D' extension
      swigging Box2D\Box2D.i to Box2D\Box2D_wrap.cpp
      swig.exe -python -c++ -IBox2D -small -O -includeall -ignoremissing -w201 -globals b2Globals -outdir
      library\Box2D -keyword -w511 -D_SWIG_KWARGS -o Box2D\Box2D_wrap.cpp Box2D\Box2D.i

      [stderr]
      C:\Users\Mauricio\AppData\Local\uv\cache\builds-v0\.tmpbolBPp\Lib\site-packages\setuptools\_distutils\dist.py:318:
      UserWarning: Unknown distribution option: 'test_suite'
        warnings.warn(msg)
      C:\Users\Mauricio\AppData\Local\uv\cache\builds-v0\.tmpbolBPp\Lib\site-packages\setuptools\dist.py:765:
      SetuptoolsDeprecationWarning: License classifiers are deprecated.
      !!

              ********************************************************************************
              Please consider removing the following classifiers in favor of a SPDX license expression:

              License :: OSI Approved :: zlib/libpng License

              See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
              ********************************************************************************

      !!
        self._finalize_license_expression()
      error: command 'swig.exe' failed: None


hint: `box2d-py` (v2.3.8) was included because `hill-climb-racing-env` (v0.1.0) depends on `box2d-py`
hint: Build failures usually indicate a problem with the package or the build environment
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> uv pip install swig
Resolved 1 package in 630ms
Prepared 1 package in 2.92s
Installed 1 package in 319ms
 + swig==4.5.1
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> swig -version

SWIG Version 4.5.1

Compiled with x86_64-w64-mingw32-g++ [x86_64-w64-mingw32]

Configured options: +pcre

Please see https://www.swig.org for reporting bugs and further information
(hill_climb_racing_dquv pip install -r requirements.txt_Refuerzo\Unidad_4\hill_climb_racing_dqn>
Resolved 34 packages in 14ms
      Built noise==1.2.2                                                                                                                                          Built box2d-py==2.3.8                                                                                                                                 Prepared 12 packages in 2m 36s
Installed 34 packages in 2.41s
 + box2d-py==2.3.8
 + cloudpickle==3.1.2
 + colorama==0.4.6
 + contourpy==1.4.0
 + cycler==0.12.1
 + farama-notifications==0.0.6
 + filelock==4.1.1
 + fonttools==4.66.1
 + fsspec==2026.9.0
 + gymnasium==1.4.0
 + hill-climb-racing-env==0.1.0
 + iniconfig==2.3.1
 + jinja2==3.1.6
 + kiwisolver==1.5.1
 + markupsafe==3.0.4
 + matplotlib==3.11.2
 + mpmath==1.3.0
 + networkx==3.7
 + noise==1.2.2
 + numpy==2.5.4
 + packaging==26.3
 + pillow==12.3.0
 + pluggy==1.6.0
 + pygame==2.6.1
 + pygments==2.21.0
 + pyparsing==3.3.3
 + pytest==9.1.1
 + python-dateutil==2.9.0.post0
 + pyyaml==6.0.3
 + setuptools==84.0.0
 + six==1.17.0
 + sympy==1.14.0
 + torch==2.14.1
 + typing-extensions==4.16.0
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python src/inspect_env.py
Observacion procesada: [0.00666667 0.02869282 0.         0.         0.         0.
 0.        ] shape: (7,) dtype: float32
Acciones: Discrete(3)
Accion 0 reward -0.5 terminated False truncated False shape (7,)
Accion 1 reward -1.0043913523356123 terminated False truncated False shape (7,)
Accion 1 reward -1.0095440546671552 terminated False truncated False shape (7,)
Accion 2 reward -1.0095779101053877 terminated False truncated False shape (7,)
Accion 1 reward -1.0145522753397627 terminated False truncated False shape (7,)
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python src/plot_results.py --logs results/logs.csv --output assets/training_curve.png
assets\training_curve.png
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python src/evaluate.py --checkpoint results/checkpoint_best.pt --episodes 20 --seed 1000 --output results/evaluation_best
{
  "episodes": 20,
  "reward_mean": -98.02572204271955,
  "reward_median": -98.04341689745587,
  "reward_std": 0.2504280272904514,
  "distance_mean": 8.775853551924229,
  "success_rate": 0.0,
  "truncation_rate": 0.0
}
(hill_climb_racing_dqImport-Csv results/evaluation_best/evaluation.csv | Format-Table -AutoSize>

episode reward             max_distance      steps terminated truncated success
------- ------             ------------      ----- ---------- --------- -------
1       -98.0370313326518  8.81535280495882  50    True       False     False
2       -98.03510967890423 8.817545138299465 50    True       False     False
3       -98.1300350824992  8.721036836504936 50    True       False     False
4       -98.13104836146039 8.719888515770435 50    True       False     False
5       -98.04082838694256 8.810979314148426 50    True       False     False
6       -98.13078180948895 8.720199577510357 50    True       False     False
7       -98.03431336085004 8.818471804261208 50    True       False     False
8       -98.0374910036723  8.814824745059013 50    True       False     False
9       -98.04541842142743 8.805729448795319 50    True       False     False
10      -98.13150183359784 8.719393983483315 50    True       False     False
11      -96.95213524500531 8.898990228772163 51    True       False     False
12      -98.13048092524213 8.720550686120987 50    True       False     False
13      -98.04141537348431 8.810263127088547 50    True       False     False
14      -98.133793036143   8.716748096048832 50    True       False     False
15      -98.13027111689252 8.720766752958298 50    True       False     False
16      -98.13100163141888 8.719950914382935 50    True       False     False
17      -98.03451649347943 8.818250149488449 50    True       False     False
18      -98.0371247927348  8.815242908895016 50    True       False     False
19      -98.1313459078471  8.719565346837044 50    True       False     False
20      -98.03879706064862 8.81332065910101  50    True       False     False


(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python src/evaluate.py --checkpoint results/checkpoint_final.pt --episodes 20 --seed 1000 --output results/evaluation_final
{
  "episodes": 20,
  "reward_mean": -97.06439408461254,
  "reward_median": -97.02522031466167,
  "reward_std": 0.08459510840245249,
  "distance_mean": 6.964009860530496,
  "success_rate": 0.0,
  "truncation_rate": 0.0
}
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python -c "import sys; from importlib.metadata import version; print('Python:', sys.version); [print(p + ':', version(p)) for p in ['torch', 'gymnasium', 'numpy', 'hill-climb-racing-env', 'box2d-py', 'noise']]"
Python: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
torch: 2.14.1
gymnasium: 1.4.0
numpy: 2.5.4
hill-climb-racing-env: 0.1.0
box2d-py: 2.3.8
noise: 1.2.2
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python -c "import gymnasium as gym; import hill_racing_env; import inspect; e=gym.make('hill_racing_env/HillRacing-v0'); print('Archivo del entorno:', inspect.getfile(type(e.unwrapped))); print('RESET:'); print(inspect.getsource(type(e.unwrapped).reset)); print('STEP:'); print(inspect.getsource(type(e.unwrapped).step)); e.close()"
Archivo del entorno: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn\.venv\Lib\site-packages\hill_racing_env\envs\hill_racing.py
RESET:
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        info = {}
        # Destroy world
        self._destroy_world()
        # Generate new world
        self.world.contactListener = ContactListener()
        self._generate_ground(seed=seed)
        self._generate_agent()
        self.step_stuck_counter = 0  # Set step counter to 0
        self.step_counter = 0
        self.position_list.clear()  # Empty list
        self.total_airtime_counter = 0
        # Get the initial observations
        observations = self._get_obs()
        # Render mode
        if self.render_mode == "human":
            self.render()

        return observations, info

STEP:
    def step(self, action: int | np.float32):
        terminated = False
        truncated = False
        reward = 0  # initial reward of -1, if the agent does completely nothing

        # Execute action
        self._execute_action(action)
        # Step forward in the world
        self.world.Step(timeStep=1.0 / self.metadata["render_fps"], velocityIterations=6 * 30,
                        positionIterations=2 * 30)
        # Increase step counter
        self.step_stuck_counter += 1
        self.step_counter += 1
        # Increase airtime counter
        if self.agent.total_airtime > 0:
            self.total_airtime_counter += self.agent.total_airtime
        # Update position
        self.position_list.append((int(self.agent.car.pos_x), self.step_counter))
        # Update agent status
        self.agent.update()

        # Check if agent is stuck
        if (math.floor(self.agent.car.pos_x) % 20 == 0 and self.previous_stuck_pos !=
                math.floor(self.agent.car.pos_x)):  # when we made more
            # than 20 metres distance reset count, and we are not at the same position we were stuck in
            self.step_stuck_counter = 0
            self.previous_stuck_pos = math.floor(self.agent.car.pos_x)
        else:  # When no significant distance has been made for a long time, the agent must be stuck
            if self.step_stuck_counter > self.max_steps:
                truncated = True
                reward = -100

        # If agent is dead
        if self.agent.dead:
            terminated = True
            reward = -100
        elif self.agent.score >= MAX_SCORE:  # If max score is achieved
            terminated = True

        # Reward shaping if agent is still alive or not stuck
        if not truncated and not terminated:
            reward = self._get_reward(action)

        # Get the current step observation and info for debugging
        observation = self._get_obs()

        info = {
            "car_position": (self.agent.car.pos_x, self.agent.car.pos_y),
            "prev_max_distance": self.agent.car.prev_max_distance,
            "score": self.agent.score,
            "dead": self.agent.car.dead,
            "steps_stuck": self.step_stuck_counter,
            "airtime_counter": self.agent.airtime_counter,
            "total_airtime": self.total_airtime_counter,
            "on_ground": (self.agent.car.wheels[0].on_ground, self.agent.car.wheels[1].on_ground),
            "position_list": self.position_list
        }
        # print(reward, info, observation, action)  # For debugging purposes
        return observation, reward, terminated, truncated, info

(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python -c "import gymnasium as gym; import hill_racing_env; import inspect; e=gym.make('hill_racing_env/HillRacing-v0'); c=type(e.unwrapped); print('GENERACIÓN DEL TERRENO:'); print(inspect.getsource(c._generate_ground)); print('MAX_SCORE:', inspect.getmodule(c).MAX_SCORE); e.close()"
GENERACIÓN DEL TERRENO:
    def _generate_ground(self, seed: Optional[int] = None):
        # Variables
        ground_template = ground.Ground(original_noise=self.original_noise)  # Template to store the ground vectors
        ground_template.randomize_ground(seed=seed)  # Randomizes the ground using the difficulty and perlin noise

        # Generate until we find ground that is not too steep
        while ground_template.groundTooSteep():
            ground_template = ground.Ground(original_noise=self.original_noise)
            ground_template.randomize_ground(seed=seed)

        # Add the ground to the world
        self.ground = ground.Ground(self.world)
        self.ground.cloneFrom(ground_template)  # Copy the ground_template to self.ground
        self.ground.setBodies(self.world)  # Add the bodies to the world

MAX_SCORE: 1000
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python -c "import gymnasium as gym; import hill_racing_env; import inspect; e=gym.make('hill_racing_env/HillRacing-v0'); e.reset(seed=1000); print(inspect.getsource(type(e.unwrapped.ground).randomize_ground)); e.close()"
    def randomize_ground(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        ground_seed = random.uniform(0, 100000)  # Generates a random seed that will define the terrain
        # Minimum height of ground
        min_height = 30
        # Set the length of the flat section of the ground vector
        flat_length = 500
        # Initialize a variable to store the additional height for flat ground
        height_addition = 0

        # Iterate over a range from 0 to self.distance with a step size of self.smoothness
        for i in range(0, self.distance, self.smoothness):
            # Calculate the steepness level by remapping the current distance to a height
            self.steepness_Level = np.interp(i, [0, self.distance], [130, 250])
            # Calculate the noised_y value using Perlin noise with the starting point and adjusted i value
            if self.original_noise is True:  # Whether to use the original perlin noise (0 to 1)
                flat_length = 0
                noised_y = abs(
                    perlin.original_pnoise(ground_seed + (i - flat_length) / (700 - self.steepness_Level)))
            else:  # Use perlin noise from -1 to 1
                noised_y = abs(
                    noise.pnoise1(ground_seed + (i - flat_length) / (700 - self.steepness_Level), octaves=4))
            # Determine the maximum and minimum heights for the ground vector based on the steepness level.
            max_height = hill_racing.DIFFICULTY + np.interp(self.steepness_Level, [0, 200], [0, 320])
            # Function where difficulty increases till the end
            # max_height = hill_racing.DIFFICULTY + np.interp(self.steepness_Level, [0, 250], [0, 320])
            # If value is less than the flat section length, recalculate noised_y and height_addition
            if i < flat_length:
                noised_y = abs(noise.pnoise1(ground_seed, octaves=4))
                height_addition = (flat_length - i) / 25
            # Create the ground vector with x-value i and adjusted y-value based on noised_y
            self.ground_vectors.append(
                b2Vec2(i, hill_racing.SCREEN_HEIGHT -
                       np.interp(noised_y, [0, 1], [min_height, max_height]) + height_addition))

        self.ground_vectors.append(b2Vec2(self.distance, hill_racing.SCREEN_HEIGHT))  # End point vector
        self.ground_vectors.append(b2Vec2(0, hill_racing.SCREEN_HEIGHT))  # Starting point vector
        hill_racing.SPAWNING_Y = self.ground_vectors[10].y - 100  # Calculate spawn location

        for vect in self.ground_vectors:
            vect.x /= hill_racing.SCALE
            vect.y /= hill_racing.SCALE

(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python -c "import gymnasium as gym; import hill_racing_env; import inspect; e=gym.make('hill_racing_env/HillRacing-v0'); e.reset(seed=1000); print('AGENT.UPDATE:'); print(inspect.getsource(type(e.unwrapped.agent).update)); e.close()"
AGENT.UPDATE:
    def update(self):
        # print(self.car.chassis_body.position.x, self.car.max_distance)
        # print(self.car.wheels[0].joint.speed, self.car.wheels[1].joint.speed)
        # Update the panX and panY offset for camera
        hill_racing.panX = self.car.chassis_body.position.x * hill_racing.SCALE - 100
        # hill_racing.panY = self.car.chassis_body.position.y * hill_racing.SCALE - hill_racing.SPAWNING_Y

        if self.car.dead:  # If the car is dead
            self.dead = True
        elif not self.car.dead:
            self.car.update_status()

        # Calculate and update score, score is equal to at least 0 or current max distance - spawn position
        self.score = max(0, int((self.car.max_distance - (self.spawning_x / hill_racing.SCALE))))
        # Count number of steps where agent is in the air
        if not self.car.wheels[0].on_ground and not self.car.wheels[1].on_ground:
            self.total_airtime = 0
            self.steps_in_air += 1
            if self.steps_in_air >= 30:  # Start counting airtime when we have more than 30 frames in air
                self.airtime_counter += 1
                self.steps_in_air = 0
        else:
            self.total_airtime = self.airtime_counter   # How much airtime we end with when touching ground
            self.airtime_counter = 0
            self.steps_in_air = 0

(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python src/evaluate.py --policy dqn --checkpoint results/checkpoint_best.pt --episodes 2 --seed 1000 --output results/check_best_v2
Episodio 01 | seed=1000 | reward=-98.04 | score=2 | pasos=50 | muerte=False | éxito=False
Episodio 02 | seed=1001 | reward=-98.04 | score=2 | pasos=50 | muerte=False | éxito=False

Métricas:
{
  "policy": "dqn",
  "checkpoint": "results/checkpoint_best.pt",
  "episodes": 2,
  "seed_start": 1000,
  "seed_end": 1001,
  "success_threshold": 1000,
  "reward_mean": -98.03607050577801,
  "reward_median": -98.03607050577801,
  "reward_std": 0.0009608268737792969,
  "score_mean": 2.0,
  "score_max": 2,
  "steps_mean": 50.0,
  "success_rate": 0.0,
  "death_rate": 0.0,
  "truncation_rate": 0.0,
  "idle_fraction": 0.08,
  "gas_fraction": 0.8,
  "reverse_fraction": 0.12
}
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python src/evaluate.py --policy dqn --checkpoint results/checkpoint_best.pt --episodes 2 --seed 1000 --output results/check_best_v2
Episodio 01 | seed=1000 | reward=-98.04 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 02 | seed=1001 | reward=-98.04 | score=2 | pasos=50 | muerte=True | éxito=False

Métricas:
{
  "policy": "dqn",
  "checkpoint": "results/checkpoint_best.pt",
  "episodes": 2,
  "seed_start": 1000,
  "seed_end": 1001,
  "success_threshold": 1000,
  "reward_mean": -98.03607050577801,
  "reward_median": -98.03607050577801,
  "reward_std": 0.0009608268737792969,
  "score_mean": 2.0,
  "score_max": 2,
  "steps_mean": 50.0,
  "success_rate": 0.0,
  "death_rate": 1.0,
  "truncation_rate": 0.0,
  "idle_fraction": 0.08,
  "gas_fraction": 0.8,
  "reverse_fraction": 0.12
}
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python src/evaluate.py --policy dqn --checkpoint results/checkpoint_best.pt --episodes 20 --seed 1000 --output results/evaluation_best_v2
Episodio 01 | seed=1000 | reward=-98.04 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 02 | seed=1001 | reward=-98.04 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 03 | seed=1002 | reward=-98.13 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 04 | seed=1003 | reward=-98.13 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 05 | seed=1004 | reward=-98.04 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 06 | seed=1005 | reward=-98.13 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 07 | seed=1006 | reward=-98.03 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 08 | seed=1007 | reward=-98.04 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 09 | seed=1008 | reward=-98.05 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 10 | seed=1009 | reward=-98.13 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 11 | seed=1010 | reward=-96.95 | score=2 | pasos=51 | muerte=True | éxito=False
Episodio 12 | seed=1011 | reward=-98.13 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 13 | seed=1012 | reward=-98.04 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 14 | seed=1013 | reward=-98.13 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 15 | seed=1014 | reward=-98.13 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 16 | seed=1015 | reward=-98.13 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 17 | seed=1016 | reward=-98.03 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 18 | seed=1017 | reward=-98.04 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 19 | seed=1018 | reward=-98.13 | score=2 | pasos=50 | muerte=True | éxito=False
Episodio 20 | seed=1019 | reward=-98.04 | score=2 | pasos=50 | muerte=True | éxito=False

Métricas:
{
  "policy": "dqn",
  "checkpoint": "results/checkpoint_best.pt",
  "episodes": 20,
  "seed_start": 1000,
  "seed_end": 1019,
  "success_threshold": 1000,
  "reward_mean": -98.02572204271955,
  "reward_median": -98.04341689745587,
  "reward_std": 0.2504280272904514,
  "score_mean": 2.0,
  "score_max": 2,
  "steps_mean": 50.05,
  "success_rate": 0.0,
  "death_rate": 1.0,
  "truncation_rate": 0.0,
  "idle_fraction": 0.08891108891108891,
  "gas_fraction": 0.8001998001998002,
  "reverse_fraction": 0.1108891108891109
}
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python src/evaluate.py --policy dqn --checkpoint results/checkpoint_final.pt --episodes 20 --seed 1000 --output results/evaluation_final_v2
Episodio 01 | seed=1000 | reward=-97.05 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 02 | seed=1001 | reward=-97.01 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 03 | seed=1002 | reward=-97.00 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 04 | seed=1003 | reward=-97.02 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 05 | seed=1004 | reward=-97.12 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 06 | seed=1005 | reward=-97.01 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 07 | seed=1006 | reward=-97.00 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 08 | seed=1007 | reward=-97.06 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 09 | seed=1008 | reward=-97.27 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 10 | seed=1009 | reward=-97.03 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 11 | seed=1010 | reward=-97.28 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 12 | seed=1011 | reward=-97.00 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 13 | seed=1012 | reward=-97.20 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 14 | seed=1013 | reward=-97.07 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 15 | seed=1014 | reward=-97.00 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 16 | seed=1015 | reward=-97.02 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 17 | seed=1016 | reward=-97.00 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 18 | seed=1017 | reward=-97.05 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 19 | seed=1018 | reward=-97.02 | score=0 | pasos=53 | muerte=True | éxito=False
Episodio 20 | seed=1019 | reward=-97.08 | score=0 | pasos=53 | muerte=True | éxito=False

Métricas:
{
  "policy": "dqn",
  "checkpoint": "results/checkpoint_final.pt",
  "episodes": 20,
  "seed_start": 1000,
  "seed_end": 1019,
  "success_threshold": 1000,
  "reward_mean": -97.06439408461254,
  "reward_median": -97.02522031466167,
  "reward_std": 0.08459510840245249,
  "score_mean": 0.0,
  "score_max": 0,
  "steps_mean": 53.0,
  "success_rate": 0.0,
  "death_rate": 1.0,
  "truncation_rate": 0.0,
  "idle_fraction": 0.11320754716981132,
  "gas_fraction": 0.0,
  "reverse_fraction": 0.8867924528301887
}
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python src/evaluate.py --policy random --episodes 20 --seed 1000 --output results/evaluation_random_v2
Episodio 01 | seed=1000 | reward=-90.54 | score=15 | pasos=208 | muerte=True | éxito=False
Episodio 02 | seed=1001 | reward=133.25 | score=16 | pasos=384 | muerte=True | éxito=False
Episodio 03 | seed=1002 | reward=-1327.32 | score=2 | pasos=584 | muerte=True | éxito=False
Episodio 04 | seed=1003 | reward=-371.63 | score=7 | pasos=337 | muerte=True | éxito=False
Episodio 05 | seed=1004 | reward=-2610.58 | score=21 | pasos=862 | muerte=True | éxito=False
Episodio 06 | seed=1005 | reward=-217.70 | score=2 | pasos=190 | muerte=True | éxito=False
Episodio 07 | seed=1006 | reward=205.10 | score=40 | pasos=592 | muerte=True | éxito=False
Episodio 08 | seed=1007 | reward=-585.68 | score=23 | pasos=663 | muerte=True | éxito=False
Episodio 09 | seed=1008 | reward=-2972.55 | score=1 | pasos=1201 | muerte=False | éxito=False
Episodio 10 | seed=1009 | reward=-2329.55 | score=4 | pasos=979 | muerte=True | éxito=False
Episodio 11 | seed=1010 | reward=-684.63 | score=12 | pasos=602 | muerte=True | éxito=False
Episodio 12 | seed=1011 | reward=175.31 | score=25 | pasos=330 | muerte=True | éxito=False
Episodio 13 | seed=1012 | reward=-101.82 | score=5 | pasos=135 | muerte=True | éxito=False
Episodio 14 | seed=1013 | reward=-201.42 | score=14 | pasos=391 | muerte=True | éxito=False
Episodio 15 | seed=1014 | reward=-938.38 | score=9 | pasos=538 | muerte=True | éxito=False
Episodio 16 | seed=1015 | reward=-62.41 | score=1 | pasos=110 | muerte=True | éxito=False
Episodio 17 | seed=1016 | reward=0.09 | score=8 | pasos=148 | muerte=True | éxito=False
Episodio 18 | seed=1017 | reward=-326.45 | score=0 | pasos=152 | muerte=True | éxito=False
Episodio 19 | seed=1018 | reward=-163.44 | score=4 | pasos=221 | muerte=True | éxito=False
Episodio 20 | seed=1019 | reward=-689.44 | score=0 | pasos=367 | muerte=True | éxito=False

Métricas:
{
  "policy": "random",
  "checkpoint": null,
  "episodes": 20,
  "seed_start": 1000,
  "seed_end": 1019,
  "success_threshold": 1000,
  "reward_mean": -657.9900406956673,
  "reward_median": -272.07797718048096,
  "reward_std": 917.0808832946146,
  "score_mean": 10.45,
  "score_max": 40,
  "steps_mean": 449.7,
  "success_rate": 0.0,
  "death_rate": 0.95,
  "truncation_rate": 0.05,
  "idle_fraction": 0.336446519902157,
  "gas_fraction": 0.33088725817211473,
  "reverse_fraction": 0.33266622192572826
}
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> Get-ChildItem results -Filter "*v2*" | Select-Object Name, Length

Name                 Length
----                 ------
check_best_v2
evaluation_best_v2
evaluation_final_v2
evaluation_random_v2


(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> Get-ChildItem results/evaluation_best_v2, results/evaluation_final_v2, results/evaluation_random_v2 -File | Select-Object DirectoryName, Name

DirectoryName                                                                                     Name
-------------                                                                                     ----
C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn\results\evaluation_best_v2   evaluation.csv
C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn\results\evaluation_best_v2   metrics.json
C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn\results\evaluation_final_v2  evaluation.csv
C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn\results\evaluation_final_v2  metrics.json
C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn\results\evaluation_random_v2 evaluation.csv
C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn\results\evaluation_random_v2 metrics.json


(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python src/plot_evaluation.py
Figura guardada: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn\assets\evaluation_comparison.png
Figura guardada: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn\assets\evaluation_actions.png
Episodios por política: 20
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python -c "import inspect; from env_utils import make_env; env = make_env(); print(inspect.getsource(type(env.env.unwrapped)._get_reward)); env.close()"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'env_utils'
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python -c "import sys, inspect; sys.path.insert(0, 'src'); from env_utils import make_env; env = make_env(); print(inspect.getsource(type(env.env.unwrapped)._get_reward)); env.close()"
    def _get_reward(self, action):
        reverse_reward = 0
        idle_reward = 0
        if self.reward_type == "aggressive":
            idle_reward = -0.5
            reverse_reward = -1
        elif self.reward_type == "soft":
            idle_reward = -0.1
            reverse_reward = -0.2
        match self.reward_function:
            case "distance":
                # Reward is equal to -1 + current_distance - max_distance vs soft -0.2
                if self.agent.car.pos_x < self.agent.car.prev_max_distance:
                    reward = reverse_reward + (
                            self.agent.car.pos_x - self.agent.car.prev_max_distance)
                # Reward -0.5 if agent is at or around same position as last step vs soft-0.1
                elif self.agent.car.pos_x - self.agent.car.prev_max_distance < 0.001:
                    reward = idle_reward
                # Reward is equal to 1 + current_position - max_distance
                elif self.agent.car.pos_x > self.agent.car.prev_max_distance:
                    reward = 1 + (self.agent.car.pos_x - self.agent.car.prev_max_distance)
            case "action":
                if action == 0:  # Idle
                    reward = idle_reward
                elif action == 1:  # Gas
                    reward = 1
                elif action == 2:  # Reverse
                    reward = reverse_reward
            case "wheel_speed":
                wheel_speeds = [wheel.joint.speed for wheel in self.agent.car.wheels]
                # When wheel speeds are at a nearly idle state
                if all(-1 <= speed <= 1 for speed in wheel_speeds):
                    reward = idle_reward
                # When we have wheel speeds that bring the car forward (negative wheel speed = forward)
                elif all(speed < 0 for speed in wheel_speeds):
                    reward = 1
                # When we have wheel speeds that bring the car backwards
                elif all(speed > 0 for speed in wheel_speeds):
                    reward = reverse_reward
                else:
                    reward = 0
            case "airtime_wheel_speed":
                wheel_speeds = [wheel.joint.speed for wheel in self.agent.car.wheels]
                # When wheel speeds are at a nearly idle state
                if all(-1 <= speed <= 1 for speed in wheel_speeds):
                    reward = idle_reward
                # When we have wheel speeds that bring the car forward (negative wheel speed = forward)
                elif all(speed < 0 for speed in wheel_speeds):
                    reward = 1
                # When we have wheel speeds that bring the car backwards
                elif all(speed > 0 for speed in wheel_speeds):
                    reward = reverse_reward
                else:
                    reward = 0
                # When airtime counter is bigger than 1
                if self.agent.airtime_counter > 0:
                    reward = reward + self.agent.airtime_counter
                else:
                    reward = reward - 0.5
            case "airtime_distance":
                # Reward is equal to -1 + current_distance - max_distance vs soft -0.2
                if self.agent.car.pos_x < self.agent.car.prev_max_distance:
                    reward = reverse_reward + (
                            self.agent.car.pos_x - self.agent.car.prev_max_distance)
                # Reward -0.5 if agent is at or around same position as last step vs soft-0.1
                elif self.agent.car.pos_x - self.agent.car.prev_max_distance < 0.001:
                    reward = idle_reward
                # Reward is equal to 1 + current_position - max_distance
                elif self.agent.car.pos_x > self.agent.car.prev_max_distance:
                    reward = 1 + (self.agent.car.pos_x - self.agent.car.prev_max_distance)

                # When airtime counter is bigger than 1
                if self.agent.airtime_counter > 0:
                    reward = reward + self.agent.airtime_counter
                else:
                    reward = reward - 0.5

        return reward

(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python -c "import sys, inspect; sys.path.insert(0, 'src'); from env_utils import make_env; env = make_env(); env.reset(seed=1000); print(inspect.getsource(type(env.env.unwrapped.agent.car).update_status)); env.close()"
    def update_status(self):
        self.prev_pos_y = self.pos_y
        self.pos_x = self.chassis_body.position.x
        self.pos_y = self.chassis_body.position.y

        # Check whether we are moving forward with the car
        if self.pos_x > self.max_distance:
            self.prev_max_distance = self.max_distance
            self.max_distance = self.pos_x

        # When agent falls out of screen, we set status to dead
        if not self.dead and (self.pos_y * hill_racing.SCALE > hill_racing.SCREEN_HEIGHT + 10):
            self.dead = True
            self.agent.dead = True

(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git status --short
 M src/evaluate.py
?? assets/evaluation_actions.png
?? assets/evaluation_comparison.png
?? assets/training_curve.png
?? docs/resultados_actividad5.md
?? results/check_best_v2/
?? results/evaluation_best/
?? results/evaluation_best_v2/
?? results/evaluation_final/
?? results/evaluation_final_v2/
?? results/evaluation_random_v2/
?? src/plot_evaluation.py
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git diff --stat
 src/evaluate.py | 178 +++++++++++++++++++++++++++++++++++++++++++++++++-------
 1 file changed, 157 insertions(+), 21 deletions(-)
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python -m py_compile src/evaluate.py src/plot_evaluation.py

(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> @'
>> from pathlib import Path
>> from zipfile import ZipFile, ZIP_DEFLATED
>>
>> root = Path.cwd()
>> files = [
>>     "src/evaluate.py",
>>     "src/plot_evaluation.py",
>>     "docs/resultados_actividad5.md",
>>     "assets/training_curve.png",
>>     "assets/evaluation_comparison.png",
>>     "assets/evaluation_actions.png",
>> ]
>>
>> for folder in [
>>     "evaluation_best_v2",
>>     "evaluation_final_v2",
>>     "evaluation_random_v2",
>> ]:
>>     files.extend([
>>         f"results/{folder}/evaluation.csv",
>>         f"results/{folder}/metrics.json",
>>     ])
>>
>> missing = [name for name in files if not (root / name).is_file()]
>> if missing:
>>     raise FileNotFoundError(f"Faltan archivos: {missing}")
>>
>> output = root.parent / "Actividad5_Resultados_Evaluacion.zip"
>> with ZipFile(output, "w", ZIP_DEFLATED) as archive:
>>     for name in files:
>>         archive.write(root / name, arcname=name)
>>
>> with ZipFile(output) as archive:
>>     if archive.testzip() is not None:
>>         raise RuntimeError("El ZIP contiene un archivo dañado.")
>>     print(f"ZIP creado y verificado: {output}")
>>     print(f"Archivos incluidos: {len(archive.namelist())}")
>>     for name in archive.namelist():
>>         print(f"  {name}")
>> '@ | python -
ZIP creado y verificado: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\Actividad5_Resultados_Evaluacion.zip
Archivos incluidos: 12
  src/evaluate.py
  src/plot_evaluation.py
  docs/resultados_actividad5.md
  assets/training_curve.png
  assets/evaluation_comparison.png
  assets/evaluation_actions.png
  results/evaluation_best_v2/evaluation.csv
  results/evaluation_best_v2/metrics.json
  results/evaluation_final_v2/evaluation.csv
  results/evaluation_final_v2/metrics.json
  results/evaluation_random_v2/evaluation.csv
  results/evaluation_random_v2/metrics.json
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> dir


    Directorio: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      11/10/2026  6:38 p. m.                .venv
d-----      11/10/2026  7:16 p. m.                assets
d-----      11/10/2026  5:35 p. m.                configs
d-----      11/10/2026  7:16 p. m.                docs
d-----      11/10/2026  7:01 p. m.                results
d-----      11/10/2026  7:05 p. m.                src
d-----      11/10/2026  5:35 p. m.                tests
-a----      11/10/2026  5:35 p. m.             44 .gitignore
-a----      11/10/2026  5:35 p. m.           4468 README.md
-a----      11/10/2026  5:35 p. m.            109 requirements.txt


(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> cd..
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> dirt
dirt : El término 'dirt' no se reconoce como nombre de un cmdlet, función, archivo de script o programa ejecutable. Compruebe si escribió correctamente el
nombre o, si incluyó una ruta de acceso, compruebe que dicha ruta es correcta e inténtelo de nuevo.
En línea: 1 Carácter: 1
+ dirt
+ ~~~~
    + CategoryInfo          : ObjectNotFound: (dirt:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> dir


    Directorio: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      11/10/2026  5:38 p. m.                hill_climb_racing_dqn
-a----      11/10/2026  7:26 p. m.         313961 Actividad5_Resultados_Evaluacion.zip


(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> python -m zipfile -l ..\Actividad5_Resultados_Evaluacion.zip
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\Mauricio\AppData\Local\Programs\Python\Python312\Lib\zipfile\__main__.py", line 4, in <module>
    main()
  File "C:\Users\Mauricio\AppData\Local\Programs\Python\Python312\Lib\zipfile\__init__.py", line 2284, in main
    with ZipFile(src, 'r', metadata_encoding=encoding) as zf:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Mauricio\AppData\Local\Programs\Python\Python312\Lib\zipfile\__init__.py", line 1336, in __init__
    self.fp = io.open(file, filemode)
              ^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '..\\Actividad5_Resultados_Evaluacion.zip'
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> dir


    Directorio: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      11/10/2026  5:38 p. m.                hill_climb_racing_dqn
-a----      11/10/2026  7:26 p. m.         313961 Actividad5_Resultados_Evaluacion.zip


(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> python -m zipfile -l ..\Actividad5_Resultados_Evaluacion.zip
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\Mauricio\AppData\Local\Programs\Python\Python312\Lib\zipfile\__main__.py", line 4, in <module>
    main()
  File "C:\Users\Mauricio\AppData\Local\Programs\Python\Python312\Lib\zipfile\__init__.py", line 2284, in main
    with ZipFile(src, 'r', metadata_encoding=encoding) as zf:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Mauricio\AppData\Local\Programs\Python\Python312\Lib\zipfile\__init__.py", line 1336, in __init__
    self.fp = io.open(file, filemode)
              ^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '..\\Actividad5_Resultados_Evaluacion.zip'
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> dir


    Directorio: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      11/10/2026  5:38 p. m.                hill_climb_racing_dqn
-a----      11/10/2026  7:26 p. m.         313961 Actividad5_Resultados_Evaluacion.zip


(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4> cd hill_climb_racing_dqn
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> dir


    Directorio: C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      11/10/2026  6:38 p. m.                .venv
d-----      11/10/2026  7:16 p. m.                assets
d-----      11/10/2026  5:35 p. m.                configs
d-----      11/10/2026  7:16 p. m.                docs
d-----      11/10/2026  7:01 p. m.                results
d-----      11/10/2026  7:05 p. m.                src
d-----      11/10/2026  5:35 p. m.                tests
-a----      11/10/2026  5:35 p. m.             44 .gitignore
-a----      11/10/2026  5:35 p. m.           4468 README.md
-a----      11/10/2026  5:35 p. m.            109 requirements.txt


(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> python -m zipfile -l ..\Actividad5_Resultados_Evaluacion.zip
File Name                                             Modified             Size
src/evaluate.py                                2026-10-11 18:56:56         5596
src/plot_evaluation.py                         2026-10-11 19:06:26         4739
docs/resultados_actividad5.md                  2026-10-11 19:19:28        12444
assets/training_curve.png                      2026-10-11 18:40:42       111577
assets/evaluation_comparison.png               2026-10-11 19:06:40       156971
assets/evaluation_actions.png                  2026-10-11 19:06:42        65402
results/evaluation_best_v2/evaluation.csv      2026-10-11 18:59:48         1418
results/evaluation_best_v2/metrics.json        2026-10-11 18:59:48          539
results/evaluation_final_v2/evaluation.csv     2026-10-11 19:01:08         1424
results/evaluation_final_v2/metrics.json       2026-10-11 19:01:08          525
results/evaluation_random_v2/evaluation.csv    2026-10-11 19:02:08         1599
results/evaluation_random_v2/metrics.json      2026-10-11 19:02:08          523
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> Get-Location

Path
----
C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn


(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git remote -v
origin  https://github.com/jddma/RL2.git (fetch)
origin  https://github.com/jddma/RL2.git (push)
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git branch --show-current
actividad5-evaluacion
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git status --short
 M src/evaluate.py
?? assets/evaluation_actions.png
?? assets/evaluation_comparison.png
?? assets/training_curve.png
?? docs/resultados_actividad5.md
?? results/check_best_v2/
?? results/evaluation_best/
?? results/evaluation_best_v2/
?? results/evaluation_final/
?? results/evaluation_final_v2/
?? results/evaluation_random_v2/
?? src/plot_evaluation.py
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git status --short
 M src/evaluate.py
?? assets/evaluation_actions.png
?? assets/evaluation_comparison.png
?? assets/training_curve.png
?? docs/resultados_actividad5.md
?? results/check_best_v2/
?? results/evaluation_best/
?? results/evaluation_best_v2/
?? results/evaluation_final/
?? results/evaluation_final_v2/
?? results/evaluation_random_v2/
?? src/plot_evaluation.py
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git fetch origin
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git branch -vv
* actividad5-evaluacion 2695aae Rename arquitectura.md to README.md
  master                2695aae [origin/master] Rename arquitectura.md to README.md
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git add -- src/evaluate.py src/plot_evaluation.py docs/resultados_actividad5.md assets/training_curve.png assets/evaluation_comparison.png assets/evaluation_actions.png results/evaluation_best_v2/evaluation.csv results/evaluation_best_v2/metrics.json results/evaluation_final_v2/evaluation.csv results/evaluation_final_v2/metrics.json results/evaluation_random_v2/evaluation.csv results/evaluation_random_v2/metrics.json
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git diff --cached --stat
 assets/evaluation_actions.png               | Bin 0 -> 65402 bytes
 assets/evaluation_comparison.png            | Bin 0 -> 156971 bytes
 assets/training_curve.png                   | Bin 0 -> 111577 bytes
 docs/resultados_actividad5.md               | 273 ++++++++++++++++++++++++++++
 results/evaluation_best_v2/evaluation.csv   |  21 +++
 results/evaluation_best_v2/metrics.json     |  20 ++
 results/evaluation_final_v2/evaluation.csv  |  21 +++
 results/evaluation_final_v2/metrics.json    |  20 ++
 results/evaluation_random_v2/evaluation.csv |  21 +++
 results/evaluation_random_v2/metrics.json   |  20 ++
 src/evaluate.py                             | 178 +++++++++++++++---
 src/plot_evaluation.py                      | 139 ++++++++++++++
 12 files changed, 692 insertions(+), 21 deletions(-)
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git diff --cached --check
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git commit -m "Actividad 5: evaluacion greedy, baseline y analisis de resultados"
[actividad5-evaluacion 7976d3d] Actividad 5: evaluacion greedy, baseline y analisis de resultados
 12 files changed, 692 insertions(+), 21 deletions(-)
 create mode 100644 assets/evaluation_actions.png
 create mode 100644 assets/evaluation_comparison.png
 create mode 100644 assets/training_curve.png
 create mode 100644 docs/resultados_actividad5.md
 create mode 100644 results/evaluation_best_v2/evaluation.csv
 create mode 100644 results/evaluation_best_v2/metrics.json
 create mode 100644 results/evaluation_final_v2/evaluation.csv
 create mode 100644 results/evaluation_final_v2/metrics.json
 create mode 100644 results/evaluation_random_v2/evaluation.csv
 create mode 100644 results/evaluation_random_v2/metrics.json
 create mode 100644 src/plot_evaluation.py
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn> git push -u origin actividad5-evaluacion
remote: Permission to jddma/RL2.git denied to mauricioplascencia.
fatal: unable to access 'https://github.com/jddma/RL2.git/': The requested URL returned error: 403
(hill_climb_racing_dqn) PS C:\IA\Simulacion_Aprendizaje_Refuerzo\Unidad_4\hill_climb_racing_dqn>
````
