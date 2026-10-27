# Clasificación de células de papanicolaou: una evaluación honesta

Auditoría metodológica sobre el dataset **SIPaKMeD** (4049 células cervicales recortadas de 966
láminas de Papanicolaou, cinco clases).

La literatura reporta rutinariamente más del 99% de acierto en este dataset. Este trabajo no busca
superar esa cifra, sino preguntarse si significa lo que parece, y explicar qué está usando realmente
el modelo para decidir.

---

## Resultados principales

| Hallazgo | Evidencia |
|---|---|
| Repartir las fotos al azar deja al **91.8%** de las células de prueba con una hermana del mismo cúmulo en el entrenamiento | medición directa sobre los pliegues |
| Ese reparto esconde **~1 de cada 4 errores** y reduce a menos de la mitad la variabilidad aparente | 0.9796 vs 0.9731 macro F1, McNemar p = 0.012 |
| Añadir 26 descriptores morfológicos sacados de contornos expertos **empeora** el modelo | 0.9671 vs 0.9731, McNemar p = 0.0075 |
| La red codifica el **color y la textura** del núcleo, pero **no su geometría** | sondeo lineal: R² 0.64 a 0.83 vs R² 0.045 a 0.19 |
| Ajustar la red a la tarea **empobrece** su representación morfológica frente a la red genérica | la red de ImageNet gana en las 12 medidas sondeadas |
| La red **sí mira el núcleo**: 2.4 veces más atención de la que le tocaría por área | Grad-CAM normalizado por área, n = 4049 |
| Con la tinción alterada, el acierto cae 20 puntos pero la atención al núcleo **sube** | 2.39 → 2.71 |

La frase que resume el trabajo: **la red mira el núcleo, pero lo lee por su color, no por su forma.**
Eso explica por qué los descriptores de forma no aportan, y por qué un cambio de tinción la rompe
mientras sigue mirando el sitio correcto.

---

## Estructura

```
notebooks/
  cervical_cancer_completo_local.ipynb    pipeline completo, ya ejecutado (con salidas)
  cervical_cancer_completo_colab.ipynb    misma versión adaptada a Google Colab
  por_fases/                              cada fase por separado, para leer o reejecutar suelta
docs/
  explicacion_fases_3_y_4.md              explicación en tres niveles, con el código comentado
  fases_3_y_4.html                        la misma explicación en formato web navegable
  instalacion_local.md                    instalación paso a paso en Linux
  plan-proyecto-cancer-cervical.pdf       plan original del proyecto
  exposicion/                             guiones de presentación
resultados/                               salida de los notebooks (ignorada por git)
```

El notebook completo es el punto de entrada. Las fases sueltas existen por si solo interesa una
parte; el contenido es el mismo.

---

## Las cinco partes

**1. Qué hay dentro del dataset.** Las 4049 células salen de 966 láminas, con 4.2 por lámina de
promedio y un caso extremo de 26. El prefijo del nombre de archivo identifica la lámina de origen, y
eso es lo que sostiene todo lo demás. Anotación completa, cero contornos faltantes, un solo caso de
anotación imposible.

**2. Cuánto engaña repartir al azar.** El mismo modelo entrenado dos veces, cambiando únicamente
cómo se reparten las fotos. La diferencia en macro F1 parece pequeña, pero contada en errores
desaparece casi un cuarto de ellos, y la variabilidad entre pliegues se reduce a menos de la mitad:
el resultado publicado no solo es alto, además parece más preciso de lo que es.

**3. ¿Ayudan las medidas del especialista?** 26 descriptores morfológicos calculados desde los
contornos anotados, comparados en una ablación donde solo cambia una variable (misma red, misma
cabeza, mismos pliegues). No ayudan: perjudican, y de forma estadísticamente detectable. Un modelo
que usa solo esas medidas, sin ver la foto, llega a 0.85, así que el problema no es su calidad.

**4. Entonces, ¿por qué no ayudan?** Sondeo lineal sobre los embeddings para medir qué información
contiene la red, con la red congelada y un control sin ajustar. Resulta que lo que le falta a la red
(la geometría del núcleo) no es lo que necesita para acertar. De paso se cayó la hipótesis de que el
reescalado destruía el tamaño original: la red lo deduce igual, con R² 0.79.

**5. Dónde mira y qué tanto aguanta.** Métrica de atención Grad-CAM normalizada por área nuclear,
medida en las 4049 células en vez de ilustrada con cuatro ejemplos, más perturbaciones de tinción,
desenfoque y compresión en tiempo de prueba.

---

## Decisiones metodológicas

Tres cosas que sostienen los resultados y conviene no romper al reutilizar el código:

- **Agrupación por lámina en todos los repartos.** Es el eje del trabajo. Cualquier comparación que
  no la respete produce números inflados.
- **Toda estadística de escalado sale solo del grupo de entrenamiento**, incluida la mediana con que
  se rellenan los pocos valores faltantes. Calcularla sobre el dataset completo es una fuga sutil,
  del mismo tipo que la del reparto.
- **Las ablaciones cambian una sola variable.** La red con y sin medidas es la misma clase, la misma
  cabeza y el mismo bucle de entrenamiento; solo cambia la dimensión de entrada.

Y en la fase 4, los embeddings de cada célula provienen del modelo que no la vio entrenando. Usar
un modelo que ya la conoce mediría memorización en vez de representación.

---

## Cómo ejecutarlo

**En Colab:** abre `notebooks/cervical_cancer_completo_colab.ipynb`, activa GPU y córrelo de arriba
a abajo. Necesita un token de Kaggle guardado en los Secrets del notebook.

**En local:** sigue `docs/instalacion_local.md`. Resumen rápido:

```bash
python3 -m venv ~/venvs/sipakmed
source ~/venvs/sipakmed/bin/activate
pip install -r requirements.txt
jupyter lab
```

Ojo con PyTorch: instalarlo desde PyPI sin más puede traer la versión de solo CPU. El archivo de
requisitos lo deja anotado.

El dataset se descarga solo con `kagglehub` (unos 6.4 GB comprimidos, 9.2 GB descomprimidos). Hace
falta un token de Kaggle en `~/.kaggle/access_token`.

Los bloques pesados guardan sus resultados en `resultados/` y los reutilizan si ya existen. Para
rehacer todo desde cero, pon `USAR_CACHE = False` en la celda de configuración. El pipeline completo
son unos 35 entrenamientos, entre 2 y 3 horas en una RTX 4050.

---

## Limitaciones

- **No se conoce de qué paciente viene cada lámina.** Las 966 láminas podrían venir de bastantes
  menos pacientes, así que puede quedar contaminación residual imposible de eliminar con los datos
  publicados. La lámina es la agrupación más fina disponible.
- **Una sola semilla.** La dirección de los efectos es estable entre máquinas, pero la magnitud de la
  fuga varió (entre un cuarto y un 40% de los errores) según la corrida. Promediar varias semillas
  daría una estimación más firme.
- **Una sola arquitectura.** El efecto del reparto no debería depender del modelo, pero no está
  comprobado aquí.
- **El sondeo lineal tiene un techo conocido.** Un R² bajo prueba que la información no es
  linealmente recuperable desde el vector posterior al promediado global, no que la red no la tenga.
  La geometría es espacial y no sobrevive a un promedio; repetir el sondeo antes del promediado
  cerraría esa duda.
- **Las perturbaciones son simuladas.** Aproximan las condiciones de otro laboratorio, no las
  reemplazan.

---

## Stack

PyTorch y torchvision (EfficientNet-B0 preentrenada en ImageNet), scikit-learn, scikit-image,
OpenCV, statsmodels, matplotlib.

Métricas: macro F1, coeficiente de Matthews, exactitud, matrices de confusión, McNemar para
comparaciones pareadas y Mann-Whitney para los contrastes de atención.

---

## Dataset

Plissiti, M. E., Dimitrakopoulos, P., Sfikas, G., Nikou, C., Krikoni, O., Charchanti, A. (2018).
*SIPAKMED: A new dataset for feature and image based classification of normal and pathological
cervical cells in Pap smear images.* IEEE International Conference on Image Processing (ICIP),
Atenas, Grecia.

Copia usada: [Cervical Cancer largest dataset (SipakMed)](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed)
en Kaggle. Original: https://www.cs.uoi.gr/~marina/sipakmed.html

El dataset no se incluye en este repositorio.

---

## Licencia

Código y documentación bajo licencia MIT (ver `LICENSE`). El dataset mantiene sus propios términos.

## Autor

Manuel M. Muñoz-Cárdenas
