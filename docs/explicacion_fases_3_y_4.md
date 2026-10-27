# Fases 3 y 4 explicadas, con el código al lado

Tres niveles para cada fase. El nivel 1 es lo que dirías en una presentación, el nivel 3 es lo que
necesitas si te preguntan en la defensa.

---

# FASE 3: ¿ayudan las medidas del especialista?

## Nivel 1 — la idea

Un citotecnólogo al microscopio no mira la célula "en general": mira cosas concretas. Qué tan grande
es el núcleo, cuánto espacio ocupa dentro de la célula, si está oscuro, si es redondo. El dataset
trae esos contornos ya dibujados a mano, así que podemos calcular esas medidas y **dárselas a la red
además de la foto**.

La idea era: si la red ve la foto *y* además recibe los números que usa el experto, debería acertar
más.

Resultado: **no acertó más, acertó menos**. 0.9731 sin las medidas, 0.9671 con ellas.

## Nivel 2 — cómo se comprobó

Lo delicado aquí no es el experimento, es la comparación. El error típico en los trabajos publicados
es entrenar una red grande con medidas y compararla contra una red distinta sin medidas. Ahí cambian
dos cosas a la vez y no sabes cuál causó la diferencia.

Nosotros usamos **la misma red, la misma cabeza, las mismas épocas, la misma semilla y los mismos
pliegues**. Lo único que cambia es si el vector de 26 medidas se le pega o no. Una sola variable.

Se compararon tres modelos:

| Modelo | Macro F1 |
|---|---|
| Solo medidas, sin ver la foto | 0.8505 |
| Solo la foto | 0.9731 |
| Foto + medidas | 0.9671 |

Y la comparación se cerró con una prueba pareada célula por célula: 49 células que el modelo acierta
sin medidas y falla con ellas, contra 25 en sentido contrario, p = 0.0075. **No es ruido: empeora de
verdad.**

Un detalle que importa: el modelo de solo medidas llega a 0.85 sin mirar una foto. O sea que **las
medidas no son basura**, tienen señal real. El problema no es la calidad de la información.

### El código, paso a paso

**1. Sacar las medidas de forma de un contorno.**

Un contorno es una lista de coordenadas. De ahí salen los descriptores geométricos:

```python
def medidas_contorno(cnt):
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    area_hull = cv2.contourArea(cv2.convexHull(cnt))

    circ = 4 * np.pi * area / (per ** 2) if per > 0 else np.nan     # 1 = círculo perfecto
    sol = area / area_hull if area_hull > 0 else np.nan             # qué tan "lleno" es
    diam = np.sqrt(4 * area / np.pi)

    if len(cnt) >= 5:                                               # la elipse necesita 5 puntos
        (_, _), (e1, e2), _ = cv2.fitEllipse(cnt)
        may, men = max(e1, e2), min(e1, e2)
        alarg = men / may if may > 0 else np.nan
        exc = np.sqrt(1 - (men / may) ** 2) if may > 0 else np.nan
    else:
        alarg = exc = np.nan

    M = cv2.moments(cnt)
    cx = M["m10"] / M["m00"] if M["m00"] > 0 else np.nan
    cy = M["m01"] / M["m00"] if M["m00"] > 0 else np.nan
    return area, per, circ, sol, diam, alarg, exc, cx, cy
```

Qué hace cada cosa:

- **`circ`** compara el área con el perímetro. Un círculo da 1; cuanto más irregular o dentado el
  borde, más baja. Es la "redondez".
- **`sol`** divide el área real entre la de su envoltura convexa (la goma elástica que rodearía el
  contorno). Si vale 1 no hay entrantes; si baja, el borde tiene muescas.
- **`fitEllipse`** ajusta una elipse y devuelve sus dos ejes. El cociente entre el menor y el mayor
  es el alargamiento: 1 es redondo, cerca de 0 es un cigarro. **Necesita al menos 5 puntos**, y de
  ahí salen los 135 valores no calculables de todo el dataset.
- **`moments`** da el centro de masa del contorno, que después se usa para saber si el núcleo está
  centrado o corrido.

**2. Juntar todo por célula.**

La función completa combina lo geométrico con lo de color y textura:

```python
def caracteristicas(fila):
    gris = cv2.cvtColor(cv2.imread(fila["img"]), cv2.COLOR_BGR2GRAY)
    alto, ancho = gris.shape
    cyt = contorno(fila["cyt"]).reshape(-1, 1, 2)
    nuc = contorno(fila["nuc"]).reshape(-1, 1, 2)

    a_c, p_c, ci_c, so_c, d_c, al_c, ex_c, cx_c, cy_c = medidas_contorno(cyt)
    a_n, p_n, ci_n, so_n, d_n, al_n, ex_n, cx_n, cy_n = medidas_contorno(nuc)

    m_cyt = np.zeros(gris.shape, np.uint8); cv2.fillPoly(m_cyt, [cyt.reshape(-1, 2)], 1)
    m_nuc = np.zeros(gris.shape, np.uint8); cv2.fillPoly(m_nuc, [nuc.reshape(-1, 2)], 1)
    m_cit = np.clip(m_cyt.astype(int) - m_nuc.astype(int), 0, 1).astype(np.uint8)
    ...
```

La parte importante son las tres máscaras. `fillPoly` rellena el polígono, así que `m_cyt` marca
toda la célula y `m_nuc` solo el núcleo. La resta da `m_cit`, **el citoplasma sin el núcleo**, que es
lo que hace falta para comparar el brillo de una zona contra la otra. Si midieras la intensidad
sobre `m_cyt` entero, estarías promediando el núcleo consigo mismo y el contraste saldría diluido.

El `np.clip(..., 0, 1)` protege contra el caso raro de un núcleo anotado fuera de su célula: sin él,
la resta daría -1 en esos píxeles.

**3. El modelo que solo usa medidas.**

```python
pred_medidas = np.full(len(df), -1)
for k in range(5):
    tr = np.where(df["fold_lamina"].values != k)[0]
    te = np.where(df["fold_lamina"].values == k)[0]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                       random_state=SEMILLA)
    m.fit(F.iloc[tr], Y[tr])
    pred_medidas[te] = m.predict(F.iloc[te])
```

Árboles de decisión potenciados, no una red. Dos motivos: es rapidísimo, y **maneja los valores
faltantes por dentro**, así que aquí no hace falta rellenar los 135 huecos. Fíjate en que usa
`fold_lamina`, los mismos pliegues que todo lo demás: eso es lo que permite comparar los tres
modelos entre sí.

**4. El escalado sin fuga. Este es el punto que hay que saber defender.**

```python
def escalar(tr, te):
    """Escala con estadísticas del ENTRENAMIENTO únicamente. Calcularlas sobre
    todo el dataset sería pasarle información del examen al modelo."""
    mu, sd = F.iloc[tr].mean(), F.iloc[tr].std().replace(0, 1)
    med = F.iloc[tr].median()
    return (np.nan_to_num(((F.iloc[tr].fillna(med) - mu) / sd).values),
            np.nan_to_num(((F.iloc[te].fillna(med) - mu) / sd).values))
```

Las tres estadísticas (`mu`, `sd`, `med`) se calculan **solo con `tr`** y después se aplican tanto a
`tr` como a `te`. Escribir `F.mean()` en vez de `F.iloc[tr].mean()` sería una fuga: el promedio del
conjunto completo contiene información de las células de examen.

Es sutil porque no rompe nada ni da error, solo infla el resultado. Y es exactamente el mismo tipo de
problema que atacamos en la fase 2 con el reparto, así que sería incoherente evitarlo allá y
cometerlo aquí.

El `.replace(0, 1)` evita dividir entre cero si alguna medida fuera constante.

**5. La misma red con y sin medidas.**

```python
def crear_red(n_extra=0):
    base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    base.classifier = nn.Identity()                 # nos quedamos con los 1280 números
    cabeza = nn.Sequential(nn.Linear(1280 + n_extra, 256), nn.ReLU(), nn.Dropout(0.3),
                           nn.Linear(256, len(CLASES)))
    return base.to(DEV), cabeza.to(DEV)
```

`nn.Identity()` reemplaza el clasificador original de 1000 clases de ImageNet por "no hagas nada", de
modo que `base(x)` devuelve el vector de 1280 en crudo. Ese corte es lo que permite después pegarle
las medidas y también lo que hace posible toda la fase 4.

`n_extra` es el único interruptor entre las dos condiciones: con 0 la cabeza recibe 1280, con 26
recibe 1306. Todo lo demás del código es literalmente el mismo.

Y en el entrenamiento, la concatenación:

```python
z = base(x)
if n_extra:
    z = torch.cat([z, ex], dim=1)
loss = criterio(cabeza(z), y)
```

**6. La comparación pareada.**

```python
def comparar(p1, p2, n1="modelo 1", n2="modelo 2"):
    """Compara dos modelos célula por célula. Solo cuentan los casos donde discrepan."""
    o1, o2 = (p1 == Y), (p2 == Y)
    t = [[int((o1 & o2).sum()), int((o1 & ~o2).sum())],
         [int((~o1 & o2).sum()), int((~o1 & ~o2).sum())]]
    r = mcnemar(t, exact=False, correction=True)
```

McNemar no compara los dos porcentajes globales, compara **los desacuerdos**. Las casillas que
importan son las de fuera de la diagonal: cuántas acierta uno y falla el otro, y viceversa. Si están
repartidas parejo, la diferencia es azar.

Esto se puede hacer aquí porque cada célula tiene predicción en las dos condiciones (los cinco
pliegues cubren todo el dataset), así que la comparación es pareada y mucho más sensible que mirar
dos medias.

## Nivel 3 — los detalles

**Las 26 medidas.** De cada contorno se sacan área, perímetro, diámetro equivalente, circularidad
(4πA/P², vale 1 en un círculo perfecto), solidez (área dividida por la de su envoltura convexa, que
detecta entrantes), alargamiento y excentricidad ajustando una elipse. Eso por duplicado, núcleo y
citoplasma, más los cocientes entre ambos. Después el desplazamiento del núcleo respecto al centro de
la célula, normalizado por la raíz del área para que no dependa del tamaño. Luego intensidad media
del núcleo, del citoplasma sin el núcleo, el contraste entre los dos y la desviación dentro del
núcleo. Textura con una matriz de coocurrencia a 32 niveles de gris: contraste, homogeneidad y
energía. Y el tamaño original del recorte.

La parte de textura merece una nota, porque tiene una decisión escondida:

```python
x, y, w_, h_ = cv2.boundingRect(nuc)
rec = gris[y:y+h_, x:x+w_]
if rec.size > 16:
    g = graycomatrix((rec // 8).astype(np.uint8), [1], [0], levels=32,
                     symmetric=True, normed=True)
```

La matriz de coocurrencia cuenta con qué frecuencia dos niveles de gris aparecen juntos a cierta
distancia. Con 256 niveles la matriz sería de 256×256 y con un núcleo de unos pocos cientos de
píxeles quedaría casi vacía, o sea puro ruido. El `// 8` la reduce a 32 niveles para que las cuentas
sean estables. `[1]` y `[0]` significan distancia de un píxel en horizontal.

**La arquitectura de fusión.** EfficientNet-B0 produce un vector de 1280 números; se concatenan las
26 medidas y entra un perceptrón de 256 unidades con dropout 0.3 hasta las 5 clases. Cuando no hay
medidas, la entrada es de 1280 y todo lo demás es idéntico.

**Por qué empeora, y no solo "no ayuda".** Tres explicaciones compatibles: las 26 entradas extra son
26×256 parámetros adicionales en la primera capa que la red tiene que aprender a ignorar, y con 4049
ejemplos eso cuesta capacidad; son redundantes respecto a lo que la red ya extrae, así que no compran
nada a cambio; y añaden ruido de optimización. La caída se concentra justo donde esperábamos ganancia
(koilocitótica pierde 1.5 puntos, metaplásica 1.2), lo cual encaja con que sean un estorbo y no una
ayuda mal aprovechada.

---

# FASE 4: entonces, ¿por qué no ayudan?

## Nivel 1 — la idea

Un resultado negativo se puede reportar como "no funcionó" y ya. Es mucho mejor explicar por qué.

La hipótesis era: **la red ya calcula esas medidas por su cuenta**, así que dárselas es repetirle algo
que ya sabe.

Para comprobarlo hicimos algo al revés de lo intuitivo. Cuando la red procesa una foto, antes de
decidir la clase produce 1280 números que la resumen. En vez de usarlos para adivinar la clase, los
usamos para **adivinar las medidas**. Si desde esos números se puede reconstruir la proporción
núcleo/célula, es que estaba ahí adentro.

Resultado: la hipótesis era **solo a medias**, y apareció algo que no esperábamos.

## Nivel 2 — qué salió

Se mide con R²: cerca de 1, la medida se reconstruye bien; cerca de 0, no está.

**Lo que la red sí tiene:** área del citoplasma 0.83, proporción núcleo/célula 0.66, oscuridad del
núcleo 0.64, contraste núcleo-citoplasma 0.64, textura 0.56. Todo eso es **color e intensidad**.

**Lo que no tiene:** área nuclear 0.16, redondez 0.19, solidez 0.08, alargamiento 0.045,
desplazamiento del núcleo -0.03 (o sea, nada). Todo eso es **forma**.

La conclusión es más interesante que la hipótesis original: **la red mira el núcleo pero lo mide por
su color, no por su geometría**. Y en la fase 3 le dimos justamente la geometría que le falta, y no
le sirvió. O sea que *lo que le falta no es lo que necesita*.

**La sorpresa:** metimos como control la misma red **sin entrenar en células**, tal como viene
preentrenada con fotos genéricas. Recupera las medidas mejor que la nuestra **en las doce, sin una
sola excepción**. El caso más marcado es el alargamiento del núcleo, que cae de 0.19 a 0.045 tras el
entrenamiento.

Eso significa que ajustar la red a esta tarea **empobrece** su descripción de la morfología. La red
aprende a quedarse solo con lo que le sirve para separar cinco clases y descarta el resto.

**Y un cabo suelto que se cerró:** creíamos que al escalar todo a 224×224 se perdía el tamaño real de
la célula. Pues no: `ancho_orig` se recupera con R² 0.79. Un recorte pequeño ampliado se ve borroso y
de grano grueso, así que la escala deja huella. Lo comprobamos entrenando con solo esos dos números
y, en efecto, no cambió nada (0.9702 vs 0.9731, p = 0.27).

### El código, paso a paso

**1. Guardar el resumen interno de cada foto.**

La función de entrenamiento ya venía preparada para devolverlo:

```python
with torch.no_grad():
    for lote in dl_te:
        ...
        z = base(x.to(DEV))                        # los 1280 números
        s = cabeza(torch.cat([z, ex], dim=1) if n_extra else z)
        preds.append(s.argmax(1).cpu().numpy())
        if devolver_emb:
            embs.append(z.float().cpu().numpy())   # esto es lo que analizamos
```

Es el mismo `z` que en la fase 3 se concatenaba con las medidas. La diferencia es que aquí, en vez de
pasarlo a la cabeza y olvidarlo, lo guardamos.

Y el bucle que lo llena:

```python
EMB = np.zeros((len(df), 1280), dtype=np.float32)
for k in range(5):
    tr = np.where(df["fold_lamina"].values != k)[0]
    te = np.where(df["fold_lamina"].values == k)[0]
    p, e = entrenar(tr, te, devolver_emb=True)
    pred_img[te] = p
    EMB[te] = e
```

**Aquí está el requisito que hace válido todo el experimento.** El resumen de las células del pliegue
`k` sale del modelo entrenado con los otros cuatro, o sea de un modelo que **nunca las vio**. Si
usaras un modelo que ya las conoce, estarías midiendo memorización, no representación, y el R²
saldría alto sin significar nada.

**2. El control: la red sin entrenar.**

```python
base_gen = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
base_gen.classifier = nn.Identity()
base_gen = base_gen.to(DEV).eval()

EMB_GEN = np.zeros((len(df), 1280), dtype=np.float32)
dl = DataLoader(Celulas(np.arange(len(df)), tf_eval), batch_size=128, num_workers=2)
pos = 0
with torch.no_grad():
    for x, y, _ in dl:
        z = base_gen(x.to(DEV))
        EMB_GEN[pos:pos+len(z)] = z.float().cpu().numpy()
        pos += len(z)
```

Exactamente la misma arquitectura y los mismos pesos de partida, pero **sin la fase de
entrenamiento**. Una sola pasada sobre todas las células, sin pliegues, porque no hay nada que pueda
memorizar: nunca vio ninguna.

Esto es lo que convierte el experimento en una comparación en vez de un número suelto. Sin este
control, un R² de 0.66 no te dice nada: ¿es mucho o poco? Con el control sabes que **es menos de lo
que había antes de entrenar**.

**3. El sondeo.**

```python
def sondear(EMBED, medida):
    """Intenta recuperar una medida desde el resumen de la red, con los mismos pliegues."""
    obj = F[medida].values.astype(float)
    validos = ~np.isnan(obj)
    pred = np.full(len(df), np.nan)
    for k in range(5):
        tr = np.where((df["fold_lamina"].values != k) & validos)[0]
        te = np.where((df["fold_lamina"].values == k) & validos)[0]
        mu, sd = EMBED[tr].mean(0), EMBED[tr].std(0) + 1e-6      # escalado solo con train
        mod = RidgeCV(alphas=ALFAS).fit((EMBED[tr] - mu) / sd, obj[tr])
        pred[te] = mod.predict((EMBED[te] - mu) / sd)
    ok = ~np.isnan(pred)
    return r2_score(obj[ok], pred[ok]), pred
```

Cuatro decisiones dentro de estas pocas líneas:

- **`RidgeCV` y no una red.** El sondeo tiene que ser **deliberadamente simple**. Si usaras un modelo
  potente para reconstruir la medida, podría estar calculándola él mismo a partir de información
  parcial, y ya no sabrías si estaba en la red o la inventó el sondeo. Una regresión lineal
  regularizada solo puede leer lo que ya está linealmente disponible.
- **`alphas=ALFAS`** hace que elija sola su nivel de regularización. Con 1280 variables y unos 3200
  ejemplos por pliegue, sin regularizar se sobreajustaría de inmediato.
- **`& validos`** excluye las células donde la medida no se pudo calcular (las de menos de 5 puntos
  en el contorno). Se excluyen del ajuste y del R², no se rellenan: rellenar aquí introduciría un
  valor inventado en la variable que estamos tratando de predecir.
- **El escalado, otra vez solo con `tr`.** Misma disciplina que en la fase 3.

El R² se calcula sobre las predicciones **fuera de muestra concatenadas** de los cinco pliegues, o
sea sobre las 4049 células, cada una predicha por un modelo que no la usó.

**4. La comprobación del tamaño.**

```python
TAM = F[["ancho_orig", "alto_orig"]]
for k in range(5):
    tr = np.where(df["fold_lamina"].values != k)[0]
    te = np.where(df["fold_lamina"].values == k)[0]
    mu, sd = TAM.iloc[tr].mean(), TAM.iloc[tr].std().replace(0, 1)
    pred_tam[te] = entrenar(tr, te,
                            ((TAM.iloc[tr] - mu) / sd).values,
                            ((TAM.iloc[te] - mu) / sd).values)
```

Es la fase 3 otra vez pero con `n_extra=2` en vez de 26. Que se pueda hacer cambiando solo qué
columnas se pasan, sin tocar ni la red ni el bucle, es precisamente la ventaja de haber unificado el
código: la comparación es indiscutible porque es el mismo programa.

## Nivel 3 — los detalles y las trampas

**Qué es exactamente el sondeo.** Una regresión de cresta con el parámetro de regularización elegido
por validación interna, entrenada sobre los 1280 números para predecir cada medida, con los mismos
cinco pliegues por lámina y el R² calculado sobre las predicciones fuera de muestra.

**La limitación honesta, y esta es la que más peso tiene.** Un R² bajo no prueba que la información
"no esté en la red". Prueba que **no es linealmente recuperable desde el vector de 1280 posterior al
promediado global**.

Ahí está la clave, y se ve en la arquitectura. EfficientNet-B0 produce internamente un mapa de 7×7
con 1280 canales; el promediado global lo colapsa a un solo número por canal, y ese es el vector que
sondeamos. Las estadísticas de intensidad y textura **sobreviven a un promedio**, porque son
promedios en sí mismas. La geometría, que es intrínsecamente espacial, **no sobrevive**: al promediar
sobre las 49 posiciones se pierde dónde estaba cada cosa.

O sea que parte de lo que medimos podría ser un efecto del cuello de botella del promediado y no de
la red en sí. Hay que declararlo, y sugiere el experimento siguiente evidente: repetir el sondeo
desde el mapa de 7×7 antes de promediar, que es accesible con el mismo gancho que usamos para los
mapas de calor en la fase 5:

```python
h = base.features[-1].register_forward_hook(lambda m, e, s: g.__setitem__("act", s))
```

Si desde ahí la forma sí aparece, la conclusión se matiza y pasa a ser "la red la calcula pero la
descarta al promediar". Si no aparece, se refuerza mucho.

**Una asimetría menor del control.** Los resúmenes de la red ajustada vienen de cinco modelos
distintos (uno por pliegue), mientras que los genéricos vienen de un único extractor fijo. Eso añade
algo de varianza al lado ajustado. No invalida nada, sobre todo porque la diferencia va en las doce
medidas en la misma dirección, pero conviene mencionarlo antes de que lo mencionen por ti.

**Sobre el R² negativo.** El desplazamiento del núcleo sale en -0.03. Un R² por debajo de cero
significa que las predicciones son peores que responder siempre el promedio: no hay ninguna señal, y
lo poco que el modelo creyó encontrar en entrenamiento no se sostuvo fuera de muestra. Es el
resultado más limpio posible de "esta información no está".
