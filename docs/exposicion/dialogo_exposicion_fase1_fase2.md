# Diálogo de exposición — Fase 1 y Fase 2
*(≈3 minutos. Los bloques de código son para mostrar en pantalla mientras se habla, no para leerlos en voz alta.)*

---

## Introducción (≈15 s)

Buenas. Voy a presentar las dos primeras fases de nuestro proyecto de clasificación de células de papanicolaou con el conjunto SIPaKMeD. Trabajamos con 4.049 fotos de células, ya etiquetadas en cinco tipos. Esto se ha hecho muchas veces antes y los resultados publicados rondan el 99%, así que en vez de buscar el número más alto, decidimos revisar si esos resultados son reales o si vienen de errores en la forma de evaluar. Empiezo por la fase 1.

---

## Fase 1 — De dónde viene cada foto (≈1 min)

La fase 1 no entrena nada, solo revisa de dónde viene cada foto. Y ahí está el hallazgo clave: estas 4.049 células no vienen de 4.049 pacientes, vienen de solo 966 láminas completas. El nombre de archivo nos dice de qué lámina salió cada una.

**[Mostrar: celda 9]**
```python
filas.append({
    "clase": c.replace("im_", ""),
    # el nombre de la clase va delante porque la lamina "001" de una clase
    # no es la misma que la "001" de otra
    "lamina": c.replace("im_", "") + "_" + stem.split("_")[0],
    "stem": stem, "img": img,
    "cyt": os.path.join(cropped, stem + "_cyt.dat"),
    "nuc": os.path.join(cropped, stem + "_nuc.dat"),
})
...
print("Células por lámina — media/mín/máx:", ...)
```
```
Células por lámina — media/mín/máx: 4.2   1   26
```

Con este código armamos una tabla que identifica la lámina de cada célula a partir del nombre del archivo: `001_01` y `001_02` son vecinas de la misma lámina. El resultado: en promedio hay 4.2 células por lámina, con un caso extremo de 26. Cada célula tiene, en promedio, tres o cuatro hermanas casi idénticas. Ese es el riesgo que medimos en la fase 2: si mezclamos hermanas entre el grupo de práctica y el de examen, el modelo aprueba el examen porque ya vio algo casi igual, no porque aprendió.

---

## Fase 2 — Medir de forma honesta (≈1:30)

En la fase 2 medimos cuánto se infla el resultado por ese problema. La estrategia: entrenar la misma red dos veces, cambiando una sola cosa, cómo se reparten las fotos.

**[Mostrar: celda 18]**
```python
# Reparto A: al azar, sin mirar de que lamina viene cada celula
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
...
# Reparto B: por lamina, las hermanas caen siempre del mismo lado
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
...
for col in ["fold_azar", "fold_lamina"]:
    s = [... "de células con hermanas al otro lado"]
```
```
fold_azar:   91.8% de células con hermanas al otro lado
fold_lamina:  0.0% de células con hermanas al otro lado
```

El primer reparto es al azar, como se hace normalmente. El segundo agrupa por lámina, para que las hermanas queden siempre del mismo lado. El reparto al azar deja al 91.8% de las células de examen con una hermana en el entrenamiento; el reparto por lámina baja eso a 0%. Entrenamos cada reparto cinco veces, con la misma red, mismas épocas y misma semilla, para que la única diferencia posible sea el reparto en sí.

**[Mostrar: celda 23]**
```python
print(f"Reparto al azar:    {a:.4f}")
print(f"Reparto por lámina: {b:.4f}")
...
print(f"El reparto al azar esconde el {(err_b-err_a)/err_b:.1%} de los errores")
```
```
Reparto al azar:    0.9796
Reparto por lámina: 0.9731
El reparto al azar esconde el 24.2% de los errores
```

Con reparto al azar el modelo llega a 0.9796 de F1 macro; con reparto por lámina, el honesto, a 0.9731. La diferencia parece mínima, 0.65 puntos, pero contada en errores es otra historia: el modelo honesto falla en 2.69% de las células, el otro en 2.04%. El reparto al azar esconde casi 1 de cada 4 errores. Además, la variación entre repeticiones baja de 0.0096 a 0.0037 —menos de la mitad—, así que el número también parece más estable de lo que es en realidad. Al repetir esto en otra máquina la magnitud cambió, entre 24% y 41% de errores escondidos, pero siempre en la misma dirección.

---

## Cierre (≈20 s)

Esto no dice que el modelo no aprenda: con evaluación honesta el acierto sigue en 0.973. Lo que se infla no es la capacidad del modelo, sino la confianza con la que creíamos estar midiéndola. Por eso las fases 1 y 2 son la base obligatoria de todo el trabajo: sin ellas, ningún resultado posterior se sostiene. Gracias.
