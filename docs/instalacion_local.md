# Correr el proyecto en local

## 1. Entorno

En una terminal, en la carpeta donde tengas el notebook:

```bash
sudo apt install -y python3-venv python3-pip

python3 -m venv ~/venvs/sipakmed
source ~/venvs/sipakmed/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scikit-learn scikit-image opencv-python-headless \
            matplotlib statsmodels scipy pillow kagglehub jupyterlab ipywidgets
```

La primera línea de `pip install torch` es la importante: si instalas torch a secas puede bajarte
la versión de solo CPU y el entrenamiento pasa de minutos a horas. Comprueba con:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Si sale `False`, revisa el driver de NVIDIA (`nvidia-smi` debe responder).

Para abrir el notebook (con el entorno activado):

```bash
jupyter lab
```

Cada vez que abras una terminal nueva hay que volver a activar el entorno con
`source ~/venvs/sipakmed/bin/activate`.

### Que el notebook use este entorno

Si lanzas `jupyter lab` con el entorno activado, el kernel por defecto ya es el correcto y no hay
que seleccionar nada. Para comprobarlo, en una celda:

```python
import sys; print(sys.executable)
```

Debe contener `venvs/sipakmed`. Si dice `/usr/bin/python3`, estás en el Python del sistema.

Si prefieres abrir el notebook desde VS Code o desde un Jupyter instalado aparte, registra el
entorno como kernel con nombre propio:

```bash
pip install ipykernel
python -m ipykernel install --user --name sipakmed --display-name "SIPaKMeD (venv)"
```

Después, en JupyterLab: **Kernel → Change Kernel → SIPaKMeD (venv)**. En VS Code aparece en el
selector de kernel arriba a la derecha. Para quitarlo: `jupyter kernelspec uninstall sipakmed`.

## 2. Token de Kaggle

En kaggle.com → tu foto → Settings → API → Create New Token. Copia el valor y guárdalo:

```bash
mkdir -p ~/.kaggle
echo "TU_TOKEN_AQUI" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

El notebook lo lee de ahí solo. No lo escribas dentro de una celda.

## 3. Espacio en disco

El dataset ocupa unos 16 GB en total: 6.4 GB el comprimido más 9.2 GB descomprimido. `kagglehub`
lo deja en `~/.cache/kagglehub/`. Si vas justo de espacio, después de la primera descarga puedes
borrar el archivo `.archive` de esa carpeta.

Los resultados (CSV, pesos, índice) van a `~/sipakmed/`, unos 150 MB.

## 4. Ajustes según tu máquina

El notebook detecta GPU, RAM y disco en la primera celda. Dos cosas a vigilar:

**VRAM.** Con 6 GB, EfficientNet-B0 a 224 y lote 64 debería entrar usando precisión mixta. Si sale
`CUDA out of memory`, baja `LOTE` a 32 en la celda de configuración. Es más lento pero no cambia
los resultados de forma apreciable.

**RAM.** El notebook carga las 4049 imágenes a memoria (unos 610 MB) para que las épocas no
dependan del disco. Con 8 GB de RAM entra, pero cierra el navegador pesado mientras entrena. Si
notas que el sistema empieza a usar swap, baja `NUM_WORKERS` a 0: los procesos de carga comparten
la memoria por copia en escritura, pero cada uno suma algo de sobrecarga.

## 5. Cuánto tarda

Cada pliegue son unos 4 o 5 minutos en una RTX 4050. El notebook completo desde cero son unos 35
entrenamientos, o sea entre 2 y 3 horas.

Si ya corriste las partes en Colab, copia los CSV desde Drive a `~/sipakmed/` y el notebook los
reutiliza con `USAR_CACHE = True`:

- `indice_celulas.csv`
- `medidas_fase3.csv`
- `predicciones_fase2.csv`, `resultados_fase2.csv`
- `predicciones_fase3.csv`
- `predicciones_fase4.csv`, `emb_finetune.npy`

Con eso solo queda por correr la parte 5 (cinco entrenamientos más los mapas de calor), unos 40
minutos.

## 6. Si algo falla

- **`ImportError: libGL.so.1`** al importar cv2: instala `sudo apt install libgl1` o usa
  `opencv-python-headless` como está indicado arriba.
- **El kernel se muere sin mensaje** durante la carga de imágenes: es falta de RAM. Cierra
  aplicaciones y vuelve a intentar.
- **Entrenamiento lentísimo**: casi seguro estás en CPU. Revisa el punto 1.
