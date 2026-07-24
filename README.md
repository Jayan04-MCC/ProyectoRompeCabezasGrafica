# 🧩 Reconstrucción de Rompecabezas y Clasificación de Objetos

Proyecto de **Visión por Computadora** que combina dos módulos independientes:

1. **Reconstrucción de rompecabezas** a partir de piezas sueltas, comparando únicamente sus **bordes** (color, gradientes de Sobel y continuidad con Canny).
2. **Detección y clasificación de objetos** mediante **YOLOv8** reentrenado (transfer learning) sobre el dataset **COCO**, agrupando las detecciones en **4 macroclases**: `Persona`, `Animal`, `Fruta`, `Objeto`.

Desarrollado en **Google Colab** con aceleración por **GPU (NVIDIA T4)**.

---

## 📑 Tabla de contenidos

- [Características](#-características)
- [Requisitos](#-requisitos)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Parte 1: Reconstrucción de rompecabezas](#-parte-1-reconstrucción-de-rompecabezas)
- [Parte 2: Clasificación de objetos con YOLOv8](#-parte-2-clasificación-de-objetos-con-yolov8)
- [Cómo ejecutar](#-cómo-ejecutar)
- [Resultados](#-resultados)
- [Autor](#-autor)

---

## ✨ Características

- **Corrección automática de piezas:** detecta el contorno de cada pieza (canal alfa), calcula su orientación con `minAreaRect` y la endereza (deskew) antes de compararla.
- **Comparación de bordes por tres técnicas:**
  - **Color (espacio LAB):** SSD entre columnas/filas del borde con búsqueda de desfase.
  - **Sobel (MGC):** compatibilidad de gradientes para penalizar discontinuidades.
  - **Canny:** continuidad de bordes detectados entre piezas contiguas.
- **Armado automático del tablero:** estima el tamaño (filas × columnas) según el número de piezas y su relación de aspecto, y arma el rompecabezas por costo mínimo probando distintas semillas.
- **Filtro Sobel implementado desde cero** (convolución manual 3×3), además del uso de OpenCV.
- **Pipeline completo de dataset:** descarga selectiva de COCO con FiftyOne, remapeo de 29 clases a 4 macroclases, división `train/val/test` y exportación a formato YOLO.
- **Entrenamiento con transfer learning** (congelando las primeras capas) y evaluación con métricas `mAP50` y `mAP50-95` por clase.

---

## 🛠 Requisitos

```bash
pip install ultralytics fiftyone scikit-learn opencv-python numpy matplotlib
```

| Componente | Versión / Detalle |
|---|---|
| Python | 3.x |
| Entorno | Google Colab (GPU T4 recomendada) |
| OpenCV | Procesamiento de imágenes |
| Ultralytics | YOLOv8 |
| FiftyOne | Descarga y gestión del dataset COCO |
| NumPy / Matplotlib | Cálculo y visualización |

---

## 📂 Estructura del proyecto

```
Rompecabezas_y_Clasificacion_de_Objetos.ipynb
│
├── Parte 1 — Reconstrucción de rompecabezas
│   ├── Carga del ZIP con la referencia y las piezas
│   ├── Detección de referencia / piezas por nombre
│   ├── Preparación de piezas (contorno, rotación, recorte)
│   ├── Comparación de bordes (Color LAB + Sobel + Canny)
│   └── Armado del tablero y reconstrucción final
│
└── Parte 2 — Clasificación de objetos (YOLOv8)
    ├── Descarga de COCO (FiftyOne)
    ├── Remapeo de 29 clases → 4 macroclases
    ├── División train/val/test y export a formato YOLO
    ├── Entrenamiento (transfer learning, 50 épocas)
    └── Evaluación (mAP) y prueba sobre la imagen reconstruida
```

---

## 🧩 Parte 1: Reconstrucción de rompecabezas

### Entrada
Un archivo **`imagen1.zip`** que contenga:
- Una imagen cuyo nombre incluya **`referencia`**.
- Las piezas cuyos nombres incluyan **`pieza`** (formato PNG con canal alfa / transparencia).

### Proceso
1. **Preparación:** por cada pieza se extrae el contorno usando el canal alfa, se calcula su ángulo con `minAreaRect`, se rota para enderezarla y se recorta a su zona ocupada.
2. **Cálculo de costos entre bordes**, combinando:
   - `color_ssd` → diferencia de color en el borde (espacio LAB).
   - `comparar_sobel` → diferencia de gradientes (Sobel / MGC).
   - `canny_mismatch` → discontinuidad de bordes (Canny).
3. **Estimación del tablero:** se calculan las combinaciones `filas × columnas` cuyo producto es el número de piezas y se elige la más cercana a la relación de aspecto.
4. **Armado:** se coloca una pieza semilla y se añaden vecinas minimizando el costo total; se prueban todas las semillas y se conserva el mejor armado.

### Salida
- `reconstruida_bordes.png` → imagen final reconstruida.
- Orden de las piezas impreso en consola.

---

## 🎯 Parte 2: Clasificación de objetos con YOLOv8

Se parte de **YOLOv8n** preentrenado en **COCO (80 clases)** y se reentrena para reconocer **4 macroclases**:

| ID | Macroclase | Clases COCO incluidas |
|----|-----------|-----------------------|
| 0 | **Persona** | `person` |
| 1 | **Animal** | `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe` |
| 2 | **Fruta** | `apple`, `banana`, `orange`, `broccoli`, `carrot` |
| 3 | **Objeto** | `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `chair`, `couch`, `bed`, `dining table`, `tv`, `laptop`, `cell phone`, `book`, `clock` |

### Pipeline
1. **Descarga** de hasta 6000 imágenes de COCO con FiftyOne (solo las clases de interés).
2. **Exportación** al formato `YOLOv5Dataset` (imágenes + etiquetas `.txt`).
3. **Remapeo** de las clases originales a los IDs `0–3` de las macroclases.
4. **División** del dataset: **65% train / 15% val / 20% test** (semilla fija = 42).
5. **Configuración** del archivo `dataset.yaml`.
6. **Entrenamiento** con transfer learning:

```python
results = model.train(
    data="/content/dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    freeze=10,      # congela las primeras capas
    augment=True,   # aumento de datos
)
```

7. **Evaluación** sobre el conjunto `test` (`mAP50`, `mAP50-95`, métricas por clase, matriz de confusión).
8. **Prueba final:** se ejecuta el modelo entrenado sobre la imagen `reconstruida_bordes.png` generada en la Parte 1.

---

## ▶️ Cómo ejecutar

1. Abre el notebook **`Rompecabezas_y_Clasificacion_de_Objetos.ipynb`** en Google Colab.
2. Selecciona un entorno de ejecución con **GPU** (`Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`).
3. **Parte 1:** ejecuta las celdas y sube tu `imagen1.zip` cuando se solicite.
4. **Parte 2:** ejecuta las celdas para descargar COCO, entrenar y evaluar el modelo.
   > ⚠️ La descarga de COCO y el entrenamiento (50 épocas) pueden tardar bastante según la GPU disponible.
5. Revisa la reconstrucción, las métricas y la predicción final.

---

## 📊 Resultados

El notebook genera automáticamente:

- 🧩 **Imagen reconstruida** del rompecabezas (`reconstruida_bordes.png`).
- 📈 **Curvas de entrenamiento** (`results.png`).
- 🔲 **Matriz de confusión** (`confusion_matrix.png`).
- 📋 **Métricas por clase** (`mAP50` para Persona, Animal, Fruta y Objeto).
- 💾 **Modelo entrenado** guardado en `best_yolov8n_custom.pt`.

---

## 👤 Autor

**Jayan Michael Cáceres Cuba**

> Proyecto académico de Visión por Computadora — reconstrucción de rompecabezas y detección de objetos con Deep Learning.
