# Caso de negocio y métricas de éxito

Resumen del problema
--------------------
El proyecto produce un modelo de pronóstico de demanda por tienda y producto basado en datos históricos de ventas e inventario. El objetivo es anticipar la demanda diaria para reducir faltantes (stock-outs) y sobre-stock, optimizar la reposición y disminuir el capital inmovilizado en inventario.

Principales métricas del modelo
------------------------------
- Métricas técnicas primarias:
  - RMSE (Raíz del Error Cuadrático Medio): mide el error absoluto en unidades. Umbral objetivo sugerido: RMSE <= 30 unidades.
  - MAE (Error Absoluto Medio): robusto ante outliers. Umbral objetivo sugerido: MAE <= 20 unidades.
  - MAPE (Error Porcentual Medio Absoluto): para medir error relativo. Umbral objetivo sugerido: MAPE <= 20%.

- Métricas secundarias:
  - Tasa de agotamiento predicha vs real (oos rate): objetivo reducir la tasa de agotamiento respecto a la línea base histórica.
  - Tasa de sobrestock detectada: objetivo disminuir costes por inventario sobrante.

Conexión con objetivos de negocio
--------------------------------
Estas métricas se conectan directamente con los objetivos de la empresa:
- RMSE/MAE/ MAPE reducidos → pronósticos más precisos → menores rupturas de stock y menos pedidos de emergencia.
- Menor tasa de agotamiento → aumenta ventas atendidas y reduce pérdida de venta.
- Menor sobrestock → reduce capital inmovilizado y costes de almacenamiento.

Notas sobre umbrales
--------------------
Los umbrales indicados son sugeridos y deben calibrarse con base en el tamaño medio de pedidos y la variabilidad histórica de las ventas por producto/tienda. Para algunos SKUs de alta variabilidad puede ser aceptable un RMSE mayor; para SKUs críticos (alto margen o rotación) exigir umbrales más estrictos.

Siguiente paso
--------------
Con los umbrales y la métrica definida, procederé a generar `requirements.txt` (ya creado) y preparar el servicio de inferencia para exponer el modelo mediante una API REST y las pruebas necesarias.
