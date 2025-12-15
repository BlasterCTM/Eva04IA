"""Interfaz grafica para visualizar los graficos generados por el pipeline."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import tkinter as tk
from tkinter import ttk

try:  # pillow mejora la calidad de escalado si está disponible
    from PIL import Image, ImageTk
except ModuleNotFoundError:
    Image = None
    ImageTk = None

_EXT_VALIDAS = {".png", ".jpg", ".jpeg"}


def _normalizar_rutas(rutas: Sequence[Path | str | None]) -> list[Path]:
    """Devuelve solo las rutas validas y existentes."""
    rutas_validas: list[Path] = []
    for ruta in rutas:
        if ruta is None:
            continue
        ruta_path = Path(ruta)
        if not ruta_path.exists():
            continue
        if ruta_path.suffix.lower() not in _EXT_VALIDAS:
            continue
        rutas_validas.append(ruta_path)
    return rutas_validas


def _abrir_en_sistema(ruta: Path) -> bool:
    """Intenta abrir un archivo con la aplicacion predeterminada."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(ruta)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(ruta)], check=False)
        else:
            subprocess.run(["xdg-open", str(ruta)], check=False)
        return True
    except Exception:
        return False


def _cargar_imagen_escalada(ruta: Path, ancho_maximo: int) -> tk.PhotoImage | None:
    """Carga una imagen reescalada respetando la proporcion original."""
    if Image is not None and ImageTk is not None:
        try:
            imagen = Image.open(ruta)
        except Exception:
            imagen = None
        else:
            ancho_actual = max(imagen.width, 1)
            escala = min(ancho_maximo / ancho_actual, 1.0)
            if escala < 1.0:
                nuevo_tamano = (
                    max(int(imagen.width * escala), 1),
                    max(int(imagen.height * escala), 1),
                )
                imagen = imagen.resize(nuevo_tamano, Image.LANCZOS)
            return ImageTk.PhotoImage(imagen)

    try:
        imagen_tk = tk.PhotoImage(file=str(ruta))
    except tk.TclError:
        return None

    ancho_actual = imagen_tk.width()
    if ancho_actual > ancho_maximo:
        factor = max(int(math.ceil(ancho_actual / ancho_maximo)), 1)
        imagen_tk = imagen_tk.subsample(factor, factor)

    return imagen_tk


def _formatear_celda(valor: Any) -> str:
    """Devuelve una representacion en texto para la celda de tabla."""
    if hasattr(valor, "item"):
        try:
            valor = valor.item()
        except Exception:
            pass
    if isinstance(valor, float):
        return f"{valor:,.2f}"
    if isinstance(valor, int):
        return f"{valor:,}"
    return "" if valor is None else str(valor)


def mostrar_interfaz_graficos(
    secciones: Mapping[str, Sequence[Path | str | None]],
    titulo: str = "Dashboard de graficos",
    kpis: Mapping[str, Any] | None = None,
    tablas: Mapping[str, Any] | None = None,
) -> None:
    """Crea una ventana agrupando las imagenes y tablas del dashboard."""
    secciones_normalizadas: dict[str, list[Path]] = {}
    total_imagenes = 0
    for nombre, rutas in secciones.items():
        rutas_validas = _normalizar_rutas(rutas)
        if rutas_validas:
            secciones_normalizadas[nombre] = rutas_validas
            total_imagenes += len(rutas_validas)

    tablas_normalizadas: dict[str, Any] = {}
    if tablas:
        for nombre, tabla in tablas.items():
            if tabla is None:
                continue
            if hasattr(tabla, "empty") and getattr(tabla, "empty", False):
                continue
            if hasattr(tabla, "columns") and hasattr(tabla, "iterrows"):
                tablas_normalizadas[nombre] = tabla

    kpis_items: list[tuple[str, str]] = []
    if kpis:
        fecha = kpis.get("fecha_hoy")
        if fecha:
            kpis_items.append(("Fecha", str(fecha)))
        kpis_items.append(("Productos en riesgo", str(kpis.get("productos_en_riesgo", 0))))
        capital = kpis.get("capital_inmovilizado", 0)
        kpis_items.append(("Capital inmovilizado", f"${capital:,.2f}"))
        ventas = kpis.get("ventas_pronosticadas", 0)
        kpis_items.append(("Ventas pronosticadas", f"{ventas:,.0f} unidades"))

    if total_imagenes == 0 and not tablas_normalizadas and not kpis_items:
        print("[UI] No hay contenidos disponibles para construir el dashboard.")
        return

    if total_imagenes == 0:
        print("[UI] No hay imagenes disponibles para construir el dashboard.")
        return

    root = tk.Tk()
    root.title(titulo)
    root.geometry("1280x720")
    root.minsize(960, 600)
    root.configure(background="#0f172a")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TNotebook", padding=6)
    style.configure("TNotebook.Tab", padding=(16, 8))
    style.configure("Caption.TLabel", font=("Segoe UI", 10, "bold"))
    style.configure("CardTitle.TLabel", font=("Segoe UI", 12, "bold"))
    style.configure("Status.TLabel", font=("Segoe UI", 9))

    header = tk.Frame(root, bg="#0b7285")
    header.pack(fill="x")
    titulo_label = tk.Label(
        header,
        text=titulo,
        font=("Segoe UI", 18, "bold"),
        bg="#0b7285",
        fg="white",
    )
    titulo_label.pack(anchor="w", padx=18, pady=10)

    resumen_partes: list[str] = []
    if secciones_normalizadas:
        resumen_partes.append(f"{len(secciones_normalizadas)} secciones de imagen")
    if tablas_normalizadas:
        resumen_partes.append(f"{len(tablas_normalizadas)} tablas")
    if kpis_items:
        resumen_partes.append("KPIs activos")

    resumen_label = tk.Label(
        header,
        text=" | ".join(resumen_partes) if resumen_partes else "",
        font=("Segoe UI", 11),
        bg="#0b7285",
        fg="white",
    )
    resumen_label.pack(anchor="w", padx=18, pady=(0, 12))

    if kpis_items:
        kpi_frame = ttk.Frame(root, padding=(18, 12))
        kpi_frame.pack(fill="x")
        for idx, (titulo_kpi, valor_kpi) in enumerate(kpis_items):
            card = ttk.Frame(kpi_frame, padding=12, relief="ridge", borderwidth=1)
            card.grid(row=0, column=idx, padx=8, sticky="nsew")
            kpi_frame.columnconfigure(idx, weight=1)
            ttk.Label(card, text=titulo_kpi, style="Caption.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(card, text=valor_kpi, font=("Segoe UI", 14, "bold")).grid(
                row=1, column=0, sticky="w", pady=(6, 0)
            )

    content = ttk.Frame(root, padding=12)
    content.pack(fill="both", expand=True)
    content.grid_rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=1)

    status_var = tk.StringVar(value="Listo")

    def _actualizar_status(mensaje: str) -> None:
        status_var.set(mensaje)

    def _abrir_imagen_con_status(r: Path) -> None:
        _actualizar_status(f"Abriendo imagen: {r.name}")
        if _abrir_en_sistema(r):
            _actualizar_status(f"Imagen abierta: {r.name}")
        else:
            _actualizar_status(f"No se pudo abrir: {r.name}")

    notebook = ttk.Notebook(content)
    notebook.grid(row=0, column=0, sticky="nsew")

    imagenes_referencias: list[tk.PhotoImage] = []

    def _desplazar_rodillo(event: tk.Event, lienzo: tk.Canvas) -> None:
        if event.delta:
            lienzo.yview_scroll(int(-event.delta / 120), "units")

    for nombre, rutas in secciones_normalizadas.items():
        frame = ttk.Frame(notebook, padding=6)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        notebook.add(frame, text=nombre)

        canvas = tk.Canvas(frame, highlightthickness=0, bg="#f8fafc")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        contenedor = ttk.Frame(canvas, padding=12)
        contenedor.columnconfigure(0, weight=1)
        canvas.create_window((0, 0), window=contenedor, anchor="nw")

        def _configurar_scroll(event: tk.Event, lienzo: tk.Canvas = canvas) -> None:
            lienzo.configure(scrollregion=lienzo.bbox("all"))

        contenedor.bind("<Configure>", _configurar_scroll)

        def _activar_scroll(event: tk.Event, lienzo: tk.Canvas = canvas) -> None:
            lienzo.bind_all("<MouseWheel>", lambda e: _desplazar_rodillo(e, lienzo))

        def _desactivar_scroll(event: tk.Event, lienzo: tk.Canvas = canvas) -> None:
            lienzo.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _activar_scroll)
        canvas.bind("<Leave>", _desactivar_scroll)

        if not rutas:
            ttk.Label(contenedor, text="No se encontraron imagenes en esta seccion.").grid(
                row=0, column=0, padx=16, pady=16, sticky="w"
            )
            continue

        fila = 0
        for ruta in rutas:
            imagen = _cargar_imagen_escalada(ruta, ancho_maximo=1050)
            if imagen is None:
                ttk.Label(
                    contenedor,
                    text=f"No se pudo cargar la imagen {ruta.name}",
                ).grid(row=fila, column=0, padx=16, pady=8, sticky="w")
                fila += 1
                continue

            imagenes_referencias.append(imagen)

            card = ttk.Frame(contenedor, padding=16, relief="ridge", borderwidth=1)
            card.grid(row=fila, column=0, sticky="nsew", padx=8, pady=8)
            card.columnconfigure(0, weight=1)

            ttk.Label(card, text=ruta.stem, style="CardTitle.TLabel").grid(
                row=0, column=0, sticky="w"
            )
            ttk.Separator(card, orient="horizontal").grid(
                row=1, column=0, sticky="ew", pady=8
            )

            etiqueta = ttk.Label(card, image=imagen, anchor="center")
            etiqueta.image = imagen
            etiqueta.grid(row=2, column=0, pady=(4, 10))

            detalles = ttk.Frame(card)
            detalles.grid(row=3, column=0, sticky="w")
            ttk.Label(detalles, text="Archivo:").grid(row=0, column=0, sticky="w")
            ttk.Label(detalles, text=ruta.name).grid(row=0, column=1, sticky="w", padx=(6, 0))
            ttk.Label(detalles, text="Ubicacion:").grid(row=1, column=0, sticky="w")
            ttk.Label(detalles, text=str(ruta.parent)).grid(
                row=1, column=1, sticky="w", padx=(6, 0)
            )

            acciones = ttk.Frame(card)
            acciones.grid(row=4, column=0, sticky="w", pady=(12, 0))

            ttk.Button(
                acciones,
                text="Abrir imagen",
                command=lambda r=ruta: _abrir_imagen_con_status(r),
            ).pack(side="left")

            fila += 1

    for nombre, tabla in tablas_normalizadas.items():
        frame = ttk.Frame(notebook, padding=12)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        notebook.add(frame, text=nombre)

        tabla_df = tabla.copy()
        columnas = [str(columna) for columna in getattr(tabla_df, "columns", [])]

        filtro_columna = None
        for candidata in ("Supermercado", "Store ID"):
            if candidata in tabla_df.columns:
                filtro_columna = candidata
                break

        filtro_var = tk.StringVar(value="Todos")
        filtro_series = None

        if filtro_columna:
            filtro_series = tabla_df[filtro_columna].astype(str)
            opciones = sorted(filtro_series.dropna().unique())
            valores_combo = ["Todos"] + opciones

            filtro_frame = ttk.Frame(frame)
            filtro_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
            ttk.Label(filtro_frame, text=f"Filtrar por {filtro_columna}:").pack(side="left")
            combo = ttk.Combobox(
                filtro_frame,
                textvariable=filtro_var,
                values=valores_combo,
                state="readonly",
                width=24,
            )
            combo.pack(side="left", padx=(8, 0))
            combo.current(0)

        tree = ttk.Treeview(frame, columns=columnas, show="headings")
        scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        tree.grid(row=1, column=0, sticky="nsew")
        scrollbar_y.grid(row=1, column=1, sticky="ns")
        scrollbar_x.grid(row=2, column=0, sticky="ew")

        for columna in columnas:
            tree.heading(columna, text=columna)
            tree.column(columna, width=max(120, len(columna) * 10), anchor="w")

        def _poblar_tree(df, mensaje: str | None = None) -> None:
            tree.delete(*tree.get_children())
            for _, fila in df.iterrows():
                valores = [_formatear_celda(fila[columna]) for columna in columnas]
                tree.insert("", "end", values=valores)
            if mensaje:
                _actualizar_status(mensaje)

        def _aplicar_filtro(*_args) -> None:
            if filtro_columna and filtro_series is not None:
                valor = filtro_var.get()
                if valor != "Todos":
                    mascara = filtro_series == valor
                    df_filtrado = tabla_df.loc[mascara]
                else:
                    df_filtrado = tabla_df
                _poblar_tree(
                    df_filtrado,
                    f"{nombre}: {len(df_filtrado)} registros visibles (filtro {valor})",
                )
            else:
                _poblar_tree(tabla_df, f"{nombre}: {len(tabla_df)} registros visibles")

        _aplicar_filtro()

        if filtro_columna and filtro_series is not None:
            combo.bind("<<ComboboxSelected>>", _aplicar_filtro)

    status_bar = ttk.Label(
        root,
        textvariable=status_var,
        style="Status.TLabel",
        anchor="w",
        padding=(12, 6),
    )
    status_bar.pack(fill="x", padx=12, pady=(0, 8))

    print("[UI] Ventana de dashboard generada. Cierrela para continuar.")
    root.mainloop()
