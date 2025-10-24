import pandas as pd

#Cargar los archivos
casos = pd.read_excel("casos_Tlaxcala_20251020_190233.xlsx", sheet_name="Sheet1")

# Guardar número de columnas antes de limpieza
num_cols_antes = casos.shape[1]
print(f"Número de columnas antes de limpieza: {num_cols_antes}")

# Función para normalizar nombres
def normalizar_columnas(df):
    df.columns = (
        df.columns.str.strip()             # quitar espacios
                  .str.lower()             # convertir a minúsculas
                  .str.replace(" ", "_")   # reemplazar espacios por _
                  .str.replace("á", "a")
                  .str.replace("é", "e")
                  .str.replace("í", "i")
                  .str.replace("ó", "o")
                  .str.replace("ú", "u")
                  .str.replace("ñ", "n")
    )
    return df

casos = normalizar_columnas(casos)
print("Columnas normalizadas")

#Eliminar columnas completamente vacías
casos = casos.dropna(axis=1, how="all")
#Si alguna columna tiene demasiados huecos (por ejemplo, más del 70% vacía)
casos = casos.dropna(axis=1, thresh=len(casos)*0.3)
print("Columnas vacías eliminadas.")
print(casos.columns)



#Evita tener el mismo expediente dos veces (eliminar valores duplicados)
duplicados = casos.duplicated().sum()
print(f"Registros duplicados: {duplicados}")

if duplicados > 0:
    casos = casos.drop_duplicates()
    print("Duplicados eliminados.")


#Convertir columna 'victima_edad' a números enteros
if "victima_edad" in casos.columns:
    # Quitar espacios, convertir a número cuando se pueda
    casos["victima_edad"] = pd.to_numeric(casos["victima_edad"].astype(str).str.strip(), errors="coerce").astype("Int64")

    # Reemplazar valores 0 o nulos por "No especificado"
    casos["victima_edad"] = casos["victima_edad"].apply(
        lambda x: "No especificado" if pd.isna(x) or x == 0 else int(x)
    )

#Eliminar columnas con más del 70% de ceros o valores nulos
cols_to_drop = []
for col in casos.columns:
    if pd.api.types.is_numeric_dtype(casos[col]):
        # Contar valores 0 o NaN
        zero_or_null_ratio = ((casos[col] == 0) | (casos[col].isnull())).mean()
        if zero_or_null_ratio > 0.70:
            cols_to_drop.append(col)

if cols_to_drop:
    print("Columnas eliminadas por tener más del 70% de ceros o valores nulos:")
    print(cols_to_drop)
    casos = casos.drop(columns=cols_to_drop)
else:
    print("No se encontraron columnas con más del 70% de ceros o valores nulos.")


#Eliminar columnas con aportación mínima
del casos["victima_migrante"]
del casos["victima_afrodescendiente"]
del casos["victima_identidad_genero"]
del casos["victima_vive_extranjero"]
del casos["victima_colonia"]
del casos["victima_localidad"]
del casos["victima_caso_name"]
del casos["hecho_extranjero_victima"]

# Después de la limpieza
num_cols_despues = casos.shape[1]
print(f"Número de columnas después de limpieza: {num_cols_despues}")

#Guardar archivo limpio
casos.to_excel("casos_limpio_Tlaxcala.xlsx", index=False)
print("Archivo limpio guardado como 'casos_limpio_Tlaxcala.xlsx'")










