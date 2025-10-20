import pandas as pd

xls = pd.ExcelFile("inputs/regular NBA.xlsx")
print("Feuilles :", xls.sheet_names)

df = pd.read_excel(xls, sheet_name="Données NBA", header=1)
print("\nColonnes de la feuille 'Données NBA' :")
print(df.columns.tolist())
