import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

# Configuration de base
st.set_page_config(page_title="NYC Taxi ML - Simple View", layout="wide")
st.title("🚖 NYC Taxi ML - Résultats")

# Dossier des données (ajustable si besoin)
data_dir = "."

def load_data(folder_name):
    # Cherche les dossiers qui contiennent le nom (ex: "metrics")
    target_folders = [d for d in glob.glob(os.path.join(data_dir, "**"), recursive=True) 
                     if os.path.isdir(d) and folder_name.lower() in d.lower()]
    
    all_files = []
    for folder in target_folders:
        all_files.extend(glob.glob(os.path.join(folder, "*.parquet")))
    
    if not all_files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)

# 1. Chargement et affichage des métriques
st.header("📊 Métriques des Modèles")
df_metrics = load_data("metrics")
if not df_metrics.empty:
    df_metrics = df_metrics.drop_duplicates().reset_index(drop=True)
    st.dataframe(df_metrics, use_container_width=True)
    
    # Graphique de comparaison simple
    fig_metrics = px.bar(df_metrics, x="model_name", y=["r2", "rmse"], barmode="group", title="Comparaison R2 et RMSE")
    st.plotly_chart(fig_metrics, use_container_width=True)
else:
    st.warning("Aucune métrique trouvée.")

st.markdown("---")

# 2. Chargement et affichage des prédictions
st.header("📈 Prédictions (Réel vs Prédit)")
df_preds = load_data("predictions")
if not df_preds.empty:
    st.write(f"Total des lignes analysées : **{len(df_preds):,}**")
    
    # Échantillonnage pour la fluidité du graphique
    sample_size = min(2000, len(df_preds))
    sample_df = df_preds.sample(sample_size)
    
    fig_scatter = px.scatter(
        sample_df, 
        x="total_amount", 
        y="prediction", 
        title=f"Nuage de points (Échantillon de {sample_size} lignes)",
        labels={"total_amount": "Prix Réel ($)", "prediction": "Prix Prédit ($)"},
        opacity=0.5
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.subheader("Extrait des données")
    st.dataframe(df_preds.head(50), use_container_width=True)
else:
    st.warning("Aucune prédiction trouvée.")
