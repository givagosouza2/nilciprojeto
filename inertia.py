import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import io

st.set_page_config(layout="wide")

st.title("Análise Vetorial com Elipse de Inércia (a partir de coordenadas espaciais)")

st.markdown("""
Este aplicativo calcula vetores a partir de coordenadas X e Y em série temporal
e analisa sua **distribuição vetorial**, incluindo a **elipse de inércia**, seus eixos principais e o **S-index**.
""")

uploaded_file = st.file_uploader(
    "📄 Carregue um arquivo .txt ou .csv com colunas: Tempo, X, Y, ...",
    type=["txt", "csv"]
)

def ler_arquivo(uploaded_file):
    """
    Lê arquivos txt/csv com separadores variados:
    tab, vírgula, ponto e vírgula ou múltiplos espaços.
    """
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")

    # Tenta primeiro como tabulado
    try:
        df = pd.read_csv(io.StringIO(text), sep=r"\t+", engine="python")
        if df.shape[1] >= 3:
            return df
    except:
        pass

    # Tenta separadores mistos
    try:
        df = pd.read_csv(io.StringIO(text), sep=r"[;,]+|\s{2,}", engine="python")
        if df.shape[1] >= 3:
            return df
    except:
        pass

    # Última tentativa: qualquer whitespace
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")
    return df

def padronizar_colunas(df):
    df.columns = [str(c).strip() for c in df.columns]

    # tenta localizar X e Y por nome
    colunas_lower = {c.lower(): c for c in df.columns}

    x_col = None
    y_col = None
    t_col = None

    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["x", "posx"]:
            x_col = c
        elif cl in ["y", "posy"]:
            y_col = c
        elif cl in ["time", "tempo", "t"]:
            t_col = c

    # fallback por posição
    if t_col is None and df.shape[1] >= 1:
        t_col = df.columns[0]
    if x_col is None and df.shape[1] >= 2:
        x_col = df.columns[1]
    if y_col is None and df.shape[1] >= 3:
        y_col = df.columns[2]

    return df, t_col, x_col, y_col

def calcular_elipse_inercia(x, y):
    X = np.vstack([x, y])
    cov = np.cov(X)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.maximum(eigvals, 0)  # evita negativo numérico pequeno

    eixo_maior = np.sqrt(eigvals[0])
    eixo_menor = np.sqrt(eigvals[1])

    razao = eixo_maior / eixo_menor if eixo_menor > 0 else np.inf
    angulo = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    s_index = razao

    return eixo_maior, eixo_menor, razao, angulo, s_index, eigvecs, eigvals

if uploaded_file:
    try:
        df = ler_arquivo(uploaded_file)
        df, t_col, x_col, y_col = padronizar_colunas(df)

        if df.shape[1] < 3:
            st.error("O arquivo deve conter pelo menos três colunas: tempo, X e Y.")
            st.stop()

        # converter para numérico
        df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

        df = df.dropna(subset=[x_col, y_col]).reset_index(drop=True)

        if len(df) < 2:
            st.error("O arquivo não possui dados suficientes para calcular vetores.")
            st.stop()

        tempo = df[t_col].values if t_col in df.columns else np.arange(len(df))
        x = df[x_col].values
        y = df[y_col].values

        dx = np.diff(x)
        dy = np.diff(y)

        if len(dx) < 2:
            st.error("Não há vetores suficientes para a análise.")
            st.stop()

        st.subheader("Pré-visualização dos dados")
        st.dataframe(df.head())

        st.subheader("Vetores calculados")
        st.write(f"Total de pontos: {len(x)}")
        st.write(f"Total de vetores: {len(dx)}")

        eixo_maior, eixo_menor, razao, angulo, s_index, eigvecs, eigvals = calcular_elipse_inercia(dx, dy)

        st.subheader("📊 Resultados")
        st.write(f"**Eixo maior:** {eixo_maior:.4f}")
        st.write(f"**Eixo menor:** {eixo_menor:.4f}")
        st.write(f"**Razão entre eixos:** {razao:.4f}")
        st.write(f"**Ângulo da elipse (graus):** {angulo:.2f}°")
        st.write(f"**S-index:** {s_index:.4f}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📍 Passo 1: Posições registradas")
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.plot(x, y, 'o-', alpha=0.6, label="Trajetória")
            ax1.set_title("Coordenadas originais (X, Y)")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.axhline(0, color='gray', lw=1)
            ax1.axvline(0, color='gray', lw=1)

            margem_x = max(10, 0.05 * (np.max(x) - np.min(x) + 1))
            margem_y = max(10, 0.05 * (np.max(y) - np.min(y) + 1))
            ax1.set_xlim(np.min(x) - margem_x, np.max(x) + margem_x)
            ax1.set_ylim(np.min(y) - margem_y, np.max(y) + margem_y)

            ax1.set_aspect('equal')
            ax1.legend()
            st.pyplot(fig1)

        with col2:
            st.subheader("📈 Passo 2: Vetores de deslocamento e elipse de inércia")
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.scatter(dx, dy, alpha=0.6, label="Vetores de deslocamento")

            ax2.quiver(
                np.zeros_like(dx), np.zeros_like(dy), dx, dy,
                angles='xy', scale_units='xy', scale=1, alpha=0.5
            )

            width, height = 2 * np.sqrt(eigvals)
            ellipse = Ellipse(
                (0, 0), width, height, angle=angulo,
                edgecolor='red', fc='None', lw=2, label='Elipse de inércia'
            )
            ax2.add_patch(ellipse)

            ax2.axhline(0, color='black', lw=1)
            ax2.axvline(0, color='black', lw=1)

            lim = max(np.max(np.abs(dx)), np.max(np.abs(dy)), 1) * 1.2
            ax2.set_xlim(-lim, lim)
            ax2.set_ylim(-lim, lim)

            ax2.set_aspect('equal')
            ax2.legend()
            ax2.set_title("Distribuição Vetorial com Setas e Elipse de Inércia")
            ax2.set_xlabel("ΔX")
            ax2.set_ylabel("ΔY")
            st.pyplot(fig2)

        st.subheader("🎮 Passo 3: Animação frame a frame")

        # frame vai de 1 até len(dx)-1 para evitar estouro
        max_frame = len(dx) - 1
        frame = st.slider(
            "Deslize para visualizar a trajetória e vetores acumulados:",
            min_value=1,
            max_value=max_frame,
            value=1
        )

        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: trajetória acumulada com vetores
        ax3a.plot(x[:frame+1], y[:frame+1], '-', color='black')

        for i in range(frame):
            ax3a.arrow(
                x[i], y[i], dx[i], dy[i],
                head_width=2, head_length=2,
                fc='gray', ec='black', length_includes_head=True
            )

        ax3a.arrow(
            x[frame], y[frame], dx[frame], dy[frame],
            head_width=2, head_length=2,
            fc='blue', ec='blue', length_includes_head=True
        )

        ax3a.set_title("Trajetória acumulada com vetores")
        ax3a.set_xlabel("X")
        ax3a.set_ylabel("Y")
        ax3a.set_xlim(np.min(x) - margem_x, np.max(x) + margem_x)
        ax3a.set_ylim(np.min(y) - margem_y, np.max(y) + margem_y)

        # Subplot 2: vetores acumulados com elipse parcial
        ax3b.quiver(
            np.zeros_like(dx[:frame+1]),
            np.zeros_like(dy[:frame+1]),
            dx[:frame+1], dy[:frame+1],
            angles='xy', scale_units='xy', scale=1,
            color='black', alpha=0.5
        )

        if frame > 2:
            eixo_maior_f, eixo_menor_f, razao_f, angulo_f, s_index_f, eigvecs_f, eigvals_f = calcular_elipse_inercia(
                dx[:frame+1], dy[:frame+1]
            )
            width_f, height_f = 2 * np.sqrt(eigvals_f)
            ellipse_f = Ellipse(
                (0, 0), width_f, height_f,
                angle=angulo_f, edgecolor='red', fc='None', lw=2
            )
            ax3b.add_patch(ellipse_f)
            ax3b.set_title(f"Vetores até o frame {frame}\nS-index parcial: {s_index_f:.2f}")
        else:
            ax3b.set_title("Vetores acumulados")

        ax3b.axhline(0, color='black', lw=1)
        ax3b.axvline(0, color='black', lw=1)
        ax3b.set_xlabel("ΔX")
        ax3b.set_ylabel("ΔY")
        ax3b.set_xlim(-lim, lim)
        ax3b.set_ylim(-lim, lim)

        st.pyplot(fig3)

    except Exception as e:
        st.error(f"Erro ao abrir/processar o arquivo: {e}")

else:
    st.info("Aguardando upload de arquivo com colunas: tempo, X, Y...")
