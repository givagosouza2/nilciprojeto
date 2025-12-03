import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

st.set_page_config(layout="wide")

st.title("Análise Vetorial com Elipse de Inércia (a partir de coordenadas espaciais)")

st.markdown("""
Este aplicativo calcula vetores a partir de coordenadas X e Y em série temporal
e analisa sua **distribuição vetorial**, incluindo a **elipse de inércia**, seus eixos principais e o **S-index**.
""")

uploaded_file = st.file_uploader(
    "📄 Carregue um arquivo .txt ou .csv com colunas: Tempo, X, Y, ...", type=["txt", "csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')

    if df.shape[1] < 3:
        st.error("O arquivo deve conter pelo menos três colunas: tempo, X e Y.")
        st.stop()

    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values

    dx = np.diff(x)
    dy = np.diff(y)

    st.subheader("Vetores calculados")
    st.write(f"Total de vetores: {len(dx)}")

    def calcular_elipse_inercia(x, y):
        X = np.vstack([x, y])
        cov = np.cov(X)
        eigvals, eigvecs = np.linalg.eig(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eixo_maior = np.sqrt(eigvals[0])
        eixo_menor = np.sqrt(eigvals[1])
        razao = eixo_maior / eixo_menor if eixo_menor != 0 else np.inf
        angulo = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        s_index = razao

        return eixo_maior, eixo_menor, razao, angulo, s_index, eigvecs, eigvals

    eixo_maior, eixo_menor, razao, angulo, s_index, eigvecs, eigvals = calcular_elipse_inercia(
        dx, dy)

    st.subheader("📊 Resultados")
    st.write(f"**Eixo maior:** {eixo_maior:.4f}")
    st.write(f"**Eixo menor:** {eixo_menor:.4f}")
    st.write(f"**Razão entre eixos:** {razao:.4f}")
    st.write(f"**Ângulo da elipse (graus):** {angulo:.2f}°")
    st.write(f"**S-index:** {s_index:.4f}")

    # Passo 1: Coordenadas originais
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Passo 1: Posições registradas no smartphone")
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(x, y, 'o-', alpha=0.6, label="Trajetória")
        ax1.set_title("Coordenadas originais (X, Y)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.axhline(0, color='gray', lw=1)
        ax1.axvline(0, color='gray', lw=1)
        ax1.set_xlim(0, 1440)
        ax1.set_ylim(0, 2730)
        ax1.set_aspect('equal')
        ax1.legend()
        st.pyplot(fig1)
    with col2:
        # Passo 2: Vetores e elipse final
        st.subheader("📈 Passo 2: Vetores de deslocamento e elipse de inércia")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.scatter(dx, dy, alpha=0.6, label="Vetores de deslocamento")

        # 🔧 Plotando setas (quiver) no gráfico de vetores acumulados
        ax2.quiver(np.zeros_like(dx), np.zeros_like(dy), dx, dy,
                   angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5)

        width, height = 2 * np.sqrt(eigvals)
        ellipse = Ellipse((0, 0), width, height, angle=angulo,
                          edgecolor='red', fc='None', lw=2, label='Elipse de inércia')
        ax2.add_patch(ellipse)

        #ax2.axhline(0, color='gray', lw=1)
        #ax2.axvline(0, color='gray', lw=1)
        ax2.axhline(0, color='black', lw=1)
        ax2.axvline(0, color='black', lw=1)
        
        ax2.set_aspect('equal')
        ax2.legend()
        ax2.set_title("Distribuição Vetorial com Setas e Elipse de Inércia")
        ax2.set_xlabel("\u0394X")
        ax2.set_ylabel("\u0394Y")
        st.pyplot(fig2)

    # Passo 3: Animação frame a frame
    st.subheader("🎮 Passo 3: Animação frame a frame")
    max_frame = len(dx)
    frame = st.slider(
        "Deslize para visualizar a trajetória e vetores acumulados:", 0, max_frame, 1)

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: trajetória acumulada com vetores desenhados
    ax3a.plot(x[:frame], y[:frame], '-', color='gray')
    # 🔧 Ajuste no tamanho das setas para visualização adequada
    for i in range(frame):
        ax3a.arrow(x[i], y[i], dx[i], dy[i], head_width=2,
                   head_length=2, fc='gray', ec='gray')

    ax3a.arrow(x[frame], y[frame], dx[frame], dy[frame], head_width=2,
               head_length=2, fc='blue', ec='blue')
    ax3a.set_title("Trajetória acumulada com vetores")
    ax3a.set_xlabel("X")
    ax3a.set_ylabel("Y")
    ax3a.set_xlim(min(x), max(x))
    ax3a.set_ylim(min(y), max(y))

    # ax3a.set_aspect('equal')

    # Subplot 2: vetores acumulados com elipse parcial
    # ax3b.scatter(dx[:frame+1], dy[:frame+1], color='black',
    #             alpha=0.6, label="Vetores \u0394X, \u0394Y")
    # 🔧 Plotando setas no subplot de vetores acumulados
    ax3b.quiver(np.zeros_like(dx[:frame+1]), np.zeros_like(dy[:frame+1]), dx[:frame+1],
                dy[:frame+1], angles='xy', scale_units='xy', scale=1, color='black', alpha=0.5)
    ax3b.quiver(np.zeros_like(dx[frame:frame+1]), np.zeros_like(dy[frame:frame+1]), dx[frame:frame+1],
                dy[frame:frame+1], angles='xy', scale_units='xy', scale=1, color='black', alpha=0.5)

    if frame > 2:
        eixo_maior_f, eixo_menor_f, razao_f, angulo_f, s_index_f, eigvecs_f, eigvals_f = calcular_elipse_inercia(
            dx[:frame], dy[:frame])
        width_f, height_f = 2 * np.sqrt(eigvals_f)
        ellipse_f = Ellipse((0, 0), width_f, height_f,
                            angle=angulo_f, edgecolor='red', fc='None', lw=2)
        ax3b.add_patch(ellipse_f)
        ax3b.set_title(
            f"Vetores até o frame {frame}\nS-index parcial: {s_index_f:.2f}")
    else:
        ax3b.set_title("Vetores acumulados")

    ax3b.axhline(0, color='black', lw=1)
    ax3b.axvline(0, color='black', lw=1)
    ax3b.set_xlabel("\u0394X")
    ax3b.set_ylabel("\u0394Y")
    ax3b.set_xlim(-300, 300)
    ax3b.set_ylim(-300, 300)
    # ax3b.set_aspect('equal')
    ax3b.legend()

    st.pyplot(fig3)
else:
    st.info("Aguardando upload de arquivo com colunas: tempo, X, Y...")





