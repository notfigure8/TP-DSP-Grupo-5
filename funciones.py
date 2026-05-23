import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile

# =============================================================================
# PARTE 1 - HERRAMIENTAS DE ANÁLISIS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. GRÁFICO
# -----------------------------------------------------------------------------

def graficar_senales(x, fs=1, titulo=None, xlabel="Tiempo [s]", xlim=None, subplots=False, stem=False, ncols=1, titles=None):
    """
    Grafica una o varias señales en el tiempo.

    Parámetros:
        x        : array o lista de arrays con las señales
        fs       : frecuencia de muestreo (Hz)
        titulo   : título del gráfico
        xlim     : tupla (min, max) para limitar el eje X, ej: (0, 0.5)
        subplots : si True, cada señal en su propio subplot
        stem     : si True, usa stem en lugar de plot (para señales discretas)
        ncols    : número de columnas cuando subplots=True (default 1)
        titles   : lista de strings con el título de cada subplot (opcional)
    """
    if isinstance(x, np.ndarray) and x.ndim == 1:
        x = [x]

    n = len(x)

    def _plot(ax, t, senal, label=None):
        if stem:
            ax.stem(t, senal, label=label)
        else:
            ax.plot(t, senal, label=label)

    if subplots and n >= 2:
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
        fig.suptitle(titulo)
        axes = np.array(axes).flatten()
        for i, senal in enumerate(x):
            t = np.arange(len(senal)) / fs
            _plot(axes[i], t, senal)
            subplot_title = titles[i] if titles is not None and i < len(titles) else f"Señal {i + 1}"
            axes[i].set_title(subplot_title)
            axes[i].set_ylabel("Amplitud")
            axes[i].set_xlabel(xlabel)
            axes[i].grid(True)
            if xlim is not None:
                axes[i].set_xlim(xlim)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, senal in enumerate(x):
            t = np.arange(len(senal)) / fs
            label = f"Señal {i + 1}" if n >= 2 else None
            _plot(ax, t, senal, label=label)
        ax.set_title(titulo)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        if xlim is not None:
            ax.set_xlim(xlim)
        if n >= 2:
            ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def graficar_espectros(x, fs=1, titulo="Respuesta en Frecuencia", xlim=None, subplots=False, titles=None):
    """
    Grafica el módulo del espectro de Fourier de una o varias señales.

    Parámetros:
        x        : array o lista de arrays con las señales
        fs       : frecuencia de muestreo (Hz)
        xlim     : tupla (min, max) para limitar el eje X, ej: (0, 1000)
        subplots : si True, cada señal en su propio subplot
        titles   : lista de strings con el título de cada subplot (opcional)
    """
    if isinstance(x, np.ndarray) and x.ndim == 1:
        x = [x]

    n = len(x)
    espectros = []
    freqs_list = []
    for senal in x:
        N = len(senal)
        X = np.fft.fft(senal)
        freqs = np.fft.fftfreq(N, d=1/fs)
        mitad = N // 2
        espectros.append(np.abs(X[:mitad]) / (N / 2))
        freqs_list.append(freqs[:mitad])

    if subplots and n >= 2:
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
        fig.suptitle(titulo)
        for i, (esp, freqs, ax) in enumerate(zip(espectros, freqs_list, axes)):
            ax.plot(freqs, esp)
            subplot_title = titles[i] if titles is not None and i < len(titles) else f"Señal {i + 1}"
            ax.set_title(subplot_title)
            ax.set_ylabel("Amplitud")
            ax.grid(True)
            if xlim is not None:
                ax.set_xlim(xlim)
        axes[-1].set_xlabel("Frecuencia [Hz]")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (esp, freqs) in enumerate(zip(espectros, freqs_list)):
            label = f"Señal {i + 1}" if n >= 2 else None
            ax.plot(freqs, esp, label=label)
        ax.set_title(titulo)
        ax.set_xlabel("Frecuencia [Hz]")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        if xlim is not None:
            ax.set_xlim(xlim)
        if n >= 2:
            ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 2. RESPUESTA EN FRECUENCIA DEL SISTEMA
# -----------------------------------------------------------------------------

def calcular_H(x, y):
    """
    Calcula la respuesta en frecuencia H(ω) = Y(ω) / X(ω).

    Parámetros:
        x : señal de entrada x[n]
        y : señal de salida y[n]

    Retorna:
        H     : respuesta en frecuencia (array complejo)
        freqs : frecuencias normalizadas correspondientes
    """
    X = np.fft.fft(x)
    Y = np.fft.fft(y)

    # Evitar división por cero
    H = np.where(np.abs(X) > 1e-10, Y / X, 0)
    freqs = np.fft.fftfreq(len(x))

    return H, freqs


def graficar_H(H, freqs, titulo="Respuesta en frecuencia", xlim=None, subplots=False, titles=None):
    """
    Grafica módulo y fase de H(ω).

    Parámetros:
        H        : array complejo o lista de arrays con la respuesta en frecuencia
        freqs    : array o lista de arrays con las frecuencias correspondientes
        xlim     : tupla (min, max) para limitar el eje X en ambos subplots
        subplots : si True, cada H en su propia fila de subplots (módulo | fase)
        titles   : lista de strings con el título de cada fila de subplots (opcional)
    """
    if isinstance(H, np.ndarray) and H.ndim == 1:
        H = [H]
        freqs = [freqs]

    n = len(H)

    if subplots and n >= 2:
        fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
        fig.suptitle(titulo)
        for i, (Hi, fi) in enumerate(zip(H, freqs)):
            mitad = len(fi) // 2
            row_title = titles[i] if titles is not None and i < len(titles) else f"H {i + 1}"
            axes[i, 0].plot(fi[:mitad], np.abs(Hi[:mitad]))
            axes[i, 0].set_title(f"{row_title} - Módulo")
            axes[i, 0].set_xlabel("Frecuencia normalizada")
            axes[i, 0].set_ylabel("|H(ω)|")
            axes[i, 0].grid(True)
            if xlim is not None:
                axes[i, 0].set_xlim(xlim)

            axes[i, 1].plot(fi[:mitad], np.angle(Hi[:mitad]))
            axes[i, 1].set_title(f"{row_title} - Fase")
            axes[i, 1].set_xlabel("Frecuencia normalizada")
            axes[i, 1].set_ylabel("∠H(ω) [rad]")
            axes[i, 1].grid(True)
            if xlim is not None:
                axes[i, 1].set_xlim(xlim)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        for i, (Hi, fi) in enumerate(zip(H, freqs)):
            mitad = len(fi) // 2
            label = f"H {i + 1}" if n >= 2 else None
            ax1.plot(fi[:mitad], np.abs(Hi[:mitad]), label=label)
            ax2.plot(fi[:mitad], np.angle(Hi[:mitad]), label=label)

        ax1.set_title(titulo + " - Módulo")
        ax1.set_xlabel("Frecuencia normalizada")
        ax1.set_ylabel("|H(ω)|")
        ax1.grid(True)
        if xlim is not None:
            ax1.set_xlim(xlim)
        if n >= 2:
            ax1.legend(loc="best")

        ax2.set_title(titulo + " - Fase")
        ax2.set_xlabel("Frecuencia normalizada")
        ax2.set_ylabel("∠H(ω) [rad]")
        ax2.grid(True)
        if xlim is not None:
            ax2.set_xlim(xlim)
        if n >= 2:
            ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 3. FILTROS - RESPUESTA AL IMPULSO
# -----------------------------------------------------------------------------

def filtro_media_movil(M):
    """
    Genera la respuesta al impulso del filtro media móvil.
    h[n] = 1/M para n = 0, 1, ..., M-1

    Parámetros:
        M : largo de la ventana

    Retorna:
        h : respuesta al impulso (array de longitud M)
    """
    return np.ones(M) / M


def filtro_peine(b0, b1, b2):
    """
    Genera la respuesta al impulso del filtro peine.
    h[n] = b0*delta[n] + b1*delta[n-1] + b2*delta[n-2]

    Parámetros:
        b0, b1, b2 : coeficientes del filtro

    Retorna:
        h : respuesta al impulso (array de longitud 3)
    """
    return np.array([b0, b1, b2])


# -----------------------------------------------------------------------------
# 4. FILTRADO DE SEÑALES
# -----------------------------------------------------------------------------

def filtrar_convolucion(x, h):
    """
    Filtra la señal x con el filtro h mediante convolución en el tiempo.

    Parámetros:
        x : señal de entrada
        h : respuesta al impulso del filtro

    Retorna:
        y : señal filtrada
    """
    return np.convolve(x, h)


def filtrar_frecuencia(x, h):
    """
    Filtra la señal x con el filtro h mediante convolución circular en frecuencia.
    Usa zero-padding para evitar aliasing circular (equivalente a convolución lineal).

    Parámetros:
        x : señal de entrada
        h : respuesta al impulso del filtro

    Retorna:
        y : señal filtrada
    """
    N = len(x) + len(h) - 1  # longitud de la convolución lineal
    X = np.fft.fft(x, n=N)
    H = np.fft.fft(h, n=N)
    Y = X * H
    return np.real(np.fft.ifft(Y))


# -----------------------------------------------------------------------------
# 5. GENERACIÓN DE SEÑALES DE PRUEBA
# -----------------------------------------------------------------------------

def generar_suma_tonos(frecuencias, amplitudes, fs, duracion):
    """
    Genera una señal como suma de tonos puros.

    Parámetros:
        frecuencias : lista de frecuencias [Hz]
        amplitudes  : lista de amplitudes correspondientes
        fs          : frecuencia de muestreo [Hz]
        duracion    : duración de la señal [s]

    Retorna:
        senal : array con la señal generada
    """
    t = np.arange(0, duracion, 1/fs)
    senal = np.zeros(len(t))
    for f, a in zip(frecuencias, amplitudes):
        senal += a * np.sin(2 * np.pi * f * t)
    return senal


def agregar_ruido_blanco(senal, amplitud):
    """
    Agrega ruido blanco gaussiano a una señal.

    Parámetros:
        senal    : señal original
        amplitud : amplitud del ruido

    Retorna:
        senal con ruido agregado
    """
    ruido = amplitud * np.random.randn(len(senal))
    return senal + ruido


# -----------------------------------------------------------------------------
# 6. TRUNCADO DE FILTRO FIR
# -----------------------------------------------------------------------------

def truncar_fir(h, M):
    """
    Trunca un filtro FIR conservando M coeficientes
    centrados respecto del impulso original.

    Parámetros:
        h : respuesta al impulso original
        M : cantidad de coeficientes

    Retorna:
        h_truncado
    """
    N = len(h)
    inicio = (N - M) // 2
    fin = inicio + M
    return h[inicio:fin]


# -----------------------------------------------------------------------------
# 7. CARGAR ARCHIVOS
# -----------------------------------------------------------------------------

def cargar_filtro_fir(ruta_archivo):
    """
    Carga coeficientes desde archivo .npy y devuelve el filtro FIR.

    Parámetros:
        ruta_archivo : string con la ruta al archivo .npy

    Retorna:
        h : respuesta al impulso (array)
    """
    coeficientes = np.load(ruta_archivo)
    return coeficientes


def cargar_wav(ruta_archivo, normalizar=True):
    """
    Carga un archivo WAV y opcionalmente normaliza a float [-1, 1].

    Parámetros:
        ruta_archivo : string con el nombre del archivo .wav
        normalizar   : bool, si True convierte a float32 en rango [-1, 1]

    Retorna:
        fs    : frecuencia de muestreo (Hz)
        datos : array con la señal
    """


    fs, datos = wavfile.read(ruta_archivo)

    if normalizar:
        tipo = datos.dtype
        if np.issubdtype(tipo, np.integer):
            max_valor = np.iinfo(tipo).max
            datos = datos.astype(np.float32) / max_valor

    return fs, datos
