import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as signal
import soundfile as sf

# =============================================================================
# PARTE 1 - HERRAMIENTAS DE ANÁLISIS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. GRÁFICO
# -----------------------------------------------------------------------------

def graficar_senales(x, fs=1, titulo=None, xlabel="Tiempo [s]", xlim=None, ylim=None, subplots=False, stem=False, ncols=1, titles=None):
    """
    Grafica una o varias señales en el tiempo.

    Parámetros:
        x        : array o lista de arrays con las señales
        fs       : frecuencia de muestreo (Hz)
        titulo   : título del gráfico
        xlim     : tupla (min, max) para limitar el eje X, ej: (0, 0.5)
        ylim     : tupla (min, max) para limitar el eje Y
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
            if ylim is not None:
                axes[i].set_ylim(ylim)
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
        if ylim is not None:
            ax.set_ylim(ylim)
        if n >= 2:
            ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def graficar_espectros(x, fs=1, titulo="Respuesta en Frecuencia", xlim=None, ylim=None, subplots=False, titles=None):
    """
    Grafica el módulo del espectro de Fourier de una o varias señales.

    Parámetros:
        x        : array o lista de arrays con las señales
        fs       : frecuencia de muestreo (Hz)
        xlim     : tupla (min, max) para limitar el eje X, ej: (0, 1000)
        ylim     : tupla (min, max) para limitar el eje Y
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
            if ylim is not None:
                ax.set_ylim(ylim)
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
        if ylim is not None:
            ax.set_ylim(ylim)
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


def graficar_H(H, freqs, titulo="Respuesta en frecuencia", xlim=None, ylim=None, subplots=False, titles=None):
    """
    Grafica módulo y fase de H(ω).

    Parámetros:
        H        : array complejo o lista de arrays con la respuesta en frecuencia
        freqs    : array o lista de arrays con las frecuencias correspondientes
        xlim     : tupla (min, max) para limitar el eje X en ambos subplots
        ylim     : tupla (min, max) para limitar el eje Y del módulo
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
            if ylim is not None:
                axes[i, 0].set_ylim(ylim)

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
        if ylim is not None:
            ax1.set_ylim(ylim)
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
        senal con ruido agregado, ruido
    """
    ruido = amplitud * np.random.randn(len(senal))
    return senal + ruido, ruido

def generar_melodia(notas, duraciones, fs, amplitudes=None):
    """
    Genera una melodía con notas que se tocan una tras otra.

    Parámetros:
        notas      : lista de frecuencias [Hz]
        duraciones : lista de duraciones para cada nota [s]
        fs         : frecuencia de muestreo [Hz]
        amplitudes : lista de amplitudes (opcional, default 1.0)

    Retorna:
        senal : array con la melodía completa
    """
    if amplitudes is None:
        amplitudes = [1.0] * len(notas)

    senal = []
    for freq, dur, amp in zip(notas, duraciones, amplitudes):
        t = np.arange(0, dur, 1/fs)
        nota = amp * np.sin(2 * np.pi * freq * t)

        # Pequeño fade al inicio y final de cada nota (evita clics)
        fade_samples = int(0.01 * fs)
        if len(nota) > 2 * fade_samples:
            nota[:fade_samples] *= np.linspace(0, 1, fade_samples)
            nota[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        senal.extend(nota)

    return np.array(senal)

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
# 7. CARGAR ARCHIVOS Y GENERAR
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

def generate_wav(señal, titulo, fs):

    señal = np.array(señal, dtype=np.float64)

    # evitar clipping solamente
    señal = np.clip(señal, -1, 1)

    sf.write(titulo, señal, fs)
# -----------------------------------------------------------------------------
# 8. CÁLCULO DE DENSIDADES ESPECTRALES Y COHERENCIA
# -----------------------------------------------------------------------------

def analizar_coherencia_sistema(x, y, fs, nperseg=1024):
    """
    Calcula las densidades espectrales y la coherencia cuadrática entre
    una señal de entrada (x) y salida (y) usando el método de Welch.

    Parámetros:
        x       : señal de entrada x[n]
        y       : señal de salida y[n]
        fs      : frecuencia de muestreo [Hz]
        nperseg : tamaño de la ventana para el método de Welch (default 1024)

    Retorna:
        freqs      : array de frecuencias correspondientes [Hz]
        Gxx        : Densidad espectral de potencia de x
        Gyy        : Densidad espectral de potencia de y
        Gxy        : Densidad espectral de potencia cruzada
        coherencia : Coherencia cuadrática γ²(ω)
    """
    # 1. Auto-espectros (Gxx y Gyy)
    # Por defecto window='hann' y noverlap=nperseg//2
    freqs, Gxx = signal.welch(x, fs=fs, nperseg=nperseg)
    _, Gyy = signal.welch(y, fs=fs, nperseg=nperseg)

    # 2. Espectro Cruzado (Gxy)
    _, Gxy = signal.csd(x, y, fs=fs, nperseg=nperseg)

    # 3. Coherencia Cuadrática (Usando la fórmula de la cátedra)
    # np.real() es solo por seguridad, la fórmula ya asegura un resultado real
    coherencia = np.real((np.abs(Gxy)**2) / (Gxx * Gyy))

    # Alternativa directa (hace exactamente el mismo cálculo interno):
    # freqs, coherencia = signal.coherence(x, y, fs=fs, nperseg=nperseg)

    return freqs, Gxx, Gyy, Gxy, coherencia


def graficar_identificacion_sistema(x, y, fs, nperseg=1024, titulo="Análisis del Sistema"):
    """
    Calcula y grafica la Respuesta en Frecuencia empírica H(w) y
    la Coherencia cuadrática de un sistema desconocido.

    Parámetros:
        x       : señal de entrada x[n]
        y       : señal de salida y[n]
        fs      : frecuencia de muestreo [Hz]
        nperseg : tamaño de la ventana de Welch
        titulo  : título general del gráfico
    """
    # Obtener densidades
    freqs, Gxx, Gyy, Gxy, coherencia = analizar_coherencia_sistema(x, y, fs, nperseg)

    # Estimador H1 de la respuesta en frecuencia (mitiga ruido a la salida)
    # Evitar división por cero
    H = np.zeros_like(Gxy, dtype=complex)
    mascara = Gxx > 1e-15
    H[mascara] = Gxy[mascara] / Gxx[mascara]

    # Preparar el gráfico
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(titulo)

    # Subplot 1: Magnitud en dB
    H_mag_db = 20 * np.log10(np.abs(H) + 1e-10) # 1e-10 evita log(0)
    ax1.plot(freqs, H_mag_db, color='blue')
    ax1.set_title("Módulo de H(ω)")
    ax1.set_ylabel("Magnitud [dB]")
    ax1.grid(True)

    # Subplot 2: Fase
    H_fase = np.angle(H)
    ax2.plot(freqs, H_fase, color='orange')
    ax2.set_title("Fase de H(ω)")
    ax2.set_ylabel("Fase [rad]")
    ax2.grid(True)

    # Subplot 3: Coherencia
    ax3.plot(freqs, coherencia, color='green')
    ax3.set_title(r"Coherencia Cuadrática $\gamma_{xy}^2(\omega)$")
    ax3.set_xlabel("Frecuencia [Hz]")
    ax3.set_ylabel("Coherencia [0 - 1]")
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
