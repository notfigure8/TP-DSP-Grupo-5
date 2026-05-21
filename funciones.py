import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# =============================================================================
# PARTE 1 - HERRAMIENTAS DE ANÁLISIS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. GRAFICACIÓN
# -----------------------------------------------------------------------------

def graficar_senal(x, fs=1, titulo="Señal temporal", xlabel="Tiempo [s]"):
    """
    Grafica una o varias señales en el tiempo.
    
    Parámetros:
        x   : array o lista de arrays con las señales
        fs  : frecuencia de muestreo (Hz)
        titulo: título del gráfico
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if isinstance(x, np.ndarray) and x.ndim == 1:
        x = [x]
    
    for senal in x:
        t = np.arange(len(senal)) / fs
        ax.plot(t, senal)
    
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def graficar_espectro(x, fs=1, titulo="Espectro de Fourier"):
    """
    Grafica el módulo del espectro de Fourier de una o varias señales.
    
    Parámetros:
        x  : array o lista de arrays con las señales
        fs : frecuencia de muestreo (Hz)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if isinstance(x, np.ndarray) and x.ndim == 1:
        x = [x]
    
    for senal in x:
        N = len(senal)
        X = np.fft.fft(senal)
        freqs = np.fft.fftfreq(N, d=1/fs)
        # Solo frecuencias positivas
        mitad = N // 2
        ax.plot(freqs[:mitad], np.abs(X[:mitad]))
    
    ax.set_title(titulo)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("|X(ω)|")
    ax.grid(True)
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


def graficar_H(H, freqs, titulo="Respuesta en frecuencia"):
    """
    Grafica módulo y fase de H(ω).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    mitad = len(freqs) // 2
    
    ax1.plot(freqs[:mitad], np.abs(H[:mitad]))
    ax1.set_title(titulo + " - Módulo")
    ax1.set_xlabel("Frecuencia normalizada")
    ax1.set_ylabel("|H(ω)|")
    ax1.grid(True)
    
    ax2.plot(freqs[:mitad], np.angle(H[:mitad]))
    ax2.set_title(titulo + " - Fase")
    ax2.set_xlabel("Frecuencia normalizada")
    ax2.set_ylabel("∠H(ω) [rad]")
    ax2.grid(True)
    
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
# 6. CARGAR ARCHIVOS
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
    from scipy.io import wavfile
    
    fs, datos = wavfile.read(ruta_archivo)
    
    if normalizar:
        # Usar el valor máximo del tipo de dato
        tipo = datos.dtype
        if np.issubdtype(tipo, np.integer):
            # Obtener el máximo valor para ese tipo entero
            max_valor = np.iinfo(tipo).max
            datos = datos.astype(np.float32) / max_valor
    
    return fs, datos