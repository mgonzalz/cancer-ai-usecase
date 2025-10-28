# ======= Base ligera Python =======
FROM python:3.10-slim

# Evita prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# ======= Dependencias del sistema para OpenCV =======
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ======= Directorio de la app =======
WORKDIR /app
COPY . /app

# ======= Python deps =======
RUN pip install --no-cache-dir -r requirements.txt

# ======= Ajustes de runtime =======
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV HF_HOME=/root/.cache/huggingface

# (Opcional) variables por defecto para el modelo
ENV HF_MODEL_REPO=mgonzalz/skinbridge-cnn
ENV HF_MODEL_FILENAME=model.keras

# ======= Exponer puerto del Space =======
EXPOSE 7860

# ======= Arranque (modo claro forzado) =======
CMD ["streamlit", "run", "streamlit_app/App.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--theme.base=light", "--server.enableXsrfProtection=false"]
