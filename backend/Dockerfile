# 1. Gunakan base image resmi Python
FROM python:3.11-slim

# 2. Tentukan direktori kerja di dalam kontainer
WORKDIR /app

# 3. Instal dependensi sistem yang mungkin dibutuhkan oleh XGBoost/Scikit-Learn
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Salin file requirements.txt terlebih dahulu (untuk caching layer pip)
COPY requirements.txt .

# 5. Instal semua pustaka Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Salin seluruh kode backend dan folder models ke dalam kontainer
COPY . .

# 7. Tentukan port yang digunakan oleh FastAPI
EXPOSE 8000

# 8. Perintah untuk menjalankan server FastAPI saat kontainer menyala
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]