FROM python:3.8-slim

WORKDIR /app

# Cài đặt các thư viện cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép code
COPY . .

# Expose port cho MetaTrader 5 API
EXPOSE 5000

# Command để chạy chiến lược
CMD ["python", "rsi_trading_strategy.py"]
