FROM python:3.12-slim

# Install system dependencies for Prisma, Python, and Supervisor
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libssl-dev \
    nodejs \
    npm \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Poetry
RUN pip install --upgrade pip && pip install poetry

# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app/src

# Copy dependency files first for layer caching
COPY pyproject.toml poetry.lock ./

# Install Python dependencies without dev
RUN poetry install --no-root --without dev

# Copy app source files
COPY config /app/config
COPY src /app/src
COPY prisma /app/prisma

# Generate Prisma client
RUN poetry run prisma generate --schema=/app/prisma/schema.prisma

# default cmd
CMD ["poetry", "run", "python", "src/main.py"]
