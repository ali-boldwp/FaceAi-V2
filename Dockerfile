# syntax=docker/dockerfile:1
FROM node:20-alpine AS frontend-build
WORKDIR /frontend

ARG VITE_API_URL=.
ENV VITE_API_URL=$VITE_API_URL

COPY faceai/frontend/package*.json ./
RUN npm ci
COPY faceai/frontend/ ./
RUN npm run build

FROM python:3.11-slim AS app
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 nginx supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY faceai/backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY faceai/backend /app/backend
COPY --from=frontend-build /frontend/dist /usr/share/nginx/html

RUN set -eux; \
    rm -f /etc/nginx/sites-enabled/default; \
    cat > /etc/nginx/conf.d/default.conf <<'EOF'
server {
  listen 80;
  server_name _;
  root /usr/share/nginx/html;

  location /api/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }

  location / {
    try_files $uri /index.html;
  }
}
EOF

RUN set -eux; \
    cat > /etc/supervisor/supervisord.conf <<'EOF'
[supervisord]
nodaemon=true

[program:uvicorn]
command=uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir /app/backend
directory=/app/backend
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:nginx]
command=nginx -g "daemon off;"
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
EOF

EXPOSE 80
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
