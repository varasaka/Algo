# Run Algo Agent on a server/VPS with a static IP (needed for Zerodha API orders).
#   docker build -t algo-agent .
#   docker run -d --restart unless-stopped -p 8000:8000 -v algo-data:/data \
#       -e ALGO_AGENT_PIN=482193 --name algo-agent algo-agent
# Put HTTPS in front (e.g. Caddy) before exposing it to the internet.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY algo_agent ./algo_agent
RUN pip install --no-cache-dir .
ENV ALGO_AGENT_HOME=/data TZ=Asia/Kolkata
VOLUME /data
EXPOSE 8000
CMD ["sh", "-c", "python -m algo_agent app --host 0.0.0.0 --port 8000 --no-browser ${ALGO_AGENT_PIN:+--pin $ALGO_AGENT_PIN}"]
