# Deploying AllianceAI to an Alpine server (Docker + JupyterLab)

This walks you from a bare Alpine box to a running AllianceAI you drive from a
Jupyter notebook in your browser.

> **Why a Debian image on an Alpine host?** The container's base image is
> independent of the host OS. The scientific Python stack (numpy/scipy/prophet)
> only ships prebuilt wheels for glibc, not Alpine's musl libc. Building on
> Alpine would compile everything from source (slow, breakage-prone). The
> `Dockerfile` uses `python:3.12-slim` (Debian) which runs fine on your Alpine
> Docker host. **You do not install Python or any of these libraries on the
> server — only Docker.**

---

## 1. Install Docker on the Alpine server

SSH into the server, then:

```sh
# Enable the community repo if needed, then install Docker + compose plugin
apk update
apk add docker docker-cli-compose

# Start Docker now and on every boot
rc-update add docker default
service docker start

# (optional) run docker as a non-root user
addgroup youruser docker

# Verify
docker version
docker compose version
```

---

## 2. Get the project onto the server

Pick whichever you have:

**Option A — git (if you've pushed it to a repo):**
```sh
git clone <your-repo-url> allianceai
cd allianceai
```

**Option B — copy from your machine** (run this on your *laptop*, not the server):
```sh
# From the project folder on Windows (PowerShell) or any machine with scp:
scp -r "C:\Users\arman\Desktop\software projects\AllianceAI" youruser@SERVER_IP:~/allianceai
```
The `.dockerignore` keeps the bulky local venv and caches out of the image, but
`scp -r` copies everything — for a lean transfer, zip without `alliance-venv/`
first, or use git.

---

## 3. Configure secrets (`.env`)

On the server, in the project directory, make sure `.env` exists with your keys.
**Never bake secrets into the image** — Compose injects `.env` at runtime.

```sh
cat > .env <<'EOF'
LLM_PROVIDER=auto
OPENROUTER_API_KEY=sk-or-v1-...your key...
OPENROUTER_MODEL=anthropic/claude-haiku-4.5
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EDGAR_USER_AGENT=AllianceAI research (contact: you@example.com)

# Choose a real login token for JupyterLab (don't leave the default):
JUPYTER_TOKEN=pick-a-long-random-string
EOF
chmod 600 .env
```

---

## 4. Build and start

```sh
docker compose up -d --build
```

First build takes a few minutes (it downloads the scientific wheels + Prophet).
Subsequent starts are instant. Check it's healthy:

```sh
docker compose ps
docker compose logs -f          # Ctrl-C to stop following
```

The DuckDB learning store, your notebooks, and generated reports persist in a
named volume + bind mounts, so **the AI keeps its learned calibration across
restarts and rebuilds.**

---

## 5. Open JupyterLab from your browser

JupyterLab listens on the server's port `8888`. **Don't expose it raw to the
internet** — tunnel it over SSH from your laptop:

```sh
# Run on your LAPTOP. Leaves a secure tunnel open; keep this terminal running.
ssh -N -L 8888:localhost:8888 youruser@SERVER_IP
```

Then open in your browser:

```
http://localhost:8888/lab
```

Enter the `JUPYTER_TOKEN` you set in `.env`. Open
`notebooks/AllianceAI_demo.ipynb` and run the cells — it analyzes a ticker,
shows the decision/confidence/highlights, runs a backtest, and renders the
interactive HTML report inline.

---

## 6. Everyday commands

```sh
docker compose logs -f                       # watch output
docker compose restart                       # restart the service
docker compose down                          # stop (data volume is kept)
docker compose up -d --build                 # rebuild after code changes

# Run a one-off analysis from the CLI (no notebook):
docker compose exec allianceai allianceai AAPL --output-dir /app/reports

# Seed/refresh the learning store via backtest:
docker compose exec allianceai allianceai AAPL MSFT --backtest --bt-max-steps 12
```

Reports land in `./reports/` on the host (bind-mounted), so you can also serve
or download them directly.

---

## 7. Security checklist

- **Keep `8888` off the public internet.** Use the SSH tunnel (step 5), or put
  it behind a reverse proxy with TLS + auth (Caddy/Traefik/nginx) if you need
  remote access without a tunnel.
- **Set a strong `JUPYTER_TOKEN`** — anyone with it gets a Python shell in the
  container.
- **`.env` holds live API keys** — `chmod 600`, never commit it. The
  `.dockerignore` and (recommended) a `.gitignore` entry keep it out of images
  and repos.
- The container runs as root inside its own namespace; that's normal for a
  single-tenant box. For multi-tenant hosts, add a non-root `USER` to the
  Dockerfile.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | `service docker start` (and `rc-update add docker default`) |
| Build fails on `prophet` wheel | Remove `prophet` from the Dockerfile's pip line — the code auto-falls back to Holt-Winters |
| Jupyter asks for a password/token | It's the `JUPYTER_TOKEN` from `.env` |
| Report shows blank charts | The report loads plotly.js from a CDN — your browser needs internet (the server doesn't) |
| Highlights are rule-based, not LLM | Check `OPENROUTER_API_KEY` (or `ANTHROPIC_API_KEY`) is set in `.env` and the container was restarted after editing it |
