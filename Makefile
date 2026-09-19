# Force cmd.exe regardless of which shell `make` itself was launched from (PowerShell,
# cmd, or Git Bash) — recipes below use backslash paths, which only cmd.exe accepts as
# an executable path (Git Bash's sh strips them, cmd.exe rejects forward slashes there).
SHELL := cmd.exe
.SHELLFLAGS := /c

.PHONY: dev backend frontend install

dev:
	$(MAKE) -j2 backend frontend

backend:
	cd backend && .venv\Scripts\python.exe -m scripts.run_server

frontend:
	cd frontend && npm run tauri dev

install:
	cd backend && python -m venv .venv && .venv\Scripts\pip.exe install -e ".[dev]"
	cd frontend && npm install
