.PHONY: dev backend frontend install

dev:
	$(MAKE) -j2 backend frontend

backend:
	cd backend && .venv/Scripts/python.exe scripts/run_server.py

frontend:
	cd frontend && npm run tauri dev

install:
	cd backend && python -m venv .venv && .venv/Scripts/pip.exe install -e ".[dev]"
	cd frontend && npm install
