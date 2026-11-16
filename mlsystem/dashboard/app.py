"""Flask application factory for the demo dashboard."""
from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from . import demo_data


def create_app() -> Flask:
    app = Flask(__name__)

    def cursor_arg() -> int | None:
        return request.args.get("cursor", type=int)

    @app.get("/")
    def index():
        return render_template("index.html", page="dashboard", snapshot=demo_data.snapshot(cursor_arg()))

    @app.get("/model-health")
    def model_health():
        return render_template("model_health.html", page="health", snapshot=demo_data.snapshot(cursor_arg()))

    @app.get("/api/snapshot")
    def api_snapshot():
        response = jsonify(demo_data.snapshot(cursor_arg()))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok", synthetic=True)

    return app
