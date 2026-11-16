"""
Demo dashboard: the page layout of the system's web UI, fed with synthetic data.

    python -m mlsystem.dashboard          # then open http://127.0.0.1:5057

The production dashboard is a large Flask application over live predictions and
trades. This is a small stand-in with the same shape - header and status chip,
indicator cards, a filterable prediction chart and table per target, and a model
health page - where every number comes from the seeded synthetic pipeline in
``mlsystem.quality.examples.pipeline_demo``. Nothing here reads real data.
"""
from .app import create_app

__all__ = ["create_app"]
