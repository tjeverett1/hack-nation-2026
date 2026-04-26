"""SafeMD.ai - portable RAG pipeline (runs outside Databricks workspaces)."""

from .safemd_pipeline import SafeMDPipeline, load_incidents

__all__ = ["SafeMDPipeline", "load_incidents"]
