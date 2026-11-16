"""Demo settings for the showcase (production values are configured per deployment)."""

# Rolling window of history kept per interval in the feature cache: "<n>M" months, "<n>Y" years, or "ALL".
FEATURE_CACHE_WINDOWS = {"1m": "3M", "5m": "6M", "30m": "1Y", "60m": "2Y", "1d": "ALL"}


def parse_cache_window(interval: str) -> int | None:
    """Return the number of trading days to keep for ``interval``, or None for full history."""
    window = FEATURE_CACHE_WINDOWS.get(interval, "ALL")
    if window == "ALL":
        return None
    if window.endswith("M"):
        return int(window[:-1]) * 21  # ~21 trading days per month
    if window.endswith("Y"):
        return int(window[:-1]) * 252  # ~252 trading days per year
    raise ValueError(f"Unsupported cache window: {window!r}")
