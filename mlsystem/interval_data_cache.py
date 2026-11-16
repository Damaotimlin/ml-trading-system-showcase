"""
Interval Data Cache - In-memory caching for SQLite data

This module provides a thread-safe, process-wide cache for interval data to reduce
SQLite contention when multiple workers (GPU, CPU, PREDICT) access the same intervals.

Key features:
- LRU cache with configurable size limits
- Thread-safe access with fine-grained locking per interval
- Automatic cache invalidation on market open/close
- Memory-efficient storage with compression support
- Cache warming for frequently-used intervals

Design goals:
1. Eliminate SQLite lock waits when multiple PREDICT workers use same interval
2. Allow GPU, CPU, and PREDICT workers to run in parallel without DB contention
3. Reduce disk I/O by keeping hot data in RAM
4. Maintain data freshness during market hours
"""

import os
import time
import threading
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple, List, Callable
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    df: pd.DataFrame
    interval: str
    ticker: str
    cached_at: float  # Unix timestamp
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0

    def __post_init__(self):
        if self.size_bytes == 0:
            # Estimate memory usage
            self.size_bytes = self.df.memory_usage(deep=True).sum()
        self.last_access = time.time()


class IntervalDataCache:
    """
    Thread-safe LRU cache for interval data

    Usage:
        cache = IntervalDataCache(max_entries=100, max_memory_mb=500)

        # Try to get from cache first
        df = cache.get(ticker, interval)
        if df is None:
            # Cache miss - load from DB
            df = load_raw_df(ticker, interval)
            cache.put(ticker, interval, df)

        # Use cached data
        process(df)
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_memory_mb: float = 500,
        ttl_seconds: float = 3600,  # 1 hour default TTL (when adaptive disabled)
        enable_compression: bool = False,
        adaptive_ttl: bool = True,  # Enable adaptive TTL based on market state
    ):
        """
        Initialize cache

        Args:
            max_entries: Maximum number of cached intervals
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for cache entries (seconds) - used when adaptive_ttl=False
            enable_compression: Enable DataFrame compression (trades CPU for memory)
            adaptive_ttl: If True, adjust TTL based on market state (5min during market, 24h after close)
        """
        self.max_entries = max_entries
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.base_ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression
        self.adaptive_ttl = adaptive_ttl

        # Cache storage: (ticker, interval) -> CacheEntry
        self._cache: OrderedDict[Tuple[str, str], CacheEntry] = OrderedDict()

        # Fine-grained locks per interval to allow parallel access
        self._interval_locks: Dict[Tuple[str, str], threading.RLock] = {}

        # Global cache lock (only for cache structure modifications)
        self._cache_lock = threading.RLock()

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0,
            'total_bytes': 0,
        }
        self.stats_lock = threading.Lock()

        # Market state tracking (for cache invalidation and adaptive TTL)
        self._market_open_time: Optional[float] = None
        self._last_invalidation: Optional[float] = None
        self._market_is_open: bool = False  # Track if market is currently open

        # Adaptive TTL settings (configurable via environment)
        self._ttl_market_open = float(os.getenv("INTERVAL_CACHE_TTL_MARKET_OPEN", "300"))  # 5 min
        self._ttl_market_closed = float(os.getenv("INTERVAL_CACHE_TTL_MARKET_CLOSED", "86400"))  # 24 hours

    def _get_current_ttl(self) -> float:
        """
        Get current TTL based on market state (if adaptive enabled)

        Returns:
            TTL in seconds based on current market state
        """
        if not self.adaptive_ttl:
            return self.base_ttl_seconds

        # Use shorter TTL during market hours (data changes frequently)
        # Use longer TTL after market close (data is stable)
        if self._market_is_open:
            return self._ttl_market_open
        else:
            return self._ttl_market_closed

    def _get_interval_lock(self, ticker: str, interval: str) -> threading.RLock:
        """Get or create lock for specific interval"""
        key = (ticker, interval)
        with self._cache_lock:
            if key not in self._interval_locks:
                self._interval_locks[key] = threading.RLock()
            return self._interval_locks[key]

    def get(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Get cached data for interval

        Returns:
            DataFrame if found and valid, None if cache miss
        """
        key = (ticker, interval)

        # Fast path: check if key exists without lock
        with self._cache_lock:
            if key not in self._cache:
                with self.stats_lock:
                    self.stats['misses'] += 1
                return None

        # Slow path: acquire interval-specific lock
        lock = self._get_interval_lock(ticker, interval)
        with lock:
            with self._cache_lock:
                entry = self._cache.get(key)
                if entry is None:
                    with self.stats_lock:
                        self.stats['misses'] += 1
                    return None

                # Check TTL (use adaptive TTL based on market state)
                now = time.time()
                current_ttl = self._get_current_ttl()
                if now - entry.cached_at > current_ttl:
                    # Expired - remove and return miss
                    del self._cache[key]
                    with self.stats_lock:
                        self.stats['misses'] += 1
                        self.stats['evictions'] += 1
                        self.stats['total_bytes'] -= entry.size_bytes
                    return None

                # Valid hit - update access stats and move to end (LRU)
                entry.access_count += 1
                entry.last_access = now
                self._cache.move_to_end(key)

                with self.stats_lock:
                    self.stats['hits'] += 1

                return entry.df.copy()  # Return copy to prevent mutations

    def put(
        self,
        ticker: str,
        interval: str,
        df: pd.DataFrame,
        force: bool = False
    ) -> bool:
        """
        Store data in cache

        Args:
            ticker: Ticker symbol
            interval: Interval string
            df: DataFrame to cache
            force: If True, bypass size limits

        Returns:
            True if cached successfully, False if rejected
        """
        if df is None or df.empty:
            return False

        key = (ticker, interval)
        lock = self._get_interval_lock(ticker, interval)

        with lock:
            # Estimate entry size
            df_copy = df.copy()
            entry_size = df_copy.memory_usage(deep=True).sum()

            # Check if we need to evict
            with self._cache_lock:
                # Remove old entry if exists
                if key in self._cache:
                    old_entry = self._cache[key]
                    with self.stats_lock:
                        self.stats['total_bytes'] -= old_entry.size_bytes

                # Check size limits (unless force=True)
                if not force:
                    # Evict old entries if needed
                    while (
                        len(self._cache) >= self.max_entries or
                        self.stats['total_bytes'] + entry_size > self.max_memory_bytes
                    ):
                        if not self._cache:
                            # Cache empty but entry too large
                            return False

                        # Remove oldest (first) entry
                        old_key, old_entry = self._cache.popitem(last=False)
                        with self.stats_lock:
                            self.stats['evictions'] += 1
                            self.stats['total_bytes'] -= old_entry.size_bytes

                        # Clean up lock
                        if old_key in self._interval_locks:
                            del self._interval_locks[old_key]

                # Create cache entry
                entry = CacheEntry(
                    df=df_copy,
                    interval=interval,
                    ticker=ticker,
                    cached_at=time.time(),
                    size_bytes=int(entry_size),
                )

                # Store in cache
                self._cache[key] = entry
                with self.stats_lock:
                    self.stats['total_bytes'] += entry.size_bytes

                return True

    def invalidate(self, ticker: Optional[str] = None, interval: Optional[str] = None):
        """
        Invalidate cache entries

        Args:
            ticker: If provided, only invalidate this ticker
            interval: If provided, only invalidate this interval

        If both are None, invalidates entire cache
        """
        with self._cache_lock:
            if ticker is None and interval is None:
                # Clear everything
                count = len(self._cache)
                self._cache.clear()
                self._interval_locks.clear()
                with self.stats_lock:
                    self.stats['invalidations'] += count
                    self.stats['total_bytes'] = 0
                return

            # Selective invalidation
            keys_to_remove = []
            for key in self._cache:
                cached_ticker, cached_interval = key
                if (ticker is None or cached_ticker == ticker) and \
                   (interval is None or cached_interval == interval):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                entry = self._cache.pop(key)
                with self.stats_lock:
                    self.stats['invalidations'] += 1
                    self.stats['total_bytes'] -= entry.size_bytes

                # Clean up lock
                if key in self._interval_locks:
                    del self._interval_locks[key]

    def on_market_open(self):
        """
        Called when market opens - invalidate all cached data to force refresh
        """
        now = time.time()
        self._market_open_time = now
        self._last_invalidation = now
        self._market_is_open = True  # Market is now open (use short TTL)
        self.invalidate()  # Clear all cache on market open

        if self.adaptive_ttl:
            print(f"[cache] Market opened - adaptive TTL set to {self._ttl_market_open}s ({self._ttl_market_open/60:.1f} min)")

    def on_market_close(self):
        """
        Called when market closes - data is now stable, cache can be kept longer
        """
        self._market_is_open = False  # Market is now closed (use long TTL)

        # Don't invalidate on close - data is stable now, cache is valuable
        # Just update TTL for future accesses
        if self.adaptive_ttl:
            print(f"[cache] Market closed - adaptive TTL set to {self._ttl_market_closed}s ({self._ttl_market_closed/3600:.1f} hours)")

    def warm_cache(
        self,
        intervals: List[Tuple[str, str]],
        loader_fn: Callable[[str, str], pd.DataFrame]
    ):
        """
        Pre-load intervals into cache

        Args:
            intervals: List of (ticker, interval) tuples to warm
            loader_fn: Function to load data: loader_fn(ticker, interval) -> DataFrame
        """
        for ticker, interval in intervals:
            try:
                df = loader_fn(ticker, interval)
                if df is not None and not df.empty:
                    self.put(ticker, interval, df, force=False)
            except Exception as e:
                # Log but don't fail
                print(f"[cache] Warning: Failed to warm cache for {ticker} {interval}: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self.stats_lock:
            stats = self.stats.copy()

        with self._cache_lock:
            stats['entries'] = len(self._cache)
            stats['max_entries'] = self.max_entries
            stats['memory_mb'] = stats['total_bytes'] / (1024 * 1024)
            stats['max_memory_mb'] = self.max_memory_bytes / (1024 * 1024)

        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        stats['hit_rate'] = (
            stats['hits'] / total_requests if total_requests > 0 else 0.0
        )

        return stats

    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        print(f"[IntervalDataCache Stats]")
        print(f"  Entries: {stats['entries']}/{stats['max_entries']}")
        print(f"  Memory: {stats['memory_mb']:.1f}/{stats['max_memory_mb']:.1f} MB")
        print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
        print(f"  Hit Rate: {stats['hit_rate']:.1%}")
        print(f"  Evictions: {stats['evictions']}, Invalidations: {stats['invalidations']}")

        # Show adaptive TTL info
        if self.adaptive_ttl:
            current_ttl = self._get_current_ttl()
            market_state = "OPEN" if self._market_is_open else "CLOSED"
            print(f"  Adaptive TTL: {current_ttl}s (market {market_state})")


# Global cache instance (process-wide singleton)
_global_cache: Optional[IntervalDataCache] = None
_global_cache_lock = threading.Lock()


def get_global_cache() -> IntervalDataCache:
    """
    Get or create global cache instance

    This is a singleton that persists across the entire process lifetime,
    allowing all workers (GPU, CPU, PREDICT) to share cached data.
    """
    global _global_cache

    if _global_cache is not None:
        return _global_cache

    with _global_cache_lock:
        if _global_cache is None:
            # Read config from environment
            max_entries = int(os.getenv("INTERVAL_CACHE_MAX_ENTRIES", "100"))
            max_memory_mb = float(os.getenv("INTERVAL_CACHE_MAX_MEMORY_MB", "500"))
            ttl_seconds = float(os.getenv("INTERVAL_CACHE_TTL_SECONDS", "3600"))
            adaptive_ttl = os.getenv("INTERVAL_CACHE_ADAPTIVE_TTL", "true").lower() in {"1", "true", "yes", "on"}

            _global_cache = IntervalDataCache(
                max_entries=max_entries,
                max_memory_mb=max_memory_mb,
                ttl_seconds=ttl_seconds,
                adaptive_ttl=adaptive_ttl,
            )

            ttl_mode = "adaptive" if adaptive_ttl else f"{ttl_seconds}s fixed"
            print(f"[cache] Initialized global interval cache: "
                  f"{max_entries} entries, {max_memory_mb} MB, TTL={ttl_mode}")

        return _global_cache


def cached_load_raw_df(
    ticker: str,
    interval: str,
    loader_fn: Callable[[str, str], pd.DataFrame],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load interval data with caching

    This is a drop-in replacement for load_raw_df() that adds caching.

    Args:
        ticker: Ticker symbol
        interval: Interval string
        loader_fn: Function to load from DB: loader_fn(ticker, interval) -> DataFrame
        force_refresh: If True, bypass cache and force DB load

    Returns:
        DataFrame with interval data

    Example:
        from mlsystem.interval_data_cache import cached_load_raw_df

        # Old way (no caching):
        df = load_raw_df('DEMO', '30m')

        # New way (with caching):
        df = cached_load_raw_df('DEMO', '30m', lambda t, i: load_raw_df(t, i))
    """
    cache = get_global_cache()

    # Try cache first (unless force_refresh)
    if not force_refresh:
        df = cache.get(ticker, interval)
        if df is not None:
            return df

    # Cache miss or force refresh - load from DB
    df = loader_fn(ticker, interval)

    # Store in cache
    if df is not None and not df.empty:
        cache.put(ticker, interval, df)

    return df


def warm_interval_cache(intervals: List[Tuple[str, str]], loader_fn: Callable[[str, str], pd.DataFrame]):
    """
    Pre-load intervals into cache (call at startup or bucket transition)

    Args:
        intervals: List of (ticker, interval) tuples to pre-load
        loader_fn: Function to load from the database: loader_fn(ticker, interval) -> DataFrame
    """
    cache = get_global_cache()
    cache.warm_cache(intervals, loader_fn)


def invalidate_interval_cache(ticker: Optional[str] = None, interval: Optional[str] = None):
    """
    Invalidate cached data (call on market open or data updates)

    Args:
        ticker: If provided, only invalidate this ticker
        interval: If provided, only invalidate this interval
    """
    cache = get_global_cache()
    cache.invalidate(ticker, interval)


def on_market_open():
    """Call when market opens to invalidate stale cache"""
    cache = get_global_cache()
    cache.on_market_open()


def on_market_close():
    """Call when market closes - data is now stable"""
    cache = get_global_cache()
    cache.on_market_close()


def warm_cache_for_active_intervals(
    ticker: str,
    intervals: List[str],
    load_raw_df: Callable[[str, str], pd.DataFrame],
    verbose: bool = False,
):
    """
    Warm cache with commonly-used intervals (call at startup or bucket transition)

    Args:
        ticker: Ticker symbol to warm
        intervals: List of interval strings (e.g., ["1m", "30m", "4h"])
        load_raw_df: Function to load from the database: load_raw_df(ticker, interval) -> DataFrame
        verbose: Print warming progress
    """
    cache = get_global_cache()

    if verbose:
        print(f"[cache] Warming cache for {ticker} with {len(intervals)} intervals...")

    start_time = time.time()
    warmed = 0

    for interval in intervals:
        try:
            df = load_raw_df(ticker, interval)
            if df is not None and not df.empty:
                cache.put(ticker, interval, df, force=False)
                warmed += 1
                if verbose:
                    print(f"[cache]   ✓ Warmed {ticker} {interval} ({len(df)} rows)")
        except Exception as e:
            if verbose:
                print(f"[cache]   ✗ Failed to warm {ticker} {interval}: {e}")

    elapsed = time.time() - start_time
    if verbose:
        print(f"[cache] Warming complete: {warmed}/{len(intervals)} intervals in {elapsed:.2f}s")


def print_cache_stats():
    """Print cache statistics"""
    cache = get_global_cache()
    cache.print_stats()
