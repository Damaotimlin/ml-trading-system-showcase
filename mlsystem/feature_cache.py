"""
Feature Cache Manager with configurable rolling windows

Caches preprocessed features to speed up model training by 10-30x.
Cache is automatically invalidated on market open when new data arrives.

Cache Strategy:
- Weekend: Compute features during deep-search, populate cache
- Weekday market_off: Load from cache, append incremental data
- Rolling window: Configurable per interval (1m=3M, 30m=1Y, 60m=ALL)

Cache Invalidation:
- Automatic on market open (new data available)
- On best.json updates (hyperparameter changes)
- On mode/config changes (hybrid → returns, etc.)
"""
import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class FeatureCache:
    """
    Persistent cache for preprocessed features with rolling window support

    Cache format:
    - {cache_key}.parquet: Feature DataFrame (ZSTD compressed)
    - {cache_key}.meta.json: Metadata (interval, mode, timestamps, etc.)
    """

    def __init__(self, cache_dir: str = "/data/Model/feature_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.hit_count = 0
        self.miss_count = 0

    def _build_key(
        self,
        ticker: str,
        interval: str,
        mode: str,
        horizon: int,
        scaler_hash: str
    ) -> str:
        """
        Generate unique cache key from parameters

        Args:
            ticker: Ticker symbol(s) - joined with underscore if multiple
            interval: Time interval (1m, 30m, 60m, etc.)
            mode: Data representation mode (absolute, delta, mix, hybrid, returns)
            horizon: Prediction horizon steps
            scaler_hash: Hash of scaler configuration

        Returns:
            12-character hex cache key
        """
        key_str = f"{ticker}_{interval}_{mode}_{horizon}_{scaler_hash}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Load cached features if available

        Args:
            cache_key: Unique cache identifier

        Returns:
            Dict with 'features' (DataFrame or dict of DataFrames) and 'metadata', or None
        """
        meta_path = self.cache_dir / f"{cache_key}.meta.json"

        if not meta_path.exists():
            self.miss_count += 1
            return None

        try:
            with open(meta_path) as f:
                metadata = json.load(f)

            interval = metadata.get('interval')

            if metadata.get('splits_dict'):
                splits = {}
                for split_name in ('train', 'val', 'test'):
                    split_path = self.cache_dir / f"{cache_key}_{split_name}.parquet"
                    if split_path.exists():
                        df = pd.read_parquet(split_path)
                        if interval:
                            df = self._apply_rolling_window(df, interval)
                        splits[split_name] = df
                if not splits:
                    self.miss_count += 1
                    return None
                self.hit_count += 1
                return {'features': splits, 'metadata': metadata}

            parquet_path = self.cache_dir / f"{cache_key}.parquet"
            if not parquet_path.exists():
                self.miss_count += 1
                return None

            features_df = pd.read_parquet(parquet_path)
            if interval:
                features_df = self._apply_rolling_window(features_df, interval)

            self.hit_count += 1
            return {'features': features_df, 'metadata': metadata}

        except Exception as e:
            logger.warning(f"[feature_cache] Failed to load cache {cache_key}: {e}")
            self.miss_count += 1
            return None

    def _apply_rolling_window(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Apply rolling window pruning based on interval configuration

        Args:
            df: Feature DataFrame with DatetimeIndex or Date/Datetime column
            interval: Time interval (1m, 30m, 60m)

        Returns:
            Pruned DataFrame within rolling window
        """
        from .settings import parse_cache_window

        window_days = parse_cache_window(interval)

        if window_days is None:
            # 'ALL' - keep full history
            return df

        # Calculate cutoff with 40% buffer for safety
        cutoff_days = int(window_days * 1.4)
        cutoff = pd.Timestamp.now(tz='US/Eastern') - pd.Timedelta(days=cutoff_days)

        # Determine if datetime is index or column
        if isinstance(df.index, pd.DatetimeIndex):
            # DatetimeIndex - use index for filtering
            pruned = df[df.index >= cutoff]
        elif 'Date' in df.columns:
            # Date column - convert to datetime and filter
            date_col = pd.to_datetime(df['Date'], errors='coerce')
            pruned = df[date_col >= cutoff]
        elif 'Datetime' in df.columns:
            # Datetime column - convert to datetime and filter
            datetime_col = pd.to_datetime(df['Datetime'], errors='coerce')
            pruned = df[datetime_col >= cutoff]
        else:
            # No datetime column/index found - return as-is (no pruning possible)
            logger.warning(f"[feature_cache] No datetime index/column found, skipping rolling window pruning")
            return df

        if len(pruned) < len(df):
            logger.debug(f"[feature_cache] Pruned {len(df) - len(pruned)} rows (keeping {len(pruned)} within {window_days} trading days)")

        return pruned

    def set(
        self,
        cache_key: str,
        features_df: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Save features to cache

        Args:
            cache_key: Unique cache identifier
            features_df: Feature DataFrame to cache
            metadata: Metadata dict (interval, mode, created_at, etc.)
        """
        parquet_path = self.cache_dir / f"{cache_key}.parquet"
        meta_path = self.cache_dir / f"{cache_key}.meta.json"

        try:
            # Add timestamp if not present
            if 'created_at' not in metadata:
                metadata['created_at'] = datetime.now(timezone.utc).isoformat()

            # Save features (ZSTD compression for speed + size)
            features_df.to_parquet(parquet_path, compression='zstd')

            # Save metadata
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"[feature_cache] ✓ Cached {len(features_df)} rows ({cache_key})")

        except Exception as e:
            logger.error(f"[feature_cache] Failed to save cache {cache_key}: {e}")

    def set_splits(
        self,
        cache_key: str,
        splits: Dict[str, pd.DataFrame],
        metadata: Dict[str, Any]
    ) -> None:
        """Save train/val/test splits as separate parquet files."""
        meta_path = self.cache_dir / f"{cache_key}.meta.json"

        try:
            if 'created_at' not in metadata:
                metadata['created_at'] = datetime.now(timezone.utc).isoformat()
            metadata['splits_dict'] = True

            for split_name, df in splits.items():
                if df is not None and isinstance(df, pd.DataFrame):
                    split_path = self.cache_dir / f"{cache_key}_{split_name}.parquet"
                    df.to_parquet(split_path, compression='zstd')

            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            total_rows = sum(len(df) for df in splits.values() if df is not None)
            logger.info(f"[feature_cache] ✓ Cached {total_rows} rows across {len(splits)} splits ({cache_key})")

        except Exception as e:
            logger.error(f"[feature_cache] Failed to save split cache {cache_key}: {e}")

    def append_incremental(
        self,
        cache_key: str,
        new_rows: pd.DataFrame
    ) -> bool:
        """
        Append only new rows to existing cache (incremental update)

        Args:
            cache_key: Unique cache identifier
            new_rows: New feature rows to append

        Returns:
            True if successful, False otherwise
        """
        cached = self.get(cache_key)
        if cached is None:
            logger.warning(f"[feature_cache] Cannot append - cache key {cache_key} not found")
            return False

        try:
            # Merge and deduplicate (keep latest)
            combined = pd.concat([cached['features'], new_rows])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()

            # Update metadata
            cached['metadata']['last_updated'] = datetime.now(timezone.utc).isoformat()
            cached['metadata']['row_count'] = len(combined)

            # Save back
            self.set(cache_key, combined, cached['metadata'])

            logger.info(f"[feature_cache] ✓ Appended {len(new_rows)} rows to {cache_key}")
            return True

        except Exception as e:
            logger.error(f"[feature_cache] Failed to append to cache {cache_key}: {e}")
            return False

    def invalidate_all(self) -> int:
        """
        Clear all cache (called on market open when new data arrives)

        Returns:
            Number of cache entries deleted
        """
        count = 0

        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
            count += 1

        for f in self.cache_dir.glob("*.meta.json"):
            f.unlink()

        if count > 0:
            logger.info(f"[feature_cache] Invalidated {count} cache entries")

        return count

    def invalidate_key(self, cache_key: str) -> bool:
        """
        Invalidate specific cache entry

        Args:
            cache_key: Unique cache identifier

        Returns:
            True if deleted, False if not found
        """
        parquet_path = self.cache_dir / f"{cache_key}.parquet"
        meta_path = self.cache_dir / f"{cache_key}.meta.json"

        deleted = False

        if parquet_path.exists():
            parquet_path.unlink()
            deleted = True

        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        if deleted:
            logger.info(f"[feature_cache] Invalidated cache key {cache_key}")

        return deleted

    def is_valid(
        self,
        cached: Dict[str, Any],
        db_path: str,
        best_json_path: Optional[str] = None,
        current_mode: Optional[str] = None,
        current_horizon: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Check if cached features are still valid

        Invalidation triggers:
        1. DB has newer data than cache
        2. best.json updated (hyperparameter changes)
        3. Config changed (mode, horizon, etc.)

        Args:
            cached: Cached data dict from get()
            db_path: Path to source database
            best_json_path: Optional path to best.json
            current_mode: Current data representation mode
            current_horizon: Current prediction horizon

        Returns:
            (is_valid, reason) tuple
        """
        metadata = cached['metadata']

        # Check 1: DB has newer data
        try:
            db_mtime = Path(db_path).stat().st_mtime
            cache_created = pd.Timestamp(metadata['created_at']).timestamp()

            if db_mtime > cache_created:
                return False, "DB has newer data"
        except Exception as e:
            return False, f"DB check failed: {e}"

        # Check 2: best.json updated
        if best_json_path and Path(best_json_path).exists():
            try:
                best_json_mtime = Path(best_json_path).stat().st_mtime

                if best_json_mtime > cache_created:
                    return False, "best.json updated (hyperparameters changed)"
            except Exception:
                pass  # best.json check is optional

        # Check 3: Mode changed
        if current_mode and metadata.get('mode') != current_mode:
            return False, f"mode changed ({metadata.get('mode')} → {current_mode})"

        # Check 4: Horizon changed
        if current_horizon and metadata.get('horizon_steps') != current_horizon:
            return False, f"horizon changed ({metadata.get('horizon_steps')} → {current_horizon})"

        return True, "valid"

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and health metrics

        Returns:
            Dict with cache statistics
        """
        entries = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in entries) / 1024**2  # MB

        # Calculate hit rate
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0

        # Find oldest and newest cache
        oldest = None
        newest = None

        for meta_file in self.cache_dir.glob("*.meta.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                    created = pd.Timestamp(meta['created_at'])

                    if oldest is None or created < oldest:
                        oldest = created
                    if newest is None or created > newest:
                        newest = created
            except Exception:
                continue

        return {
            'cache_dir': str(self.cache_dir),
            'entries': len(entries),
            'size_mb': round(total_size, 2),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate_pct': round(hit_rate, 1),
            'oldest_entry': oldest.isoformat() if oldest else None,
            'newest_entry': newest.isoformat() if newest else None,
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on cache

        Returns:
            Dict with health status and issues found
        """
        issues = []
        orphaned = []
        corrupted = []

        # Check for orphaned files (parquet without meta or vice versa)
        parquet_files = set(f.stem for f in self.cache_dir.glob("*.parquet"))
        meta_files = set(f.stem.replace('.meta', '') for f in self.cache_dir.glob("*.meta.json"))

        orphaned_parquet = parquet_files - meta_files
        orphaned_meta = meta_files - parquet_files

        if orphaned_parquet:
            issues.append(f"{len(orphaned_parquet)} orphaned .parquet files")
            orphaned.extend([f"{k}.parquet" for k in orphaned_parquet])

        if orphaned_meta:
            issues.append(f"{len(orphaned_meta)} orphaned .meta.json files")
            orphaned.extend([f"{k}.meta.json" for k in orphaned_meta])

        # Check for corrupted cache entries
        for cache_key in parquet_files & meta_files:
            try:
                cached = self.get(cache_key)
                if cached is None:
                    corrupted.append(cache_key)
            except Exception:
                corrupted.append(cache_key)

        if corrupted:
            issues.append(f"{len(corrupted)} corrupted cache entries")

        # Overall health status
        is_healthy = len(issues) == 0

        return {
            'healthy': is_healthy,
            'issues': issues,
            'orphaned_files': orphaned,
            'corrupted_entries': corrupted,
            'recommendation': 'Run cache.clean_orphaned() to fix issues' if not is_healthy else 'Cache is healthy'
        }

    def clean_orphaned(self) -> int:
        """
        Clean up orphaned and corrupted cache files

        Returns:
            Number of files cleaned
        """
        health = self.health_check()
        cleaned = 0

        # Remove orphaned files
        for filename in health['orphaned_files']:
            filepath = self.cache_dir / filename
            if filepath.exists():
                filepath.unlink()
                cleaned += 1
                logger.info(f"[feature_cache] Cleaned orphaned file: {filename}")

        # Remove corrupted entries
        for cache_key in health['corrupted_entries']:
            if self.invalidate_key(cache_key):
                cleaned += 1

        if cleaned > 0:
            logger.info(f"[feature_cache] Cleaned {cleaned} cache files")

        return cleaned


def get_global_cache() -> FeatureCache:
    """Get singleton feature cache instance"""
    global _global_cache

    if '_global_cache' not in globals():
        _global_cache = FeatureCache()

    return _global_cache
