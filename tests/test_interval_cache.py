#!/usr/bin/env python3
"""
Test interval data cache functionality

Usage:
    python3 tests/test_interval_cache.py
"""

import time
import pandas as pd
from mlsystem.interval_data_cache import IntervalDataCache


def test_basic_caching():
    """Test basic cache operations"""
    print("=" * 60)
    print("TEST: Basic Caching")
    print("=" * 60)

    cache = IntervalDataCache(max_entries=10, max_memory_mb=100, ttl_seconds=3600)

    # Create test DataFrame
    df = pd.DataFrame({
        'Datetime': pd.date_range('2025-01-01', periods=1000, freq='1min'),
        'Open': [100.0 + i * 0.1 for i in range(1000)],
        'High': [100.5 + i * 0.1 for i in range(1000)],
        'Low': [99.5 + i * 0.1 for i in range(1000)],
        'Close': [100.0 + i * 0.1 for i in range(1000)],
        'Volume': [1000000] * 1000,
    })

    # Test cache miss
    result = cache.get('DEMO', '30m')
    assert result is None, "Expected cache miss on first access"
    print("✓ Cache miss (expected)")

    # Store in cache
    success = cache.put('DEMO', '30m', df)
    assert success, "Failed to store in cache"
    print("✓ Data stored in cache")

    # Test cache hit
    result = cache.get('DEMO', '30m')
    assert result is not None, "Expected cache hit"
    assert len(result) == len(df), "Cached data has wrong length"
    assert list(result.columns) == list(df.columns), "Cached data has wrong columns"
    print("✓ Cache hit (expected)")

    # Check stats
    stats = cache.get_stats()
    assert stats['hits'] == 1, f"Expected 1 hit, got {stats['hits']}"
    assert stats['misses'] == 1, f"Expected 1 miss, got {stats['misses']}"
    assert stats['entries'] == 1, f"Expected 1 entry, got {stats['entries']}"
    print(f"✓ Stats correct: {stats['hits']} hits, {stats['misses']} misses")

    print("\n✅ Basic caching test PASSED\n")


def test_lru_eviction():
    """Test LRU eviction when cache is full"""
    print("=" * 60)
    print("TEST: LRU Eviction")
    print("=" * 60)

    # Small cache for testing eviction
    cache = IntervalDataCache(max_entries=3, max_memory_mb=100, ttl_seconds=3600)

    # Create small test DataFrames
    df = pd.DataFrame({
        'Datetime': pd.date_range('2025-01-01', periods=10, freq='1min'),
        'Close': [100.0] * 10,
    })

    # Fill cache to capacity
    cache.put('DEMO', '1m', df)
    cache.put('DEMO', '5m', df)
    cache.put('DEMO', '30m', df)

    stats = cache.get_stats()
    assert stats['entries'] == 3, f"Expected 3 entries, got {stats['entries']}"
    print(f"✓ Cache filled to capacity: {stats['entries']}/3 entries")

    # Add one more (should evict oldest)
    cache.put('DEMO', '60m', df)

    stats = cache.get_stats()
    assert stats['entries'] == 3, f"Expected 3 entries after eviction, got {stats['entries']}"
    assert stats['evictions'] == 1, f"Expected 1 eviction, got {stats['evictions']}"
    print(f"✓ LRU eviction occurred: {stats['evictions']} evictions")

    # Oldest entry should be gone
    result = cache.get('DEMO', '1m')
    assert result is None, "Oldest entry should have been evicted"
    print("✓ Oldest entry evicted (expected)")

    # Newer entries should still exist
    result = cache.get('DEMO', '60m')
    assert result is not None, "Newest entry should exist"
    print("✓ Newest entry still cached")

    print("\n✅ LRU eviction test PASSED\n")


def test_ttl_expiry():
    """Test TTL expiration"""
    print("=" * 60)
    print("TEST: TTL Expiration")
    print("=" * 60)

    # Very short fixed TTL for testing (adaptive TTL would use market-state timings instead)
    cache = IntervalDataCache(max_entries=10, max_memory_mb=100, ttl_seconds=1, adaptive_ttl=False)

    df = pd.DataFrame({'Close': [100.0]})

    # Store in cache
    cache.put('DEMO', '30m', df)
    print("✓ Data stored in cache")

    # Should hit immediately
    result = cache.get('DEMO', '30m')
    assert result is not None, "Should be cached immediately"
    print("✓ Cache hit (immediate)")

    # Wait for TTL to expire
    print("  Waiting 2 seconds for TTL expiry...")
    time.sleep(2)

    # Should miss after TTL expires
    result = cache.get('DEMO', '30m')
    assert result is None, "Should be evicted after TTL"
    print("✓ Cache miss after TTL expiry (expected)")

    stats = cache.get_stats()
    assert stats['evictions'] == 1, f"Expected 1 eviction due to TTL, got {stats['evictions']}"

    print("\n✅ TTL expiration test PASSED\n")


def test_invalidation():
    """Test cache invalidation"""
    print("=" * 60)
    print("TEST: Cache Invalidation")
    print("=" * 60)

    cache = IntervalDataCache(max_entries=10, max_memory_mb=100, ttl_seconds=3600)

    df = pd.DataFrame({'Close': [100.0]})

    # Store multiple entries
    cache.put('DEMO', '30m', df)
    cache.put('ACME', '30m', df)
    cache.put('DEMO', '60m', df)

    stats = cache.get_stats()
    assert stats['entries'] == 3, f"Expected 3 entries, got {stats['entries']}"
    print(f"✓ {stats['entries']} entries cached")

    # Invalidate specific ticker
    cache.invalidate(ticker='DEMO')

    stats = cache.get_stats()
    assert stats['entries'] == 1, f"Expected 1 entry after DEMO invalidation, got {stats['entries']}"
    assert stats['invalidations'] == 2, f"Expected 2 invalidations, got {stats['invalidations']}"
    print("✓ DEMO entries invalidated (2 entries)")

    # ACME should still exist
    result = cache.get('ACME', '30m')
    assert result is not None, "ACME should still be cached"
    print("✓ ACME entry still cached")

    # Add more entries
    cache.put('DEMO', '30m', df)
    cache.put('IWM', '30m', df)

    # Invalidate all
    cache.invalidate()

    stats = cache.get_stats()
    assert stats['entries'] == 0, f"Expected 0 entries after full invalidation, got {stats['entries']}"
    print("✓ All entries invalidated")

    print("\n✅ Cache invalidation test PASSED\n")


def test_parallel_access():
    """Test concurrent access from multiple threads"""
    print("=" * 60)
    print("TEST: Parallel Access (Thread Safety)")
    print("=" * 60)

    import threading

    cache = IntervalDataCache(max_entries=50, max_memory_mb=200, ttl_seconds=3600)

    df = pd.DataFrame({
        'Datetime': pd.date_range('2025-01-01', periods=1000, freq='1min'),
        'Close': [100.0] * 1000,
    })

    errors = []

    def worker(worker_id):
        """Worker thread that accesses cache"""
        try:
            interval = f"{worker_id % 3}m"  # Use 3 different intervals

            # Try to get from cache
            result = cache.get('DEMO', interval)

            # If miss, store in cache
            if result is None:
                cache.put('DEMO', interval, df)
                time.sleep(0.001)  # Small delay

            # Get again (should hit this time)
            result = cache.get('DEMO', interval)
            assert result is not None, f"Worker {worker_id} failed to get cached data"

        except Exception as e:
            errors.append(f"Worker {worker_id}: {e}")

    # Launch multiple worker threads
    threads = []
    num_workers = 10

    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"
    print(f"✓ {num_workers} workers completed without errors")

    stats = cache.get_stats()
    print(f"✓ Final stats: {stats['hits']} hits, {stats['misses']} misses, {stats['entries']} entries")

    print("\n✅ Parallel access test PASSED\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("INTERVAL DATA CACHE TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_basic_caching,
        test_lru_eviction,
        test_ttl_expiry,
        test_invalidation,
        test_parallel_access,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}\n")
            failed.append((test.__name__, e))
            import traceback
            traceback.print_exc()

    print("=" * 60)
    if failed:
        print(f"❌ FAILED: {len(failed)}/{len(tests)} tests")
        for name, error in failed:
            print(f"  - {name}: {error}")
    else:
        print(f"✅ SUCCESS: All {len(tests)} tests passed!")
    print("=" * 60)

    return len(failed) == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
