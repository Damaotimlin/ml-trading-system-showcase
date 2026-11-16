from mlsystem.settings import parse_cache_window


def test_cache_window_parsing():
    assert parse_cache_window("1m") == 63      # 3 months
    assert parse_cache_window("30m") == 252    # 1 year
    assert parse_cache_window("1d") is None    # full history
    assert parse_cache_window("unknown") is None
