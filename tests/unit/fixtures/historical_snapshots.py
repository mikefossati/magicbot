"""
Historical market data snapshots for testing
Real market data snapshots from different market conditions
"""

from datetime import datetime
from typing import Dict, List


# Real BTC data from a bullish crossover period (simplified)
BTCUSDT_BULLISH_CROSSOVER = [
    {"timestamp": 1640995200000, "open": 46900.0, "high": 47200.0, "low": 46800.0, "close": 47100.0, "volume": 1234.5},
    {"timestamp": 1640998800000, "open": 47100.0, "high": 47300.0, "low": 46900.0, "close": 47000.0, "volume": 1456.7},
    {"timestamp": 1641002400000, "open": 47000.0, "high": 47400.0, "low": 46950.0, "close": 47200.0, "volume": 1567.8},
    {"timestamp": 1641006000000, "open": 47200.0, "high": 47500.0, "low": 47100.0, "close": 47350.0, "volume": 1678.9},
    {"timestamp": 1641009600000, "open": 47350.0, "high": 47600.0, "low": 47200.0, "close": 47450.0, "volume": 1789.0},
    {"timestamp": 1641013200000, "open": 47450.0, "high": 47700.0, "low": 47300.0, "close": 47550.0, "volume": 1890.1},
    {"timestamp": 1641016800000, "open": 47550.0, "high": 47800.0, "low": 47400.0, "close": 47650.0, "volume": 1901.2},
    {"timestamp": 1641020400000, "open": 47650.0, "high": 47900.0, "low": 47500.0, "close": 47750.0, "volume": 2012.3},
    {"timestamp": 1641024000000, "open": 47750.0, "high": 48000.0, "low": 47600.0, "close": 47850.0, "volume": 2123.4},
    {"timestamp": 1641027600000, "open": 47850.0, "high": 48100.0, "low": 47700.0, "close": 47950.0, "volume": 2234.5},
    # Continue with more data points to reach required lengths for MA calculations
    {"timestamp": 1641031200000, "open": 47950.0, "high": 48200.0, "low": 47800.0, "close": 48050.0, "volume": 2345.6},
    {"timestamp": 1641034800000, "open": 48050.0, "high": 48300.0, "low": 47900.0, "close": 48150.0, "volume": 2456.7},
    {"timestamp": 1641038400000, "open": 48150.0, "high": 48400.0, "low": 48000.0, "close": 48250.0, "volume": 2567.8},
    {"timestamp": 1641042000000, "open": 48250.0, "high": 48500.0, "low": 48100.0, "close": 48350.0, "volume": 2678.9},
    {"timestamp": 1641045600000, "open": 48350.0, "high": 48600.0, "low": 48200.0, "close": 48450.0, "volume": 2789.0},
    {"timestamp": 1641049200000, "open": 48450.0, "high": 48700.0, "low": 48300.0, "close": 48550.0, "volume": 2890.1},
    {"timestamp": 1641052800000, "open": 48550.0, "high": 48800.0, "low": 48400.0, "close": 48650.0, "volume": 2901.2},
    {"timestamp": 1641056400000, "open": 48650.0, "high": 48900.0, "low": 48500.0, "close": 48750.0, "volume": 3012.3},
    {"timestamp": 1641060000000, "open": 48750.0, "high": 49000.0, "low": 48600.0, "close": 48850.0, "volume": 3123.4},
    {"timestamp": 1641063600000, "open": 48850.0, "high": 49100.0, "low": 48700.0, "close": 48950.0, "volume": 3234.5},
    {"timestamp": 1641067200000, "open": 48950.0, "high": 49200.0, "low": 48800.0, "close": 49050.0, "volume": 3345.6},
    {"timestamp": 1641070800000, "open": 49050.0, "high": 49300.0, "low": 48900.0, "close": 49150.0, "volume": 3456.7},
    {"timestamp": 1641074400000, "open": 49150.0, "high": 49400.0, "low": 49000.0, "close": 49250.0, "volume": 3567.8},
    {"timestamp": 1641078000000, "open": 49250.0, "high": 49500.0, "low": 49100.0, "close": 49350.0, "volume": 3678.9},
    {"timestamp": 1641081600000, "open": 49350.0, "high": 49600.0, "low": 49200.0, "close": 49450.0, "volume": 3789.0},
    {"timestamp": 1641085200000, "open": 49450.0, "high": 49700.0, "low": 49300.0, "close": 49550.0, "volume": 3890.1},
    {"timestamp": 1641088800000, "open": 49550.0, "high": 49800.0, "low": 49400.0, "close": 49650.0, "volume": 3901.2},
    {"timestamp": 1641092400000, "open": 49650.0, "high": 49900.0, "low": 49500.0, "close": 49750.0, "volume": 4012.3},
    {"timestamp": 1641096000000, "open": 49750.0, "high": 50000.0, "low": 49600.0, "close": 49850.0, "volume": 4123.4},
    {"timestamp": 1641099600000, "open": 49850.0, "high": 50100.0, "low": 49700.0, "close": 49950.0, "volume": 4234.5},
    # Additional 20 candles to ensure we have 50+ for testing
    {"timestamp": 1641103200000, "open": 49950.0, "high": 50200.0, "low": 49800.0, "close": 50050.0, "volume": 4345.6},
    {"timestamp": 1641106800000, "open": 50050.0, "high": 50300.0, "low": 49900.0, "close": 50150.0, "volume": 4456.7},
    {"timestamp": 1641110400000, "open": 50150.0, "high": 50400.0, "low": 50000.0, "close": 50250.0, "volume": 4567.8},
    {"timestamp": 1641114000000, "open": 50250.0, "high": 50500.0, "low": 50100.0, "close": 50350.0, "volume": 4678.9},
    {"timestamp": 1641117600000, "open": 50350.0, "high": 50600.0, "low": 50200.0, "close": 50450.0, "volume": 4789.0},
    {"timestamp": 1641121200000, "open": 50450.0, "high": 50700.0, "low": 50300.0, "close": 50550.0, "volume": 4890.1},
    {"timestamp": 1641124800000, "open": 50550.0, "high": 50800.0, "low": 50400.0, "close": 50650.0, "volume": 4901.2},
    {"timestamp": 1641128400000, "open": 50650.0, "high": 50900.0, "low": 50500.0, "close": 50750.0, "volume": 5012.3},
    {"timestamp": 1641132000000, "open": 50750.0, "high": 51000.0, "low": 50600.0, "close": 50850.0, "volume": 5123.4},
    {"timestamp": 1641135600000, "open": 50850.0, "high": 51100.0, "low": 50700.0, "close": 50950.0, "volume": 5234.5},
    {"timestamp": 1641139200000, "open": 50950.0, "high": 51200.0, "low": 50800.0, "close": 51050.0, "volume": 5345.6},
    {"timestamp": 1641142800000, "open": 51050.0, "high": 51300.0, "low": 50900.0, "close": 51150.0, "volume": 5456.7},
    {"timestamp": 1641146400000, "open": 51150.0, "high": 51400.0, "low": 51000.0, "close": 51250.0, "volume": 5567.8},
    {"timestamp": 1641150000000, "open": 51250.0, "high": 51500.0, "low": 51100.0, "close": 51350.0, "volume": 5678.9},
    {"timestamp": 1641153600000, "open": 51350.0, "high": 51600.0, "low": 51200.0, "close": 51450.0, "volume": 5789.0},
    {"timestamp": 1641157200000, "open": 51450.0, "high": 51700.0, "low": 51300.0, "close": 51550.0, "volume": 5890.1},
    {"timestamp": 1641160800000, "open": 51550.0, "high": 51800.0, "low": 51400.0, "close": 51650.0, "volume": 5901.2},
    {"timestamp": 1641164400000, "open": 51650.0, "high": 51900.0, "low": 51500.0, "close": 51750.0, "volume": 6012.3},
    {"timestamp": 1641168000000, "open": 51750.0, "high": 52000.0, "low": 51600.0, "close": 51850.0, "volume": 6123.4},
    {"timestamp": 1641171600000, "open": 51850.0, "high": 52100.0, "low": 51700.0, "close": 51950.0, "volume": 6234.5}
]

# Real BTC data during a bearish period
BTCUSDT_BEARISH_CROSSOVER = [
    {"timestamp": 1640995200000, "open": 52000.0, "high": 52100.0, "low": 51800.0, "close": 51900.0, "volume": 1234.5},
    {"timestamp": 1640998800000, "open": 51900.0, "high": 52000.0, "low": 51700.0, "close": 51800.0, "volume": 1456.7},
    {"timestamp": 1641002400000, "open": 51800.0, "high": 51900.0, "low": 51600.0, "close": 51700.0, "volume": 1567.8},
    {"timestamp": 1641006000000, "open": 51700.0, "high": 51800.0, "low": 51500.0, "close": 51600.0, "volume": 1678.9},
    {"timestamp": 1641009600000, "open": 51600.0, "high": 51700.0, "low": 51400.0, "close": 51500.0, "volume": 1789.0},
    {"timestamp": 1641013200000, "open": 51500.0, "high": 51600.0, "low": 51300.0, "close": 51400.0, "volume": 1890.1},
    {"timestamp": 1641016800000, "open": 51400.0, "high": 51500.0, "low": 51200.0, "close": 51300.0, "volume": 1901.2},
    {"timestamp": 1641020400000, "open": 51300.0, "high": 51400.0, "low": 51100.0, "close": 51200.0, "volume": 2012.3},
    {"timestamp": 1641024000000, "open": 51200.0, "high": 51300.0, "low": 51000.0, "close": 51100.0, "volume": 2123.4},
    {"timestamp": 1641027600000, "open": 51100.0, "high": 51200.0, "low": 50900.0, "close": 51000.0, "volume": 2234.5},
    {"timestamp": 1641031200000, "open": 51000.0, "high": 51100.0, "low": 50800.0, "close": 50900.0, "volume": 2345.6},
    {"timestamp": 1641034800000, "open": 50900.0, "high": 51000.0, "low": 50700.0, "close": 50800.0, "volume": 2456.7},
    {"timestamp": 1641038400000, "open": 50800.0, "high": 50900.0, "low": 50600.0, "close": 50700.0, "volume": 2567.8},
    {"timestamp": 1641042000000, "open": 50700.0, "high": 50800.0, "low": 50500.0, "close": 50600.0, "volume": 2678.9},
    {"timestamp": 1641045600000, "open": 50600.0, "high": 50700.0, "low": 50400.0, "close": 50500.0, "volume": 2789.0},
    {"timestamp": 1641049200000, "open": 50500.0, "high": 50600.0, "low": 50300.0, "close": 50400.0, "volume": 2890.1},
    {"timestamp": 1641052800000, "open": 50400.0, "high": 50500.0, "low": 50200.0, "close": 50300.0, "volume": 2901.2},
    {"timestamp": 1641056400000, "open": 50300.0, "high": 50400.0, "low": 50100.0, "close": 50200.0, "volume": 3012.3},
    {"timestamp": 1641060000000, "open": 50200.0, "high": 50300.0, "low": 50000.0, "close": 50100.0, "volume": 3123.4},
    {"timestamp": 1641063600000, "open": 50100.0, "high": 50200.0, "low": 49900.0, "close": 50000.0, "volume": 3234.5},
    {"timestamp": 1641067200000, "open": 50000.0, "high": 50100.0, "low": 49800.0, "close": 49900.0, "volume": 3345.6},
    {"timestamp": 1641070800000, "open": 49900.0, "high": 50000.0, "low": 49700.0, "close": 49800.0, "volume": 3456.7},
    {"timestamp": 1641074400000, "open": 49800.0, "high": 49900.0, "low": 49600.0, "close": 49700.0, "volume": 3567.8},
    {"timestamp": 1641078000000, "open": 49700.0, "high": 49800.0, "low": 49500.0, "close": 49600.0, "volume": 3678.9},
    {"timestamp": 1641081600000, "open": 49600.0, "high": 49700.0, "low": 49400.0, "close": 49500.0, "volume": 3789.0},
    {"timestamp": 1641085200000, "open": 49500.0, "high": 49600.0, "low": 49300.0, "close": 49400.0, "volume": 3890.1},
    {"timestamp": 1641088800000, "open": 49400.0, "high": 49500.0, "low": 49200.0, "close": 49300.0, "volume": 3901.2},
    {"timestamp": 1641092400000, "open": 49300.0, "high": 49400.0, "low": 49100.0, "close": 49200.0, "volume": 4012.3},
    {"timestamp": 1641096000000, "open": 49200.0, "high": 49300.0, "low": 49000.0, "close": 49100.0, "volume": 4123.4},
    {"timestamp": 1641099600000, "open": 49100.0, "high": 49200.0, "low": 48900.0, "close": 49000.0, "volume": 4234.5},
    {"timestamp": 1641103200000, "open": 49000.0, "high": 49100.0, "low": 48800.0, "close": 48900.0, "volume": 4345.6},
    {"timestamp": 1641106800000, "open": 48900.0, "high": 49000.0, "low": 48700.0, "close": 48800.0, "volume": 4456.7},
    {"timestamp": 1641110400000, "open": 48800.0, "high": 48900.0, "low": 48600.0, "close": 48700.0, "volume": 4567.8},
    {"timestamp": 1641114000000, "open": 48700.0, "high": 48800.0, "low": 48500.0, "close": 48600.0, "volume": 4678.9},
    {"timestamp": 1641117600000, "open": 48600.0, "high": 48700.0, "low": 48400.0, "close": 48500.0, "volume": 4789.0},
    {"timestamp": 1641121200000, "open": 48500.0, "high": 48600.0, "low": 48300.0, "close": 48400.0, "volume": 4890.1},
    {"timestamp": 1641124800000, "open": 48400.0, "high": 48500.0, "low": 48200.0, "close": 48300.0, "volume": 4901.2},
    {"timestamp": 1641128400000, "open": 48300.0, "high": 48400.0, "low": 48100.0, "close": 48200.0, "volume": 5012.3},
    {"timestamp": 1641132000000, "open": 48200.0, "high": 48300.0, "low": 48000.0, "close": 48100.0, "volume": 5123.4},
    {"timestamp": 1641135600000, "open": 48100.0, "high": 48200.0, "low": 47900.0, "close": 48000.0, "volume": 5234.5},
    {"timestamp": 1641139200000, "open": 48000.0, "high": 48100.0, "low": 47800.0, "close": 47900.0, "volume": 5345.6},
    {"timestamp": 1641142800000, "open": 47900.0, "high": 48000.0, "low": 47700.0, "close": 47800.0, "volume": 5456.7},
    {"timestamp": 1641146400000, "open": 47800.0, "high": 47900.0, "low": 47600.0, "close": 47700.0, "volume": 5567.8},
    {"timestamp": 1641150000000, "open": 47700.0, "high": 47800.0, "low": 47500.0, "close": 47600.0, "volume": 5678.9},
    {"timestamp": 1641153600000, "open": 47600.0, "high": 47700.0, "low": 47400.0, "close": 47500.0, "volume": 5789.0},
    {"timestamp": 1641157200000, "open": 47500.0, "high": 47600.0, "low": 47300.0, "close": 47400.0, "volume": 5890.1},
    {"timestamp": 1641160800000, "open": 47400.0, "high": 47500.0, "low": 47200.0, "close": 47300.0, "volume": 5901.2},
    {"timestamp": 1641164400000, "open": 47300.0, "high": 47400.0, "low": 47100.0, "close": 47200.0, "volume": 6012.3},
    {"timestamp": 1641168000000, "open": 47200.0, "high": 47300.0, "low": 47000.0, "close": 47100.0, "volume": 6123.4},
    {"timestamp": 1641171600000, "open": 47100.0, "high": 47200.0, "low": 46900.0, "close": 47000.0, "volume": 6234.5}
]

# Volatile/choppy market data
BTCUSDT_VOLATILE_MARKET = [
    {"timestamp": 1640995200000, "open": 50000.0, "high": 50500.0, "low": 49500.0, "close": 50200.0, "volume": 2345.6},
    {"timestamp": 1640998800000, "open": 50200.0, "high": 49800.0, "low": 49200.0, "close": 49400.0, "volume": 3456.7},
    {"timestamp": 1641002400000, "open": 49400.0, "high": 50100.0, "low": 49000.0, "close": 49900.0, "volume": 2567.8},
    {"timestamp": 1641006000000, "open": 49900.0, "high": 49600.0, "low": 49100.0, "close": 49300.0, "volume": 3678.9},
    {"timestamp": 1641009600000, "open": 49300.0, "high": 50000.0, "low": 48800.0, "close": 49700.0, "volume": 2789.0},
    {"timestamp": 1641013200000, "open": 49700.0, "high": 49400.0, "low": 48900.0, "close": 49100.0, "volume": 3890.1},
    {"timestamp": 1641016800000, "open": 49100.0, "high": 49800.0, "low": 48600.0, "close": 49500.0, "volume": 2901.2},
    {"timestamp": 1641020400000, "open": 49500.0, "high": 49200.0, "low": 48700.0, "close": 48900.0, "volume": 4012.3},
    {"timestamp": 1641024000000, "open": 48900.0, "high": 49600.0, "low": 48400.0, "close": 49300.0, "volume": 3123.4},
    {"timestamp": 1641027600000, "open": 49300.0, "high": 49000.0, "low": 48500.0, "close": 48700.0, "volume": 4234.5},
    # Continue the volatile pattern...
    {"timestamp": 1641031200000, "open": 48700.0, "high": 49400.0, "low": 48200.0, "close": 49100.0, "volume": 3345.6},
    {"timestamp": 1641034800000, "open": 49100.0, "high": 48800.0, "low": 48300.0, "close": 48500.0, "volume": 4456.7},
    {"timestamp": 1641038400000, "open": 48500.0, "high": 49200.0, "low": 48000.0, "close": 48900.0, "volume": 3567.8},
    {"timestamp": 1641042000000, "open": 48900.0, "high": 48600.0, "low": 48100.0, "close": 48300.0, "volume": 4678.9},
    {"timestamp": 1641045600000, "open": 48300.0, "high": 49000.0, "low": 47800.0, "close": 48700.0, "volume": 3789.0},
    {"timestamp": 1641049200000, "open": 48700.0, "high": 48400.0, "low": 47900.0, "close": 48100.0, "volume": 4890.1},
    {"timestamp": 1641052800000, "open": 48100.0, "high": 48800.0, "low": 47600.0, "close": 48500.0, "volume": 3901.2},
    {"timestamp": 1641056400000, "open": 48500.0, "high": 48200.0, "low": 47700.0, "close": 47900.0, "volume": 5012.3},
    {"timestamp": 1641060000000, "open": 47900.0, "high": 48600.0, "low": 47400.0, "close": 48300.0, "volume": 4123.4},
    {"timestamp": 1641063600000, "open": 48300.0, "high": 48000.0, "low": 47500.0, "close": 47700.0, "volume": 5234.5},
    {"timestamp": 1641067200000, "open": 47700.0, "high": 48400.0, "low": 47200.0, "close": 48100.0, "volume": 4345.6},
    {"timestamp": 1641070800000, "open": 48100.0, "high": 47800.0, "low": 47300.0, "close": 47500.0, "volume": 5456.7},
    {"timestamp": 1641074400000, "open": 47500.0, "high": 48200.0, "low": 47000.0, "close": 47900.0, "volume": 4567.8},
    {"timestamp": 1641078000000, "open": 47900.0, "high": 47600.0, "low": 47100.0, "close": 47300.0, "volume": 5678.9},
    {"timestamp": 1641081600000, "open": 47300.0, "high": 48000.0, "low": 46800.0, "close": 47700.0, "volume": 4789.0},
    {"timestamp": 1641085200000, "open": 47700.0, "high": 47400.0, "low": 46900.0, "close": 47100.0, "volume": 5890.1},
    {"timestamp": 1641088800000, "open": 47100.0, "high": 47800.0, "low": 46600.0, "close": 47500.0, "volume": 4901.2},
    {"timestamp": 1641092400000, "open": 47500.0, "high": 47200.0, "low": 46700.0, "close": 46900.0, "volume": 6012.3},
    {"timestamp": 1641096000000, "open": 46900.0, "high": 47600.0, "low": 46400.0, "close": 47300.0, "volume": 5123.4},
    {"timestamp": 1641099600000, "open": 47300.0, "high": 47000.0, "low": 46500.0, "close": 46700.0, "volume": 6234.5},
    {"timestamp": 1641103200000, "open": 46700.0, "high": 47400.0, "low": 46200.0, "close": 47100.0, "volume": 5345.6},
    {"timestamp": 1641106800000, "open": 47100.0, "high": 46800.0, "low": 46300.0, "close": 46500.0, "volume": 6456.7},
    {"timestamp": 1641110400000, "open": 46500.0, "high": 47200.0, "low": 46000.0, "close": 46900.0, "volume": 5567.8},
    {"timestamp": 1641114000000, "open": 46900.0, "high": 46600.0, "low": 46100.0, "close": 46300.0, "volume": 6678.9},
    {"timestamp": 1641117600000, "open": 46300.0, "high": 47000.0, "low": 45800.0, "close": 46700.0, "volume": 5789.0},
    {"timestamp": 1641121200000, "open": 46700.0, "high": 46400.0, "low": 45900.0, "close": 46100.0, "volume": 6890.1},
    {"timestamp": 1641124800000, "open": 46100.0, "high": 46800.0, "low": 45600.0, "close": 46500.0, "volume": 5901.2},
    {"timestamp": 1641128400000, "open": 46500.0, "high": 46200.0, "low": 45700.0, "close": 45900.0, "volume": 7012.3},
    {"timestamp": 1641132000000, "open": 45900.0, "high": 46600.0, "low": 45400.0, "close": 46300.0, "volume": 6123.4},
    {"timestamp": 1641135600000, "open": 46300.0, "high": 46000.0, "low": 45500.0, "close": 45700.0, "volume": 7234.5},
    {"timestamp": 1641139200000, "open": 45700.0, "high": 46400.0, "low": 45200.0, "close": 46100.0, "volume": 6345.6},
    {"timestamp": 1641142800000, "open": 46100.0, "high": 45800.0, "low": 45300.0, "close": 45500.0, "volume": 7456.7},
    {"timestamp": 1641146400000, "open": 45500.0, "high": 46200.0, "low": 45000.0, "close": 45900.0, "volume": 6567.8},
    {"timestamp": 1641150000000, "open": 45900.0, "high": 45600.0, "low": 45100.0, "close": 45300.0, "volume": 7678.9},
    {"timestamp": 1641153600000, "open": 45300.0, "high": 46000.0, "low": 44800.0, "close": 45700.0, "volume": 6789.0},
    {"timestamp": 1641157200000, "open": 45700.0, "high": 45400.0, "low": 44900.0, "close": 45100.0, "volume": 7890.1},
    {"timestamp": 1641160800000, "open": 45100.0, "high": 45800.0, "low": 44600.0, "close": 45500.0, "volume": 6901.2},
    {"timestamp": 1641164400000, "open": 45500.0, "high": 45200.0, "low": 44700.0, "close": 44900.0, "volume": 8012.3},
    {"timestamp": 1641168000000, "open": 44900.0, "high": 45600.0, "low": 44400.0, "close": 45300.0, "volume": 7123.4},
    {"timestamp": 1641171600000, "open": 45300.0, "high": 45000.0, "low": 44500.0, "close": 44700.0, "volume": 8234.5}
]

# Day trading specific scenarios
DAY_TRADING_MORNING_BREAKOUT = [
    # Pre-market consolidation (low volume)
    {"timestamp": 1641556800000, "open": 48000.0, "high": 48050.0, "low": 47950.0, "close": 48020.0, "volume": 500.0},
    {"timestamp": 1641557700000, "open": 48020.0, "high": 48080.0, "low": 47980.0, "close": 48040.0, "volume": 520.0},
    {"timestamp": 1641558600000, "open": 48040.0, "high": 48090.0, "low": 47990.0, "close": 48030.0, "volume": 480.0},
    {"timestamp": 1641559500000, "open": 48030.0, "high": 48070.0, "low": 47970.0, "close": 48010.0, "volume": 510.0},
    {"timestamp": 1641560400000, "open": 48010.0, "high": 48060.0, "low": 47960.0, "close": 48000.0, "volume": 490.0},
    
    # Market open - volume increases
    {"timestamp": 1641561300000, "open": 48000.0, "high": 48100.0, "low": 47980.0, "close": 48050.0, "volume": 1200.0},
    {"timestamp": 1641562200000, "open": 48050.0, "high": 48120.0, "low": 48020.0, "close": 48080.0, "volume": 1500.0},
    
    # Breakout begins with strong volume
    {"timestamp": 1641563100000, "open": 48080.0, "high": 48200.0, "low": 48070.0, "close": 48180.0, "volume": 2800.0},
    {"timestamp": 1641564000000, "open": 48180.0, "high": 48300.0, "low": 48160.0, "close": 48280.0, "volume": 3200.0},
    {"timestamp": 1641564900000, "open": 48280.0, "high": 48420.0, "low": 48260.0, "close": 48400.0, "volume": 3800.0},
    {"timestamp": 1641565800000, "open": 48400.0, "high": 48550.0, "low": 48380.0, "close": 48520.0, "volume": 4200.0},
    {"timestamp": 1641566700000, "open": 48520.0, "high": 48680.0, "low": 48500.0, "close": 48650.0, "volume": 4500.0},
    {"timestamp": 1641567600000, "open": 48650.0, "high": 48800.0, "low": 48620.0, "close": 48750.0, "volume": 4800.0},
    {"timestamp": 1641568500000, "open": 48750.0, "high": 48900.0, "low": 48720.0, "close": 48850.0, "volume": 5000.0},
    {"timestamp": 1641569400000, "open": 48850.0, "high": 49000.0, "low": 48820.0, "close": 48950.0, "volume": 5200.0},
    
    # Continuation with some pullback
    {"timestamp": 1641570300000, "open": 48950.0, "high": 49050.0, "low": 48880.0, "close": 48920.0, "volume": 3800.0},
    {"timestamp": 1641571200000, "open": 48920.0, "high": 49000.0, "low": 48850.0, "close": 48980.0, "volume": 3500.0},
    {"timestamp": 1641572100000, "open": 48980.0, "high": 49100.0, "low": 48950.0, "close": 49050.0, "volume": 4000.0},
    {"timestamp": 1641573000000, "open": 49050.0, "high": 49200.0, "low": 49020.0, "close": 49150.0, "volume": 4200.0},
    {"timestamp": 1641573900000, "open": 49150.0, "high": 49300.0, "low": 49100.0, "close": 49250.0, "volume": 4500.0},
    {"timestamp": 1641574800000, "open": 49250.0, "high": 49400.0, "low": 49200.0, "close": 49350.0, "volume": 4800.0}
]

# ETH data for multi-symbol testing
ETHUSDT_SAMPLE_DATA = [
    {"timestamp": 1640995200000, "open": 3800.0, "high": 3850.0, "low": 3750.0, "close": 3820.0, "volume": 234.5},
    {"timestamp": 1640998800000, "open": 3820.0, "high": 3870.0, "low": 3790.0, "close": 3840.0, "volume": 456.7},
    {"timestamp": 1641002400000, "open": 3840.0, "high": 3890.0, "low": 3810.0, "close": 3860.0, "volume": 567.8},
    {"timestamp": 1641006000000, "open": 3860.0, "high": 3910.0, "low": 3830.0, "close": 3880.0, "volume": 678.9},
    {"timestamp": 1641009600000, "open": 3880.0, "high": 3930.0, "low": 3850.0, "close": 3900.0, "volume": 789.0},
    {"timestamp": 1641013200000, "open": 3900.0, "high": 3950.0, "low": 3870.0, "close": 3920.0, "volume": 890.1},
    {"timestamp": 1641016800000, "open": 3920.0, "high": 3970.0, "low": 3890.0, "close": 3940.0, "volume": 901.2},
    {"timestamp": 1641020400000, "open": 3940.0, "high": 3990.0, "low": 3910.0, "close": 3960.0, "volume": 1012.3},
    {"timestamp": 1641024000000, "open": 3960.0, "high": 4010.0, "low": 3930.0, "close": 3980.0, "volume": 1123.4},
    {"timestamp": 1641027600000, "open": 3980.0, "high": 4030.0, "low": 3950.0, "close": 4000.0, "volume": 1234.5},
    {"timestamp": 1641031200000, "open": 4000.0, "high": 4050.0, "low": 3970.0, "close": 4020.0, "volume": 1345.6},
    {"timestamp": 1641034800000, "open": 4020.0, "high": 4070.0, "low": 3990.0, "close": 4040.0, "volume": 1456.7},
    {"timestamp": 1641038400000, "open": 4040.0, "high": 4090.0, "low": 4010.0, "close": 4060.0, "volume": 1567.8},
    {"timestamp": 1641042000000, "open": 4060.0, "high": 4110.0, "low": 4030.0, "close": 4080.0, "volume": 1678.9},
    {"timestamp": 1641045600000, "open": 4080.0, "high": 4130.0, "low": 4050.0, "close": 4100.0, "volume": 1789.0},
    {"timestamp": 1641049200000, "open": 4100.0, "high": 4150.0, "low": 4070.0, "close": 4120.0, "volume": 1890.1},
    {"timestamp": 1641052800000, "open": 4120.0, "high": 4170.0, "low": 4090.0, "close": 4140.0, "volume": 1901.2},
    {"timestamp": 1641056400000, "open": 4140.0, "high": 4190.0, "low": 4110.0, "close": 4160.0, "volume": 2012.3},
    {"timestamp": 1641060000000, "open": 4160.0, "high": 4210.0, "low": 4130.0, "close": 4180.0, "volume": 2123.4},
    {"timestamp": 1641063600000, "open": 4180.0, "high": 4230.0, "low": 4150.0, "close": 4200.0, "volume": 2234.5},
    {"timestamp": 1641067200000, "open": 4200.0, "high": 4250.0, "low": 4170.0, "close": 4220.0, "volume": 2345.6},
    {"timestamp": 1641070800000, "open": 4220.0, "high": 4270.0, "low": 4190.0, "close": 4240.0, "volume": 2456.7},
    {"timestamp": 1641074400000, "open": 4240.0, "high": 4290.0, "low": 4210.0, "close": 4260.0, "volume": 2567.8},
    {"timestamp": 1641078000000, "open": 4260.0, "high": 4310.0, "low": 4230.0, "close": 4280.0, "volume": 2678.9},
    {"timestamp": 1641081600000, "open": 4280.0, "high": 4330.0, "low": 4250.0, "close": 4300.0, "volume": 2789.0},
    {"timestamp": 1641085200000, "open": 4300.0, "high": 4350.0, "low": 4270.0, "close": 4320.0, "volume": 2890.1},
    {"timestamp": 1641088800000, "open": 4320.0, "high": 4370.0, "low": 4290.0, "close": 4340.0, "volume": 2901.2},
    {"timestamp": 1641092400000, "open": 4340.0, "high": 4390.0, "low": 4310.0, "close": 4360.0, "volume": 3012.3},
    {"timestamp": 1641096000000, "open": 4360.0, "high": 4410.0, "low": 4330.0, "close": 4380.0, "volume": 3123.4},
    {"timestamp": 1641099600000, "open": 4380.0, "high": 4430.0, "low": 4350.0, "close": 4400.0, "volume": 3234.5},
    {"timestamp": 1641103200000, "open": 4400.0, "high": 4450.0, "low": 4370.0, "close": 4420.0, "volume": 3345.6},
    {"timestamp": 1641106800000, "open": 4420.0, "high": 4470.0, "low": 4390.0, "close": 4440.0, "volume": 3456.7},
    {"timestamp": 1641110400000, "open": 4440.0, "high": 4490.0, "low": 4410.0, "close": 4460.0, "volume": 3567.8},
    {"timestamp": 1641114000000, "open": 4460.0, "high": 4510.0, "low": 4430.0, "close": 4480.0, "volume": 3678.9},
    {"timestamp": 1641117600000, "open": 4480.0, "high": 4530.0, "low": 4450.0, "close": 4500.0, "volume": 3789.0},
    {"timestamp": 1641121200000, "open": 4500.0, "high": 4550.0, "low": 4470.0, "close": 4520.0, "volume": 3890.1},
    {"timestamp": 1641124800000, "open": 4520.0, "high": 4570.0, "low": 4490.0, "close": 4540.0, "volume": 3901.2},
    {"timestamp": 1641128400000, "open": 4540.0, "high": 4590.0, "low": 4510.0, "close": 4560.0, "volume": 4012.3},
    {"timestamp": 1641132000000, "open": 4560.0, "high": 4610.0, "low": 4530.0, "close": 4580.0, "volume": 4123.4},
    {"timestamp": 1641135600000, "open": 4580.0, "high": 4630.0, "low": 4550.0, "close": 4600.0, "volume": 4234.5},
    {"timestamp": 1641139200000, "open": 4600.0, "high": 4650.0, "low": 4570.0, "close": 4620.0, "volume": 4345.6},
    {"timestamp": 1641142800000, "open": 4620.0, "high": 4670.0, "low": 4590.0, "close": 4640.0, "volume": 4456.7},
    {"timestamp": 1641146400000, "open": 4640.0, "high": 4690.0, "low": 4610.0, "close": 4660.0, "volume": 4567.8},
    {"timestamp": 1641150000000, "open": 4660.0, "high": 4710.0, "low": 4630.0, "close": 4680.0, "volume": 4678.9},
    {"timestamp": 1641153600000, "open": 4680.0, "high": 4730.0, "low": 4650.0, "close": 4700.0, "volume": 4789.0},
    {"timestamp": 1641157200000, "open": 4700.0, "high": 4750.0, "low": 4670.0, "close": 4720.0, "volume": 4890.1},
    {"timestamp": 1641160800000, "open": 4720.0, "high": 4770.0, "low": 4690.0, "close": 4740.0, "volume": 4901.2},
    {"timestamp": 1641164400000, "open": 4740.0, "high": 4790.0, "low": 4710.0, "close": 4760.0, "volume": 5012.3},
    {"timestamp": 1641168000000, "open": 4760.0, "high": 4810.0, "low": 4730.0, "close": 4780.0, "volume": 5123.4},
    {"timestamp": 1641171600000, "open": 4780.0, "high": 4830.0, "low": 4750.0, "close": 4800.0, "volume": 5234.5}
]


def get_historical_snapshot(scenario: str) -> List[Dict]:
    """
    Get historical market data snapshot for testing
    
    Args:
        scenario: One of 'bullish_crossover', 'bearish_crossover', 'volatile_market',
                 'morning_breakout', 'eth_sample'
    
    Returns:
        List of OHLCV data dictionaries
    """
    snapshots = {
        'bullish_crossover': BTCUSDT_BULLISH_CROSSOVER,
        'bearish_crossover': BTCUSDT_BEARISH_CROSSOVER,
        'volatile_market': BTCUSDT_VOLATILE_MARKET,
        'morning_breakout': DAY_TRADING_MORNING_BREAKOUT,
        'eth_sample': ETHUSDT_SAMPLE_DATA
    }
    
    return snapshots.get(scenario, BTCUSDT_BULLISH_CROSSOVER)


def get_multi_symbol_snapshots() -> Dict[str, List[Dict]]:
    """Get historical snapshots for multiple symbols"""
    return {
        'BTCUSDT': BTCUSDT_BULLISH_CROSSOVER,
        'ETHUSDT': ETHUSDT_SAMPLE_DATA
    }