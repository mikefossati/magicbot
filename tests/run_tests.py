"""
Test runner script for comprehensive trading strategy tests
Provides different test execution modes for development and CI/CD
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")
    print("   Install with: pip install python-dotenv")
except FileNotFoundError:
    print("⚠️  .env file not found. Using system environment variables.")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests with mocked data"""
    print("🧪 Running Unit Tests (Fast)")
    print("=" * 50)
    
    cmd = ["python", "-m", "pytest", "tests/unit/", "-v" if verbose else "-q"]
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add performance markers
    cmd.extend(["-m", "not slow"])
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"\n⏱️  Unit tests completed in {end_time - start_time:.2f} seconds")
    return result.returncode == 0


def run_integration_tests(verbose=False):
    """Run integration tests with real Binance testnet"""
    print("\n🌐 Running Integration Tests (Slow)")
    print("=" * 50)
    print("⚠️  Note: Requires valid Binance API credentials")
    print("⚠️  Tests will use real network calls to testnet")
    
    # Check for API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Missing API credentials in .env file")
        print("   Required variables:")
        print("   - BINANCE_API_KEY=your_testnet_api_key")
        print("   - BINANCE_SECRET=your_testnet_secret")
        print("   Create a .env file in the project root with these values")
        return False
    
    print(f"✅ Found API credentials (Key: {api_key[:8]}...)")
    
    cmd = ["python", "-m", "pytest", "tests/integration/", "-v" if verbose else "-q"]
    cmd.extend(["-m", "integration"])
    cmd.extend(["--tb=short"])  # Shorter traceback for cleaner output
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"\n⏱️  Integration tests completed in {end_time - start_time:.2f} seconds")
    return result.returncode == 0


def run_performance_tests(verbose=False):
    """Run performance and latency tests"""
    print("\n⚡ Running Performance Tests")
    print("=" * 50)
    
    cmd = ["python", "-m", "pytest", "tests/integration/", "-v" if verbose else "-q"]
    cmd.extend(["-k", "latency or performance or concurrent"])
    cmd.extend(["--tb=short"])
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"\n⏱️  Performance tests completed in {end_time - start_time:.2f} seconds")
    return result.returncode == 0


def run_strategy_specific_tests(strategy_name, verbose=False):
    """Run tests for a specific strategy"""
    print(f"\n🎯 Running Tests for {strategy_name}")
    print("=" * 50)
    
    cmd = ["python", "-m", "pytest", f"tests/unit/strategies/test_{strategy_name}.py", "-v" if verbose else "-q"]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"\n⏱️  {strategy_name} tests completed in {end_time - start_time:.2f} seconds")
    return result.returncode == 0


def run_day_trading_scenarios():
    """Run comprehensive day trading scenario tests"""
    print("\n📈 Running Day Trading Scenario Tests")
    print("=" * 50)
    
    cmd = ["python", "-m", "pytest", "tests/unit/strategies/test_day_trading_strategy.py", "-v"]
    cmd.extend(["-k", "scenario"])
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"\n⏱️  Day trading scenarios completed in {end_time - start_time:.2f} seconds")
    return result.returncode == 0


def run_all_tests(verbose=False, coverage=False, skip_integration=False):
    """Run complete test suite"""
    print("🚀 Running Complete Test Suite")
    print("=" * 80)
    
    total_start = time.time()
    results = []
    
    # Unit tests (always run)
    results.append(run_unit_tests(verbose, coverage))
    
    # Integration tests (optional)
    if not skip_integration:
        results.append(run_integration_tests(verbose))
        results.append(run_performance_tests(verbose))
    else:
        print("\n⏭️  Skipping integration tests")
    
    total_end = time.time()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    if all(results):
        print("✅ All tests PASSED")
        status = "PASSED"
    else:
        print("❌ Some tests FAILED")
        status = "FAILED"
    
    print(f"⏱️  Total execution time: {total_end - total_start:.2f} seconds")
    print(f"📈 Test result: {status}")
    
    return all(results)


def main():
    parser = argparse.ArgumentParser(description="Run trading strategy tests")
    parser.add_argument("--mode", choices=[
        "unit", "integration", "performance", "all", "day-trading"
    ], default="unit", help="Test mode to run")
    
    parser.add_argument("--strategy", help="Run tests for specific strategy")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent.parent)
    
    success = False
    
    if args.strategy:
        success = run_strategy_specific_tests(args.strategy, args.verbose)
    elif args.mode == "unit":
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.mode == "integration":
        success = run_integration_tests(args.verbose)
    elif args.mode == "performance":
        success = run_performance_tests(args.verbose)
    elif args.mode == "day-trading":
        success = run_day_trading_scenarios()
    elif args.mode == "all":
        success = run_all_tests(args.verbose, args.coverage, args.skip_integration)
    
    if success:
        print("\n🎉 Test execution completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test execution failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()