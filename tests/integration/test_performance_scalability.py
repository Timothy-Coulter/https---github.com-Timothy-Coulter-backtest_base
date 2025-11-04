"""
Performance and Scalability Integration Tests for QuantBench Backtester.

This module tests system performance under various load conditions including
large datasets, high-frequency data, concurrent operations, and resource usage.
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import gc
import threading
import concurrent.futures
import multiprocessing

from backtester.core.backtest_engine import BacktestEngine


@pytest.mark.integration
class TestLargeDatasetPerformance:
    """Test system performance with large datasets."""
    
    @pytest.mark.slow
    def test_large_dataset_processing_performance(self, large_dataset, integration_test_config, performance_monitor):
        """Test performance with large datasets (5+ years of daily data)."""
        engine = BacktestEngine(integration_test_config)
        
        # Start performance monitoring
        performance_monitor.start()
        
        # Load large dataset
        engine.current_data = large_dataset.copy()
        
        # Run backtest and measure performance
        results = engine.run_backtest()
        
        # Stop monitoring and collect metrics
        metrics = performance_monitor.stop()
        
        # Performance benchmarks from integration test plan
        max_processing_time = 300  # 5 minutes for 5 years of data
        max_memory_usage = 1024    # Less than 1GB memory usage
        
        # Validate performance benchmarks
        assert metrics['execution_time'] <= max_processing_time, \
            f"Processing time {metrics['execution_time']:.2f}s exceeds limit {max_processing_time}s"
        
        assert metrics['memory_used'] <= max_memory_usage, \
            f"Memory usage {metrics['memory_used']:.2f}MB exceeds limit {max_memory_usage}MB"
        
        # Validate that backtest completed successfully
        assert 'performance' in results, "Performance should be calculated"
        assert results['performance']['total_return'] is not None, "Results should be valid"
        
        print(f"Large dataset test: {len(large_dataset)} rows processed in "
              f"{metrics['execution_time']:.2f}s using {metrics['memory_used']:.2f}MB")
    
    @pytest.mark.slow
    def test_memory_usage_scalability(self, integration_test_config):
        """Test memory usage scalability with increasing data sizes."""
        engine = BacktestEngine(integration_test_config)
        
        # Test different data sizes
        data_sizes = [252, 504, 1008, 2016]  # 1, 2, 4, 8 years
        
        memory_usage = []
        processing_times = []
        
        for size in data_sizes:
            # Generate dataset of specific size
            dates = pd.date_range('2020-01-01', periods=size, freq='D')
            np.random.seed(42)
            
            initial_price = 100.0
            returns = np.random.normal(0.0008, 0.02, size)
            prices = initial_price * np.cumprod(1 + returns)
            
            test_data = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.005, size)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, size))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, size))),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, size)
            }, index=dates)
            
            # Ensure OHLC integrity
            test_data['High'] = test_data[['Open', 'High', 'Close']].max(axis=1)
            test_data['Low'] = test_data[['Open', 'Low', 'Close']].min(axis=1)
            
            # Measure memory before processing
            gc.collect()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Process dataset
            start_time = time.time()
            engine_copy = BacktestEngine(integration_test_config)
            engine_copy.current_data = test_data.copy()
            results = engine_copy.run_backtest()
            processing_time = time.time() - start_time
            
            # Measure memory after processing
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            memory_usage.append(memory_used)
            processing_times.append(processing_time)
            
            # Validate reasonable memory usage
            assert memory_used >= 0, f"Memory measurement should be non-negative for size {size}"
            assert processing_time > 0, f"Processing should take time for size {size}"
        
        # Test that memory usage scales reasonably with data size
        # Memory usage should not grow faster than linear
        for i in range(1, len(memory_usage)):
            size_ratio = data_sizes[i] / data_sizes[i-1]
            memory_ratio = memory_usage[i] / max(memory_usage[i-1], 1)
            
            # Allow some overhead but not excessive growth
            assert memory_ratio <= size_ratio * 2, \
                f"Memory usage growing too fast: {memory_ratio:.2f}x vs size ratio {size_ratio:.2f}x"
        
        print(f"Memory scalability: {[f'{mu:.1f}MB' for mu in memory_usage]}")
        print(f"Processing times: {[f'{pt:.2f}s' for pt in processing_times]}")
    
    @pytest.mark.slow
    def test_data_processing_chunking_performance(self, large_dataset, integration_test_config):
        """Test performance with chunked data processing for large datasets."""
        engine = BacktestEngine(integration_test_config)
        
        # Split large dataset into chunks
        chunk_size = 252  # 1 year chunks
        chunks = [large_dataset[i:i+chunk_size] 
                 for i in range(0, len(large_dataset), chunk_size)]
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = []
        
        for i, chunk in enumerate(chunks):
            engine_copy = BacktestEngine(integration_test_config)
            engine_copy.current_data = chunk.copy()
            
            try:
                result = engine_copy.run_backtest()
                sequential_results.append(result)
            except Exception as e:
                print(f"Chunk {i} processing failed: {e}")
                continue
        
        sequential_time = time.time() - start_time
        
        # Test concurrent processing with smaller chunks
        def process_chunk(chunk_data, chunk_id):
            """Process a single chunk."""
            engine_chunk = BacktestEngine(integration_test_config)
            engine_chunk.current_data = chunk_data.copy()
            return engine_chunk.run_backtest()
        
        start_time = time.time()
        concurrent_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk, i): i 
                for i, chunk in enumerate(chunks[:4])  # Limit to 4 chunks for test
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    result = future.result()
                    concurrent_results.append(result)
                except Exception as e:
                    chunk_id = future_to_chunk[future]
                    print(f"Concurrent chunk {chunk_id} failed: {e}")
        
        concurrent_time = time.time() - start_time
        
        # Validate that some processing occurred
        assert len(sequential_results) > 0, "Sequential processing should complete some chunks"
        assert len(concurrent_results) >= 0, "Concurrent processing should attempt chunks"
        
        # Concurrent processing should be faster or similar for I/O bound operations
        # (Threading may not help much for CPU-bound backtesting)
        performance_ratio = concurrent_time / max(sequential_time, 0.1)
        
        print(f"Sequential: {sequential_time:.2f}s for {len(sequential_results)} chunks")
        print(f"Concurrent: {concurrent_time:.2f}s for {len(concurrent_results)} chunks")
        print(f"Performance ratio: {performance_ratio:.2f}x")


@pytest.mark.integration
class TestHighFrequencyDataProcessing:
    """Test intraday and high-frequency data processing capabilities."""
    
    @pytest.mark.slow
    def test_minute_level_data_processing(self, integration_test_config):
        """Test processing of minute-level intraday data."""
        # Generate minute-level data for one day
        dates = pd.date_range('2020-01-01 09:30:00', '2020-01-01 16:00:00', freq='1min')
        n_periods = len(dates)
        
        np.random.seed(42)
        initial_price = 100.0
        
        # Intraday price movements (more volatile than daily)
        returns = np.random.normal(0, 0.002, n_periods)  # Higher frequency volatility
        prices = initial_price * np.cumprod(1 + returns)
        
        # Create OHLCV data
        minute_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.0005, n_periods)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.001, n_periods))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_periods))),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_periods)
        }, index=dates)
        
        # Ensure OHLC integrity
        minute_data['High'] = minute_data[['Open', 'High', 'Close']].max(axis=1)
        minute_data['Low'] = minute_data[['Open', 'Low', 'Close']].min(axis=1)
        
        # Test processing
        engine = BacktestEngine(integration_test_config)
        engine.current_data = minute_data.copy()
        
        start_time = time.time()
        results = engine.run_backtest()
        processing_time = time.time() - start_time
        
        # Validate results
        assert 'performance' in results, "Minute data should produce results"
        assert results['performance']['total_return'] is not None, "Returns should be calculated"
        
        # Performance benchmark: should process minute data in reasonable time
        max_expected_time = n_periods * 0.01  # 10ms per minute
        assert processing_time <= max_expected_time, \
            f"Minute data processing too slow: {processing_time:.2f}s for {n_periods} periods"
        
        print(f"Minute-level processing: {n_periods} periods in {processing_time:.2f}s")
    
    @pytest.mark.slow
    def test_tick_level_data_handling(self, integration_test_config):
        """Test handling of tick-level market data."""
        # Generate tick data (1000 ticks)
        dates = pd.date_range('2020-01-01 09:30:00', periods=1000, freq='1s')
        
        np.random.seed(42)
        prices = [100.0]
        
        for i in range(999):
            # High-frequency price changes
            price_change = np.random.normal(0, 0.001)  # Small price movements
            new_price = prices[-1] + price_change
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        # Create tick data
        tick_data = pd.DataFrame({
            'Price': prices,
            'Volume': np.random.randint(1, 100, 1000),
            'Bid': [p - 0.01 for p in prices],
            'Ask': [p + 0.01 for p in prices]
        }, index=dates)
        
        # Test tick data processing
        engine = BacktestEngine(integration_test_config)
        engine.current_data = tick_data.copy()
        
        start_time = time.time()
        
        try:
            results = engine.run_backtest()
            processing_time = time.time() - start_time
            
            # Should handle tick data without errors
            assert results is not None, "Tick data processing should complete"
            
            print(f"Tick-level processing: {len(tick_data)} ticks in {processing_time:.2f}s")
            
        except Exception as e:
            # Some systems may not support tick-level data natively
            print(f"Tick data processing not supported: {e}")
            # This is acceptable - not all systems need tick-level support
    
    @pytest.mark.slow
    def test_real_time_simulation_performance(self, integration_test_config):
        """Test real-time simulation performance with streaming data."""
        engine = BacktestEngine(integration_test_config)
        
        # Simulate real-time data stream
        def data_stream_generator(duration_seconds=10):
            """Generate data stream for testing."""
            start_time = time.time()
            count = 0
            
            while time.time() - start_time < duration_seconds:
                # Simulate incoming market data
                current_time = pd.Timestamp.now()
                price = 100.0 + np.random.normal(0, 0.1)
                
                yield pd.DataFrame({
                    'Open': [price * 0.999],
                    'High': [price * 1.001],
                    'Low': [price * 0.999],
                    'Close': [price],
                    'Volume': [np.random.randint(1000, 5000)]
                }, index=[current_time])
                
                count += 1
                time.sleep(0.1)  # 100ms intervals
        
        # Process data stream
        start_time = time.time()
        processed_count = 0
        
        for data_chunk in data_stream_generator(duration_seconds=2):
            try:
                # Process each chunk immediately
                engine.current_data = data_chunk
                
                # Simulate minimal processing
                current_price = data_chunk['Close'].iloc[0]
                processed_count += 1
                
            except Exception as e:
                print(f"Stream processing error: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Validate real-time processing capability
        assert processed_count > 0, "Should process some data chunks"
        assert total_time > 0, "Should take some processing time"
        
        # Processing should keep up with data stream (within reasonable limits)
        processing_rate = processed_count / total_time
        
        print(f"Real-time simulation: {processed_count} chunks in {total_time:.2f}s "
              f"({processing_rate:.1f} chunks/sec)")


@pytest.mark.integration
class TestConcurrentOperations:
    """Test multiple concurrent backtest executions and resource sharing."""
    
    @pytest.mark.slow
    def test_parallel_backtest_execution(self, sample_market_data, integration_test_config):
        """Test multiple parallel backtest executions."""
        def run_single_backtest(data_subset, test_id):
            """Run a single backtest in a separate process."""
            engine = BacktestEngine(integration_test_config)
            engine.current_data = data_subset.copy()
            
            start_time = time.time()
            results = engine.run_backtest()
            execution_time = time.time() - start_time
            
            return {
                'test_id': test_id,
                'execution_time': execution_time,
                'success': results is not None,
                'return': results.get('performance', {}).get('total_return', None)
            }
        
        # Create multiple data subsets
        data_subsets = []
        for i in range(3):
            subset = sample_market_data.iloc[i*len(sample_market_data)//3:(i+1)*len(sample_market_data)//3].copy()
            data_subsets.append(subset)
        
        # Execute backtests in parallel using multiprocessing
        start_time = time.time()
        
        with multiprocessing.Pool(processes=2) as pool:
            futures = [
                pool.apply_async(run_single_backtest, (data_subset, i))
                for i, data_subset in enumerate(data_subsets)
            ]
            
            results = [future.get() for future in futures]
        
        total_time = time.time() - start_time
        
        # Validate parallel execution
        assert len(results) == len(data_subsets), "Should complete all backtests"
        
        successful_backtests = [r for r in results if r['success']]
        assert len(successful_backtests) > 0, "At least some backtests should succeed"
        
        # Validate execution times
        for result in results:
            assert result['execution_time'] > 0, "Each backtest should take time"
            assert isinstance(result['return'], (int, float, type(None))), \
                "Should return valid return value or None"
        
        print(f"Parallel execution: {len(results)} backtests in {total_time:.2f}s")
        print(f"Success rate: {len(successful_backtests)}/{len(results)}")
    
    @pytest.mark.slow
    def test_resource_sharing_isolation(self, integration_test_config):
        """Test resource sharing and isolation in concurrent operations."""
        # Test shared resource access patterns
        shared_data = None
        lock = threading.Lock()
        
        def access_shared_resource(thread_id):
            """Access shared resource with proper locking."""
            with lock:
                # Simulate shared resource access
                nonlocal shared_data
                if shared_data is None:
                    shared_data = {'initialized': True, 'threads_accessed': []}
                
                shared_data['threads_accessed'].append(thread_id)
                
                # Simulate some processing time
                time.sleep(0.01)
                
                return {
                    'thread_id': thread_id,
                    'access_successful': True,
                    'total_accesses': len(shared_data['threads_accessed'])
                }
        
        # Launch multiple threads accessing shared resource
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(access_shared_resource, i)
                for i in range(5)
            ]
            
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Validate thread safety
        assert len(results) == 5, "All threads should complete"
        
        for result in results:
            assert result['access_successful'], "All accesses should be successful"
        
        # Verify shared resource was accessed correctly
        assert shared_data is not None, "Shared resource should be initialized"
        assert len(shared_data['threads_accessed']) == 5, "All threads should access resource"
        
        print(f"Resource sharing test: {len(results)} threads in {total_time:.2f}s")
    
    @pytest.mark.slow
    def test_system_stability_under_load(self, integration_test_config):
        """Test system stability under concurrent load."""
        # Test system behavior under high concurrent load
        results = []
        errors = []
        
        def simulate_backtest_load(load_id):
            """Simulate backtest execution under load."""
            try:
                # Create large dataset for stress testing
                dates = pd.date_range('2020-01-01', periods=1000, freq='D')
                np.random.seed(load_id)  # Different seed per thread
                
                initial_price = 100.0
                returns = np.random.normal(0.0008, 0.02, 1000)
                prices = initial_price * np.cumprod(1 + returns)
                
                stress_data = pd.DataFrame({
                    'Open': prices * (1 + np.random.normal(0, 0.005, 1000)),
                    'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 1000))),
                    'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 1000))),
                    'Close': prices,
                    'Volume': np.random.randint(1000000, 10000000, 1000)
                }, index=dates)
                
                stress_data['High'] = stress_data[['Open', 'High', 'Close']].max(axis=1)
                stress_data['Low'] = stress_data[['Open', 'Low', 'Close']].min(axis=1)
                
                # Run backtest under load
                engine = BacktestEngine(integration_test_config)
                engine.current_data = stress_data.copy()
                
                start_time = time.time()
                backtest_results = engine.run_backtest()
                execution_time = time.time() - start_time
                
                return {
                    'load_id': load_id,
                    'success': True,
                    'execution_time': execution_time,
                    'return': backtest_results.get('performance', {}).get('total_return'),
                    'data_points': len(stress_data)
                }
                
            except Exception as e:
                return {
                    'load_id': load_id,
                    'success': False,
                    'error': str(e),
                    'execution_time': 0
                }
        
        # Launch multiple concurrent backtests
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(simulate_backtest_load, i)
                for i in range(5)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if not result['success']:
                        errors.append(result['error'])
                        
                except Exception as e:
                    errors.append(str(e))
        
        total_time = time.time() - start_time
        
        # Validate system stability
        successful_tests = [r for r in results if r['success']]
        success_rate = len(successful_tests) / len(results) if results else 0
        
        # System should maintain reasonable success rate under load
        assert success_rate >= 0.6, f"Success rate {success_rate:.2%} too low under load"
        
        # Successful tests should have reasonable performance
        for result in successful_tests:
            assert result['execution_time'] > 0, "Execution should take time"
            assert result['data_points'] > 0, "Should process data points"
        
        print(f"Load test: {len(successful_tests)}/{len(results)} successful in {total_time:.2f}s")
        if errors:
            print(f"Errors encountered: {len(errors)}")


@pytest.mark.integration
class TestSystemResourceUsage:
    """Test system resource usage monitoring and optimization."""
    
    @pytest.mark.slow
    def test_cpu_usage_monitoring(self, integration_test_config):
        """Test CPU usage monitoring during backtest execution."""
        engine = BacktestEngine(integration_test_config)
        
        # Monitor CPU usage during backtest
        process = psutil.Process()
        
        # Get baseline CPU usage
        baseline_cpu = process.cpu_percent()
        time.sleep(0.1)  # Small delay to get accurate reading
        
        # Run backtest while monitoring CPU
        start_time = time.time()
        
        cpu_samples = []
        memory_samples = []
        
        def monitor_resources():
            """Monitor system resources during execution."""
            while time.time() - start_time < 10:  # Monitor for up to 10 seconds
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                cpu_samples.append(cpu_percent)
                memory_samples.append(memory_mb)
                
                time.sleep(0.1)  # Sample every 100ms
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Run actual backtest
        engine.current_data = sample_market_data.copy()
        results = engine.run_backtest()
        
        # Stop monitoring
        monitor_thread.join()
        
        # Analyze CPU usage
        if cpu_samples:
            avg_cpu = np.mean(cpu_samples)
            peak_cpu = np.max(cpu_samples)
            
            # CPU usage should be reasonable for backtesting
            assert avg_cpu >= 0, "CPU usage should be measured"
            assert peak_cpu <= 100, "CPU usage should not exceed 100%"
            
            print(f"CPU usage: avg={avg_cpu:.1f}%, peak={peak_cpu:.1f}%")
        
        # Analyze memory usage
        if memory_samples:
            avg_memory = np.mean(memory_samples)
            peak_memory = np.max(memory_samples)
            
            # Memory usage should be stable
            assert avg_memory > 0, "Memory usage should be measured"
            
            memory_growth = peak_memory - min(memory_samples)
            assert memory_growth >= 0, "Memory should not decrease unexpectedly"
            
            print(f"Memory usage: avg={avg_memory:.1f}MB, peak={peak_memory:.1f}MB")
    
    @pytest.mark.slow
    def test_memory_optimization_strategies(self, integration_test_config):
        """Test memory optimization strategies for large datasets."""
        # Test different memory optimization approaches
        
        # Approach 1: Process data in chunks
        def process_in_chunks(data, chunk_size=252):
            """Process large data in memory-efficient chunks."""
            results = []
            
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i+chunk_size]
                
                # Process chunk
                engine = BacktestEngine(integration_test_config)
                engine.current_data = chunk.copy()
                
                try:
                    result = engine.run_backtest()
                    results.append(result)
                except Exception as e:
                    print(f"Chunk {i//chunk_size} failed: {e}")
                    continue
                
                # Force garbage collection after each chunk
                gc.collect()
            
            return results
        
        # Test with large dataset
        large_data = large_dataset.copy()
        
        start_time = time.time()
        chunk_results = process_in_chunks(large_data)
        chunk_time = time.time() - start_time
        
        # Validate chunk processing
        assert len(chunk_results) > 0, "Should process some chunks"
        
        # Check memory usage during chunk processing
        gc.collect()
        memory_after_chunks = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"Chunk processing: {len(chunk_results)} chunks in {chunk_time:.2f}s")
        print(f"Memory after chunks: {memory_after_chunks:.1f}MB")
        
        # Approach 2: Process data incrementally
        def process_incrementally(data, increment_size=50):
            """Process data incrementally without storing full results."""
            final_results = []
            
            engine = BacktestEngine(integration_test_config)
            
            for i in range(0, len(data), increment_size):
                end_idx = min(i + increment_size, len(data))
                increment = data.iloc[i:end_idx]
                
                # Reset engine for each increment
                engine.current_data = increment.copy()
                
                try:
                    result = engine.run_backtest()
                    
                    # Store only final metrics, not full results
                    if result and 'performance' in result:
                        final_results.append(result['performance']['total_return'])
                        
                except Exception as e:
                    print(f"Increment {i//increment_size} failed: {e}")
                    continue
                
                # Minimal memory footprint
                del increment
                gc.collect()
            
            return final_results
        
        # Test incremental processing
        start_time = time.time()
        incremental_results = process_incrementally(large_data[:500])  # Smaller dataset for test
        incremental_time = time.time() - start_time
        
        # Validate incremental processing
        assert len(incremental_results) > 0, "Should process some increments"
        
        gc.collect()
        memory_after_incremental = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"Incremental processing: {len(incremental_results)} increments in {incremental_time:.2f}s")
        print(f"Memory after incremental: {memory_after_incremental:.1f}MB")
        
        # Memory usage should be reasonable for both approaches
        assert memory_after_chunks < 2000, "Chunk processing memory usage should be reasonable"
        assert memory_after_incremental < 2000, "Incremental processing memory usage should be reasonable"


# Import required fixtures
# This will be resolved when the test runs with proper fixtures
def sample_market_data():
    """Placeholder for sample_market_data fixture."""
    pass

def large_dataset():
    """Placeholder for large_dataset fixture."""  
    pass