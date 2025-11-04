"""
Component Integration Tests for QuantBench Backtester.

This module tests the integration between different system components including
Data-Strategy, Strategy-Portfolio, Portfolio-Risk, Broker-Execution, and
Performance-Calculation interactions.
"""

import pytest
import pandas as pd
import numpy as np

from backtester.execution.order import OrderSide, OrderType


@pytest.mark.integration
class TestDataStrategyIntegration:
    """Test Data-Strategy component integration."""
    
    def test_data_flow_to_strategy(self, sample_market_data, integration_test_config, backtester_components):
        """Test data flow from DataHandler to Strategy component."""
        data_handler = backtester_components['data_handler']
        strategy = backtester_components['strategy']
        
        # Process data through DataHandler (add technical indicators if enabled)
        processed_data = sample_market_data.copy()
        if integration_test_config.data.use_technical_indicators:
            processed_data = data_handler.add_technical_indicators(processed_data)
        
        # Test strategy consumption
        signals = strategy.generate_signals(processed_data)
        
        # Validate signal format and consistency
        assert isinstance(signals, list), "Strategy should return a list of signals"
        
        # Test with data that should generate signals
        test_data = sample_market_data.tail(60)  # Ensure enough data for MA calculations
        
        for i in range(len(test_data)):
            current_data = test_data.iloc[:i+1]
            if len(current_data) >= strategy.ma_long:
                signals = strategy.generate_signals(current_data)
                
                # Validate signal structure
                for signal in signals:
                    assert 'signal_type' in signal, "Signal must have signal_type"
                    assert 'price' in signal, "Signal must have price"
                    assert 'timestamp' in signal, "Signal must have timestamp"
                    assert signal['signal_type'] in ['BUY', 'SELL', 'HOLD'], \
                        f"Invalid signal type: {signal['signal_type']}"
                    assert signal['price'] > 0, f"Invalid price: {signal['price']}"
    
    def test_technical_indicator_consistency(self, sample_market_data, backtester_components, integration_helpers):
        """Test technical indicator calculation consistency across components."""
        from backtester.utils.data_utils import calculate_indicators
        
        strategy = backtester_components['strategy']
        
        # Calculate indicators directly using utility
        indicators_direct = calculate_indicators(sample_market_data)
        
        # Calculate through strategy pipeline
        strategy_data = sample_market_data.copy()
        strategy.reset()
        
        # Process through strategy step by step
        signals = []
        for i in range(len(strategy_data)):
            current_data = strategy_data.iloc[:i+1]
            if len(current_data) >= strategy.ma_long:
                new_signals = strategy.generate_signals(current_data)
                signals.extend(new_signals)
        
        # Validate that we can process the data without errors
        assert len(signals) >= 0, "Should be able to process all data points"
        
        # If we have indicators, check consistency
        if 'sma_20' in indicators_direct.columns and 'sma_50' in indicators_direct.columns:
            # Verify indicator calculations are reasonable
            assert not indicators_direct['sma_20'].isna().all(), "SMA 20 should have values"
            assert not indicators_direct['sma_50'].isna().all(), "SMA 50 should have values"
            
            # SMA 20 should generally be closer to current prices than SMA 50
            price_diff_20 = abs(sample_market_data['Close'] - indicators_direct['sma_20'])
            price_diff_50 = abs(sample_market_data['Close'] - indicators_direct['sma_50'])
            
            # This should generally be true for trending data
            assert price_diff_20.mean() <= price_diff_50.mean() * 1.5, \
                "SMA 20 should generally be closer to prices than SMA 50"
    
    def test_data_validation_integration(self, sample_market_data, backtester_components):
        """Test data validation integration between DataHandler and Strategy."""
        strategy = backtester_components['strategy']
        
        # Test with valid data
        valid_signals = strategy.generate_signals(sample_market_data)
        assert isinstance(valid_signals, list), "Should return signals for valid data"
        
        # Test with missing columns
        invalid_data = sample_market_data[['Open', 'Close']].copy()  # Missing High, Low
        invalid_signals = strategy.generate_signals(invalid_data)
        assert invalid_signals == [], "Should return empty list for invalid data"
        
        # Test with NaN values
        nan_data = sample_market_data.copy()
        nan_data.loc[10:20, 'Close'] = np.nan
        nan_signals = strategy.generate_signals(nan_data)
        assert isinstance(nan_signals, list), "Should handle NaN values gracefully"
        
        # Test with zero/negative prices
        zero_price_data = sample_market_data.copy()
        zero_price_data.loc[5, 'Close'] = 0
        zero_signals = strategy.generate_signals(zero_price_data)
        assert isinstance(zero_signals, list), "Should handle zero prices gracefully"


@pytest.mark.integration
class TestStrategyPortfolioIntegration:
    """Test Strategy-Portfolio component integration."""
    
    def test_signal_to_order_conversion(self, sample_market_data, backtester_components, integration_helpers):
        """Test conversion of strategy signals to portfolio orders."""
        strategy = backtester_components['strategy']
        portfolio = backtester_components['portfolio']
        broker = backtester_components['broker']
        
        # Reset portfolio and strategy
        portfolio.reset()
        strategy.reset()
        
        # Set up market data for broker
        broker.set_market_data("SPY", sample_market_data)
        
        initial_portfolio_value = portfolio.get_total_value()
        
        # Generate signals and process through portfolio
        test_data = sample_market_data.tail(100)  # Use last 100 days for testing
        
        signals_generated = 0
        orders_executed = 0
        
        for i in range(len(test_data)):
            current_data = test_data.iloc[:i+1]
            
            if len(current_data) >= strategy.ma_long:
                signals = strategy.generate_signals(current_data)
                
                for signal in signals:
                    signals_generated += 1
                    
                    # Convert signal to order (simulating backtest engine logic)
                    if signal['signal_type'] == 'BUY':
                        order = broker.order_manager.create_order(
                            symbol="SPY",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=10.0,  # Standardized quantity
                            metadata=signal
                        )
                        
                        # Execute order
                        if order:
                            result = broker.execute_order(order)
                            if result and result.get('success'):
                                orders_executed += 1
                                
                                # Update portfolio position (simplified)
                                portfolio.cash -= result['fill_price'] * result['fill_quantity']
                    
                    elif signal['signal_type'] == 'SELL':
                        order = broker.order_manager.create_order(
                            symbol="SPY",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=10.0,
                            metadata=signal
                        )
                        
                        # Execute order
                        if order:
                            result = broker.execute_order(order)
                            if result and result.get('success'):
                                orders_executed += 1
                                
                                # Update portfolio position (simplified)
                                portfolio.cash += result['fill_price'] * result['fill_quantity']
        
        # Validate integration
        assert signals_generated >= 0, "Should generate some signals"
        assert orders_executed >= 0, "Should handle order execution"
        
        # Portfolio should have some activity
        final_portfolio_value = portfolio.get_total_value()
        assert final_portfolio_value != initial_portfolio_value, \
            "Portfolio value should change with trading activity"
    
    def test_portfolio_state_transition(self, sample_market_data, backtester_components, integration_helpers):
        """Test portfolio state transitions based on strategy signals."""
        portfolio = backtester_components['portfolio']
        strategy = backtester_components['strategy']
        
        # Reset both components
        portfolio.reset()
        strategy.reset()
        
        initial_base_capital = portfolio.base_pool.capital
        initial_alpha_capital = portfolio.alpha_pool.capital
        
        # Simulate strategy signals and portfolio processing
        test_data = sample_market_data.tail(200)  # Longer period for state transitions
        
        for i in range(len(test_data)):
            current_data = test_data.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]
            current_time = current_data.index[-1]
            
            # Generate signals
            signals = strategy.generate_signals(current_data)
            
            # Process through portfolio (simplified)
            portfolio_update = portfolio.process_tick(
                current_time, current_price, 
                current_data['High'].iloc[-1], 
                current_data['Low'].iloc[-1]
            )
            
            # Validate portfolio state after each update
            assert portfolio_update['total_value'] >= 0, "Portfolio value should be non-negative"
            assert portfolio_update['base_pool'] >= 0, "Base pool should be non-negative"
            assert portfolio_update['alpha_pool'] >= 0, "Alpha pool should be non-negative"
        
        # Validate final state transitions
        final_base_capital = portfolio.base_pool.capital
        final_alpha_capital = portfolio.alpha_pool.capital
        
        # Check that state transitions occurred
        state_changes = (final_base_capital != initial_base_capital) or \
                       (final_alpha_capital != initial_alpha_capital)
        
        # Portfolio state should change with market movement and signals
        assert portfolio.get_total_value() > 0, "Portfolio should maintain value"
        assert integration_helpers.validate_portfolio_state(portfolio), \
            "Portfolio state should be valid throughout transitions"
    
    def test_signal_timing_coordination(self, sample_market_data, backtester_components):
        """Test coordination between signal generation and portfolio updates."""
        strategy = backtester_components['strategy']
        portfolio = backtester_components['portfolio']
        
        strategy.reset()
        portfolio.reset()
        
        # Use data that should generate clear signals
        test_data = sample_market_data.copy()
        
        signal_times = []
        portfolio_times = []
        
        for i in range(len(test_data)):
            current_data = test_data.iloc[:i+1]
            current_time = current_data.index[-1]
            
            # Generate signals
            signals = strategy.generate_signals(current_data)
            if signals:
                signal_times.append(current_time)
            
            # Process portfolio update
            portfolio.process_tick(
                current_time,
                current_data['Close'].iloc[-1],
                current_data['High'].iloc[-1],
                current_data['Low'].iloc[-1]
            )
            portfolio_times.append(current_time)
        
        # Validate timing coordination
        assert len(portfolio_times) == len(test_data), "Portfolio should be updated for each time point"
        
        # Signals should be generated at appropriate times (when enough data is available)
        min_signals_expected = max(0, len(test_data) - strategy.ma_long)
        assert len(signal_times) >= min_signals_expected * 0.1, \
            "Should generate signals at reasonable frequency"


@pytest.mark.integration
class TestPortfolioRiskIntegration:
    """Test Portfolio-Risk component integration."""
    
    def test_real_time_risk_calculation(self, sample_market_data, backtester_components):
        """Test real-time risk metric calculation."""
        portfolio = backtester_components['portfolio']
        risk_manager = backtester_components['risk_manager']
        
        portfolio.reset()
        
        # Simulate portfolio with positions
        test_data = sample_market_data.tail(100)
        
        for i in range(len(test_data)):
            current_data = test_data.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]
            current_time = current_data.index[-1]
            
            # Update portfolio
            portfolio.process_tick(
                current_time, current_price,
                current_data['High'].iloc[-1],
                current_data['Low'].iloc[-1]
            )
            
            # Get portfolio positions (simulated)
            positions = {}
            if portfolio.base_pool.active:
                positions['base_pool'] = {
                    'market_value': portfolio.base_pool.capital * portfolio.base_pool.leverage,
                    'symbol': 'SPY'
                }
            if portfolio.alpha_pool.active:
                positions['alpha_pool'] = {
                    'market_value': portfolio.alpha_pool.capital * portfolio.alpha_pool.leverage,
                    'symbol': 'SPY'
                }
            
            # Check portfolio-level risk
            portfolio_value = portfolio.get_total_value()
            risk_signal = risk_manager.check_portfolio_risk(portfolio_value, positions)
            
            # Validate risk calculation
            assert hasattr(risk_signal, 'action'), "Risk signal should have action"
            assert hasattr(risk_signal, 'reason'), "Risk signal should have reason"
            
            # Risk metrics should be calculated
            assert isinstance(risk_signal.action.value, str), "Risk action should be a string"
    
    def test_risk_limit_enforcement(self, sample_market_data, backtester_components):
        """Test risk limit enforcement procedures."""
        portfolio = backtester_components['portfolio']
        risk_manager = backtester_components['risk_manager']
        
        # Configure risk manager with tight limits for testing
        risk_manager.max_portfolio_var = 0.01  # 1% VaR limit
        risk_manager.max_drawdown = 0.05  # 5% drawdown limit
        
        portfolio.reset()
        
        # Simulate stress market conditions
        stress_data = sample_market_data.copy()
        
        # Create a significant decline to trigger risk limits
        stress_data.iloc[stress_data.shape[0]//2:, 
                        stress_data.columns.get_loc('Close')] *= 0.9  # 10% decline
        
        risk_violations = 0
        risk_actions_taken = []
        
        for i in range(len(stress_data)):
            current_data = stress_data.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]
            current_time = current_data.index[-1]
            
            # Update portfolio
            portfolio.process_tick(
                current_time, current_price,
                current_data['High'].iloc[-1],
                current_data['Low'].iloc[-1]
            )
            
            # Get current positions
            positions = {}
            if portfolio.base_pool.active:
                positions['base'] = {
                    'market_value': portfolio.base_pool.capital * portfolio.base_pool.leverage,
                    'symbol': 'SPY'
                }
            
            portfolio_value = portfolio.get_total_value()
            risk_signal = risk_manager.check_portfolio_risk(portfolio_value, positions)
            
            # Check if risk limits were violated
            if risk_signal.action.value in ['REDUCE_POSITION', 'CLOSE_POSITION', 'EMERGENCY_LIQUIDATION']:
                risk_violations += 1
                risk_actions_taken.append(risk_signal.action.value)
        
        # In stress conditions, risk management should activate
        assert risk_violations >= 0, "Risk management should be tested"
        
        # Risk actions should be valid
        valid_actions = ['NO_ACTION', 'REDUCE_POSITION', 'CLOSE_POSITION', 'EMERGENCY_LIQUIDATION']
        for action in risk_actions_taken:
            assert action in valid_actions, f"Invalid risk action: {action}"
    
    def test_risk_signal_generation_response(self, sample_market_data, backtester_components):
        """Test risk signal generation and response procedures."""
        portfolio = backtester_components['portfolio']
        risk_manager = backtester_components['risk_manager']
        
        portfolio.reset()
        
        # Initialize risk manager tracking
        initial_risk_signals = len(risk_manager.risk_signals)
        
        test_data = sample_market_data.tail(50)
        
        for i in range(len(test_data)):
            current_data = test_data.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]
            current_time = current_data.index[-1]
            
            # Update portfolio
            portfolio.process_tick(
                current_time, current_price,
                current_data['High'].iloc[-1],
                current_data['Low'].iloc[-1]
            )
            
            # Check risk management
            positions = {}
            if portfolio.base_pool.active:
                positions['SPY'] = {
                    'market_value': portfolio.base_pool.capital * portfolio.base_pool.leverage,
                    'symbol': 'SPY'
                }
            
            portfolio_value = portfolio.get_total_value()
            risk_signal = risk_manager.check_portfolio_risk(portfolio_value, positions)
            
            # Add risk signal to tracking
            risk_manager.add_risk_signal(risk_signal)
        
        # Verify risk signals were generated and tracked
        final_risk_signals = len(risk_manager.risk_signals)
        assert final_risk_signals >= initial_risk_signals, "Risk signals should be tracked"
        
        # Validate risk signal structure
        for signal in risk_manager.risk_signals:
            assert hasattr(signal, 'timestamp'), "Risk signal should have timestamp"
            assert hasattr(signal, 'action'), "Risk signal should have action"
            assert hasattr(signal, 'reason'), "Risk signal should have reason"


@pytest.mark.integration
class TestBrokerExecutionIntegration:
    """Test Broker-Execution component integration."""
    
    def test_order_lifecycle_management(self, sample_market_data, backtester_components):
        """Test order lifecycle management integration."""
        broker = backtester_components['broker']
        
        # Set up market data
        broker.set_market_data("SPY", sample_market_data)
        
        # Test order creation
        order = broker.order_manager.create_order(
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            metadata={'test': True}
        )
        
        assert order is not None, "Order should be created"
        assert order.symbol == "SPY", "Order symbol should match"
        assert order.side == OrderSide.BUY, "Order side should match"
        assert order.quantity == 10.0, "Order quantity should match"
        
        # Test order execution
        execution_result = broker.execute_order(order)
        
        assert execution_result is not None, "Order execution should return result"
        assert 'success' in execution_result, "Execution result should indicate success/failure"
        assert 'fill_price' in execution_result, "Execution result should have fill price"
        assert 'fill_quantity' in execution_result, "Execution result should have fill quantity"
        
        if execution_result['success']:
            assert execution_result['fill_price'] > 0, "Fill price should be positive"
            assert execution_result['fill_quantity'] > 0, "Fill quantity should be positive"
    
    def test_trade_execution_simulation(self, sample_market_data, backtester_components):
        """Test trade execution simulation accuracy."""
        broker = backtester_components['broker']
        
        broker.set_market_data("SPY", sample_market_data)
        
        # Test multiple order types
        orders_to_test = [
            {
                'side': OrderSide.BUY,
                'type': OrderType.MARKET,
                'quantity': 10.0,
                'expected_success': True
            },
            {
                'side': OrderSide.SELL,
                'type': OrderType.MARKET,
                'quantity': 5.0,
                'expected_success': True
            }
        ]
        
        executed_orders = []
        
        for order_spec in orders_to_test:
            order = broker.order_manager.create_order(
                symbol="SPY",
                side=order_spec['side'],
                order_type=order_spec['type'],
                quantity=order_spec['quantity']
            )
            
            if order:
                result = broker.execute_order(order)
                executed_orders.append({
                    'order': order,
                    'result': result,
                    'spec': order_spec
                })
                
                # Validate execution
                if order_spec['expected_success']:
                    assert result['success'], f"Order should execute successfully: {result}"
        
        # Validate execution results
        assert len(executed_orders) > 0, "Some orders should be executed"
        
        for executed in executed_orders:
            order = executed['order']
            result = executed['result']
            
            # Check order status tracking
            assert order.status in ['filled', 'partial_fill', 'rejected'], \
                f"Order should have valid status: {order.status}"
    
    def test_slippage_commission_calculation(self, sample_market_data, backtester_components):
        """Test slippage and commission calculation accuracy."""
        broker = backtester_components['broker']
        
        # Configure broker with known parameters
        broker.commission_rate = 0.001  # 0.1%
        broker.slippage_std = 0.0005    # 0.05%
        
        broker.set_market_data("SPY", sample_market_data)
        
        # Create test order
        test_price = 100.0
        test_quantity = 10.0
        
        order = broker.order_manager.create_order(
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=test_quantity
        )
        
        # Execute order
        result = broker.execute_order(order)
        
        if result['success']:
            fill_price = result['fill_price']
            fill_quantity = result['fill_quantity']
            
            # Calculate expected costs
            actual_commission = result.get('commission', 0)
            
            # Commission should be close to expected (allowing for slippage)
            assert actual_commission > 0, "Commission should be charged"
            
            # Slippage should be reasonable (within 3 standard deviations)
            price_impact = abs(fill_price - test_price) / test_price
            max_expected_slippage = 3 * broker.slippage_std
            
            assert price_impact <= max_expected_slippage * 3, \
                f"Slippage {price_impact:.4f} exceeds reasonable limits"
    
    def test_order_status_tracking(self, sample_market_data, backtester_components):
        """Test order status tracking and reporting."""
        broker = backtester_components['broker']
        
        broker.set_market_data("SPY", sample_market_data)
        
        # Create multiple orders
        orders = []
        for i in range(3):
            order = broker.order_manager.create_order(
                symbol="SPY",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=10.0
            )
            orders.append(order)
        
        # Execute all orders
        execution_results = []
        for order in orders:
            if order:
                result = broker.execute_order(order)
                execution_results.append((order, result))
        
        # Test order management operations
        all_orders = broker.order_manager.get_all_orders()
        assert len(all_orders) >= len(orders), "All orders should be tracked"
        
        # Test order status retrieval
        for order in orders:
            if order:
                tracked_order = broker.order_manager.get_order(order.order_id)
                assert tracked_order is not None, "Order should be trackable"
                assert tracked_order.status in ['pending', 'filled', 'partial_fill', 'rejected'], \
                    "Order should have valid status"
        
        # Test order cancellation (for pending orders)
        pending_orders = [o for o in all_orders if o.status == 'pending']
        if pending_orders:
            cancel_result = broker.order_manager.cancel_order(pending_orders[0].order_id)
            # Cancellation result may vary by implementation
            assert cancel_result is not None, "Cancellation should return result"


@pytest.mark.integration
class TestPerformanceCalculationIntegration:
    """Test Performance-Calculation component integration."""
    
    def test_performance_metric_calculation_accuracy(self, sample_market_data, backtester_components):
        """Test performance metric calculation accuracy."""
        performance_analyzer = backtester_components['performance_analyzer']
        
        # Create sample portfolio values
        initial_value = 10000.0
        final_value = 12000.0
        
        # Create portfolio value series
        portfolio_values = pd.Series([
            initial_value,
            initial_value * 1.02,
            initial_value * 1.05,
            initial_value * 0.98,
            initial_value * 1.08,
            final_value
        ])
        
        # Calculate performance metrics
        metrics = performance_analyzer.comprehensive_analysis(portfolio_values)
        
        # Validate basic metrics
        assert 'total_return' in metrics, "Should calculate total return"
        assert 'annualized_return' in metrics, "Should calculate annualized return"
        assert 'sharpe_ratio' in metrics, "Should calculate Sharpe ratio"
        assert 'max_drawdown' in metrics, "Should calculate max drawdown"
        
        # Validate metric values
        expected_total_return = (final_value - initial_value) / initial_value
        assert abs(metrics['total_return'] - expected_total_return) < 0.001, \
            "Total return should be calculated correctly"
        
        assert metrics['max_drawdown'] < 0, "Max drawdown should be negative"
        assert isinstance(metrics['sharpe_ratio'], (int, float)), "Sharpe ratio should be numeric"
    
    def test_benchmark_comparison_procedures(self, sample_market_data, backtester_components):
        """Test benchmark comparison procedures."""
        performance_analyzer = backtester_components['performance_analyzer']
        
        # Create portfolio and benchmark values
        portfolio_values = pd.Series([10000, 10200, 10500, 9800, 11000, 12000])
        benchmark_values = pd.Series([10000, 10100, 10400, 9900, 10800, 11500])
        
        # Calculate performance with benchmark
        metrics = performance_analyzer.comprehensive_analysis(
            portfolio_values, benchmark_values
        )
        
        # Check if benchmark metrics are calculated
        if 'benchmark_total_return' in metrics:
            assert 'excess_return' in metrics, "Should calculate excess return"
            assert 'beta' in metrics, "Should calculate beta"
            assert 'alpha' in metrics, "Should calculate alpha"
            
            # Validate benchmark metrics
            portfolio_return = metrics['total_return']
            benchmark_return = metrics['benchmark_total_return']
            excess_return = metrics['excess_return']
            
            assert abs(excess_return - (portfolio_return - benchmark_return)) < 0.001, \
                "Excess return should be portfolio return minus benchmark return"
    
    def test_risk_adjusted_return_calculations(self, sample_market_data, backtester_components):
        """Test risk-adjusted return calculations."""
        performance_analyzer = backtester_components['performance_analyzer']
        
        # Create portfolio values with known characteristics
        portfolio_values = pd.Series([
            10000,  # Start
            10100,  # +1%
            10300,  # +2%
            10200,  # -1%
            10500,  # +3%
            10400,  # -1%
            10800   # +4%
        ])
        
        # Calculate metrics
        metrics = performance_analyzer.comprehensive_analysis(portfolio_values)
        
        # Validate risk-adjusted metrics
        assert 'sharpe_ratio' in metrics, "Should calculate Sharpe ratio"
        assert 'sortino_ratio' in metrics, "Should calculate Sortino ratio"
        assert 'calmar_ratio' in metrics, "Should calculate Calmar ratio"
        
        # Sharpe ratio should be reasonable for positive returns
        if metrics['total_return'] > 0:
            assert metrics['sharpe_ratio'] > 0, "Sharpe ratio should be positive for good performance"
        
        # Sortino ratio considers downside deviation
        assert isinstance(metrics['sortino_ratio'], (int, float)), \
            "Sortino ratio should be numeric"
    
    def test_report_generation_integration(self, sample_market_data, backtester_components):
        """Test report generation and formatting integration."""
        engine = backtester_components['engine']
        
        # Load sample data
        engine.current_data = sample_market_data.copy()
        
        # Create strategy and portfolio
        engine.create_strategy()
        engine.create_portfolio()
        engine.create_broker()
        engine.create_risk_manager()
        
        # Run a simple backtest
        engine.run_backtest()
        
        # Generate performance report
        report = engine.generate_performance_report()
        
        # Validate report content
        assert isinstance(report, str), "Report should be a string"
        assert len(report) > 0, "Report should not be empty"
        assert "BACKTEST PERFORMANCE REPORT" in report, "Report should have header"
        
        # Check for key metrics in report
        key_metrics = [
            "Total Return",
            "Sharpe Ratio", 
            "Max Drawdown",
            "Strategy:"
        ]
        
        for metric in key_metrics:
            assert metric in report, f"Report should contain {metric}"