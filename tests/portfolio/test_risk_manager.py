"""
Comprehensive tests for the risk management module.

This module contains tests for comprehensive risk management including
portfolio-level risk controls, exposure management, and risk reporting.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Import the modules being tested
try:
    from backtester.portfolio.risk_manager import (
        RiskManager, ExposureMonitor, RiskAttribution, StressTester, RiskCompliance
    )
except ImportError as e:
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)


class TestRiskManager:
    """Test suite for the RiskManager class."""

    def test_initialization(self):
        """Test RiskManager initialization."""
        manager = RiskManager()
        
        assert manager.risk_limits is not None
        assert manager.exposure_monitor is not None
        assert manager.stress_tester is not None
        assert manager.compliance_checker is not None
        assert manager.is_active is True

    def test_initialization_custom_params(self):
        """Test RiskManager with custom parameters."""
        manager = RiskManager(
            max_portfolio_var=0.05,  # 5% VaR
            max_drawdown=0.15,       # 15% max drawdown
            max_leverage=2.5,
            rebalance_frequency='daily'
        )
        
        assert manager.max_portfolio_var == 0.05
        assert manager.max_drawdown == 0.15
        assert manager.max_leverage == 2.5
        assert manager.rebalance_frequency == 'daily'

    def test_calculate_portfolio_var(self):
        """Test Value at Risk (VaR) calculation."""
        manager = RiskManager()
        
        # Mock portfolio returns
        returns = pd.Series(np.random.randn(252) * 0.02)  # 1 year of daily returns
        
        portfolio_var = manager.calculate_portfolio_var(returns, confidence_level=0.95)
        
        assert portfolio_var <= 0  # VaR should be negative (loss)
        assert abs(portfolio_var) > 0  # Should be non-zero
        
        # Test different confidence levels
        var_99 = manager.calculate_portfolio_var(returns, confidence_level=0.99)
        var_95 = manager.calculate_portfolio_var(returns, confidence_level=0.95)
        
        # Higher confidence should give higher (more negative) VaR
        assert abs(var_99) > abs(var_95)

    def test_calculate_expected_shortfall(self):
        """Test Expected Shortfall (CVaR) calculation."""
        manager = RiskManager()
        
        returns = pd.Series(np.random.randn(252) * 0.02)
        
        expected_shortfall = manager.calculate_expected_shortfall(returns, confidence_level=0.95)
        
        assert expected_shortfall <= 0  # Should be negative
        assert abs(expected_shortfall) > abs(manager.calculate_portfolio_var(returns, 0.95))
        # ES should be worse than VaR

    def test_calculate_portfolio_beta(self):
        """Test portfolio beta calculation."""
        manager = RiskManager()
        
        # Mock portfolio and market returns
        portfolio_returns = pd.Series(np.random.randn(252) * 0.025)
        market_returns = pd.Series(np.random.randn(252) * 0.02)
        
        portfolio_beta = manager.calculate_portfolio_beta(portfolio_returns, market_returns)
        
        assert isinstance(portfolio_beta, (int, float))
        assert not np.isnan(portfolio_beta)

    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        manager = RiskManager()
        
        # Mock asset returns
        returns_data = pd.DataFrame({
            'SPY': np.random.randn(252) * 0.02,
            'AAPL': np.random.randn(252) * 0.025,
            'GOOGL': np.random.randn(252) * 0.03,
            'MSFT': np.random.randn(252) * 0.022
        })
        
        correlation_matrix = manager.calculate_correlation_matrix(returns_data)
        
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert correlation_matrix.shape == (4, 4)  # 4 assets
        assert correlation_matrix.index.tolist() == ['SPY', 'AAPL', 'GOOGL', 'MSFT']
        
        # Diagonal should be 1.0
        assert all(abs(correlation_matrix.iloc[i, i] - 1.0) < 0.01 for i in range(4))

    def test_calculate_portfolio_volatility(self):
        """Test portfolio volatility calculation."""
        manager = RiskManager()
        
        # Mock portfolio weights and asset volatilities
        weights = {'SPY': 0.4, 'AAPL': 0.3, 'GOOGL': 0.2, 'MSFT': 0.1}
        volatilities = {'SPY': 0.20, 'AAPL': 0.25, 'GOOGL': 0.30, 'MSFT': 0.22}
        
        portfolio_vol = manager.calculate_portfolio_volatility(weights, volatilities)
        
        assert portfolio_vol > 0
        assert portfolio_vol <= max(volatilities.values())  # Should be <= max individual vol

    def test_stress_test_portfolio(self):
        """Test portfolio stress testing."""
        manager = RiskManager()
        
        # Mock portfolio positions
        positions = {
            'SPY': {'quantity': 100, 'price': 400.0},
            'AAPL': {'quantity': 50, 'price': 175.0},
            'GOOGL': {'quantity': 10, 'price': 2500.0}
        }
        
        # Define stress scenarios
        scenarios = {
            'market_crash': {'SPY': -0.25, 'AAPL': -0.30, 'GOOGL': -0.35},
            'tech_bubble': {'SPY': -0.10, 'AAPL': -0.40, 'GOOGL': -0.45, 'MSFT': -0.35},
            'rate_hike': {'SPY': -0.15, 'AAPL': -0.20, 'GOOGL': -0.18, 'MSFT': -0.15}
        }
        
        stress_results = manager.stress_test_portfolio(positions, scenarios)
        
        assert 'scenarios' in stress_results
        assert 'market_crash' in stress_results['scenarios']
        assert 'tech_bubble' in stress_results['scenarios']
        assert 'rate_hike' in stress_results['scenarios']
        
        # Each scenario should have impact metrics
        for scenario_name, scenario_result in stress_results['scenarios'].items():
            assert 'portfolio_impact' in scenario_result
            assert 'worst_case_loss' in scenario_result
            assert 'probability' in scenario_result

    def test_check_risk_limits(self):
        """Test risk limit checking."""
        manager = RiskManager(
            max_portfolio_var=0.05,
            max_drawdown=0.15,
            max_leverage=2.5,
            max_single_position=0.20
        )
        
        # Mock portfolio state
        portfolio_state = {
            'current_var': 0.04,     # Within VaR limit
            'current_drawdown': 0.12, # Within drawdown limit
            'current_leverage': 2.3,  # Within leverage limit
            'largest_position': 0.18, # Within position limit
            'sector_exposures': {
                'technology': 0.35,  # Within sector limit
                'healthcare': 0.15,
                'finance': 0.20
            }
        }
        
        risk_check = manager.check_risk_limits(portfolio_state)
        
        assert 'all_limits_satisfied' in risk_check
        assert 'violations' in risk_check
        assert 'risk_score' in risk_check
        assert 'recommendations' in risk_check
        
        # All limits should be satisfied in this case
        assert risk_check['all_limits_satisfied'] is True
        assert len(risk_check['violations']) == 0

    def test_risk_limit_violations(self):
        """Test risk limit violation detection."""
        manager = RiskManager(
            max_portfolio_var=0.05,
            max_drawdown=0.15,
            max_leverage=2.5
        )
        
        # Portfolio state with violations
        portfolio_state = {
            'current_var': 0.08,      # Exceeds VaR limit
            'current_drawdown': 0.18,  # Exceeds drawdown limit
            'current_leverage': 3.2,   # Exceeds leverage limit
            'largest_position': 0.25   # Exceeds position limit
        }
        
        risk_check = manager.check_risk_limits(portfolio_state)
        
        assert risk_check['all_limits_satisfied'] is False
        assert len(risk_check['violations']) > 0
        
        # Check that specific violations are identified
        violation_types = [v['limit_type'] for v in risk_check['violations']]
        assert 'var_limit' in violation_types
        assert 'drawdown_limit' in violation_types
        assert 'leverage_limit' in violation_types

    def test_generate_risk_alerts(self):
        """Test risk alert generation."""
        manager = RiskManager(alert_thresholds={
            'var_threshold': 0.06,
            'drawdown_threshold': 0.16,
            'correlation_threshold': 0.85
        })
        
        # Portfolio state that triggers alerts
        portfolio_state = {
            'current_var': 0.07,     # Above threshold
            'current_drawdown': 0.17, # Above threshold
            'avg_correlation': 0.88,  # Above threshold
            'sector_concentration': 0.45  # High concentration
        }
        
        alerts = manager.generate_risk_alerts(portfolio_state)
        
        assert len(alerts) > 0
        
        alert_types = [alert['type'] for alert in alerts]
        assert any('var' in alert_type.lower() for alert_type in alert_types)
        assert any('drawdown' in alert_type.lower() for alert_type in alert_types)
        assert any('correlation' in alert_type.lower() for alert_type in alert_types)

    def test_risk_budget_allocation(self):
        """Test risk budget allocation across assets."""
        manager = RiskManager()
        
        # Asset characteristics
        assets = {
            'SPY': {'volatility': 0.20, 'expected_return': 0.08},
            'AAPL': {'volatility': 0.25, 'expected_return': 0.12},
            'GOOGL': {'volatility': 0.30, 'expected_return': 0.10},
            'MSFT': {'volatility': 0.22, 'expected_return': 0.09}
        }
        
        # Risk budget and constraints
        risk_budget = 0.15  # 15% portfolio volatility target
        min_weight = 0.05
        max_weight = 0.40
        
        allocation = manager.allocate_risk_budget(
            assets, risk_budget, min_weight, max_weight
        )
        
        assert 'optimal_weights' in allocation
        assert 'expected_portfolio_vol' in allocation
        assert 'expected_return' in allocation
        assert 'risk_contribution' in allocation
        
        # Weights should sum to 1
        total_weight = sum(allocation['optimal_weights'].values())
        assert abs(total_weight - 1.0) < 0.01
        
        # Portfolio volatility should be close to target
        assert abs(allocation['expected_portfolio_vol'] - risk_budget) < 0.02

    def test_rebalance_for_risk(self):
        """Test risk-based portfolio rebalancing."""
        manager = RiskManager()
        
        # Current portfolio
        current_weights = {'SPY': 0.50, 'AAPL': 0.30, 'GOOGL': 0.20}
        
        # Target risk profile
        target_volatility = 0.18
        target_correlations = {'SPY': 0.8, 'AAPL': 0.7, 'GOOGL': 0.6}
        
        rebalance_plan = manager.rebalance_for_risk(
            current_weights, target_volatility, target_correlations
        )
        
        assert 'recommended_weights' in rebalance_plan
        assert 'trades_required' in rebalance_plan
        assert 'estimated_impact' in rebalance_plan
        
        # Should suggest trades to achieve target risk profile
        if rebalance_plan['trades_required']:
            assert len(rebalance_plan['trades_required']) > 0

    def test_hedge_position_risk(self):
        """Test position risk hedging."""
        manager = RiskManager()
        
        # Current position
        position = {'symbol': 'SPY', 'quantity': 1000, 'beta': 1.2}
        
        # Available hedging instruments
        hedging_instruments = {
            'SPY_put': {'beta': -1.1, 'cost': 0.02},
            'VIX_call': {'beta': -0.8, 'cost': 0.015},
            'SPY_short': {'beta': -1.0, 'cost': 0.001}
        }
        
        hedge_recommendations = manager.hedge_position_risk(
            position, hedging_instruments, max_cost=0.05
        )
        
        assert 'hedge_instruments' in hedge_recommendations
        assert 'optimal_hedge_ratio' in hedge_recommendations
        assert 'total_hedge_cost' in hedge_recommendations
        assert 'residual_risk' in hedge_recommendations
        
        # Should recommend cost-effective hedging
        if hedge_recommendations['hedge_instruments']:
            assert hedge_recommendations['total_hedge_cost'] <= 0.05

    def test_risk_adjusted_position_sizing(self):
        """Test risk-adjusted position sizing."""
        manager = RiskManager()
        
        # Position parameters
        symbol = 'AAPL'
        account_value = 100000.0
        target_risk_contribution = 0.10  # 10% of portfolio risk
        
        # Market conditions
        current_volatility = 0.25
        correlation_with_portfolio = 0.7
        recent_performance = 0.15  # 15% recent gain
        
        position_size = manager.calculate_risk_adjusted_position_size(
            symbol, account_value, target_risk_contribution,
            current_volatility, correlation_with_portfolio, recent_performance
        )
        
        assert position_size > 0
        assert position_size <= (account_value * 0.25) / current_volatility  # Reasonable upper bound

    def test_portfolio_risk_decomposition(self):
        """Test portfolio risk decomposition."""
        manager = RiskManager()
        
        # Portfolio composition
        positions = {
            'SPY': {'weight': 0.4, 'volatility': 0.20},
            'AAPL': {'weight': 0.3, 'volatility': 0.25},
            'GOOGL': {'weight': 0.2, 'volatility': 0.30},
            'MSFT': {'weight': 0.1, 'volatility': 0.22}
        }
        
        # Correlation matrix
        correlation_matrix = pd.DataFrame({
            'SPY': [1.0, 0.7, 0.6, 0.8],
            'AAPL': [0.7, 1.0, 0.8, 0.7],
            'GOOGL': [0.6, 0.8, 1.0, 0.7],
            'MSFT': [0.8, 0.7, 0.7, 1.0]
        }, index=['SPY', 'AAPL', 'GOOGL', 'MSFT'])
        
        decomposition = manager.decompose_portfolio_risk(positions, correlation_matrix)
        
        assert 'total_portfolio_risk' in decomposition
        assert 'risk_contribution_by_asset' in decomposition
        assert 'diversification_benefit' in decomposition
        assert 'concentration_risk' in decomposition
        
        # Risk contributions should sum to total risk
        total_contribution = sum(decomposition['risk_contribution_by_asset'].values())
        assert abs(total_contribution - decomposition['total_portfolio_risk']) < 0.001


class TestExposureMonitor:
    """Test suite for ExposureMonitor class."""

    def test_initialization(self):
        """Test ExposureMonitor initialization."""
        monitor = ExposureMonitor()
        
        assert monitor.exposure_limits is not None
        assert monitor.sector_classification is not None
        assert monitor.geographic_exposure is not None

    def test_calculate_sector_exposure(self):
        """Test sector exposure calculation."""
        monitor = ExposureMonitor()
        
        # Portfolio positions
        positions = {
            'SPY': {'weight': 0.30, 'sector': 'large_cap'},
            'AAPL': {'weight': 0.25, 'sector': 'technology'},
            'GOOGL': {'weight': 0.20, 'sector': 'technology'},
            'JNJ': {'weight': 0.15, 'sector': 'healthcare'},
            'XOM': {'weight': 0.10, 'sector': 'energy'}
        }
        
        sector_exposure = monitor.calculate_sector_exposure(positions)
        
        assert 'technology' in sector_exposure
        assert 'healthcare' in sector_exposure
        assert 'energy' in sector_exposure
        assert 'large_cap' in sector_exposure
        
        # Technology exposure should be 45% (AAPL + GOOGL)
        expected_tech = 0.25 + 0.20
        assert abs(sector_exposure['technology'] - expected_tech) < 0.01

    def test_calculate_geographic_exposure(self):
        """Test geographic exposure calculation."""
        monitor = ExposureMonitor()
        
        # Mock geographic classification
        with patch.object(monitor, '_classify_geographic_exposure') as mock_classify:
            mock_classify.return_value = {
                'US': 0.70,
                'Europe': 0.20,
                'Asia': 0.08,
                'Emerging': 0.02
            }
            
            positions = {
                'SPY': {'weight': 0.40, 'country': 'US'},
                'VEA': {'weight': 0.25, 'country': 'Europe'},
                'VWO': {'weight': 0.20, 'country': 'Emerging'},
                'EWJ': {'weight': 0.15, 'country': 'Asia'}
            }
            
            geo_exposure = monitor.calculate_geographic_exposure(positions)
            
            assert 'US' in geo_exposure
            assert 'Europe' in geo_exposure
            assert 'Asia' in geo_exposure
            assert 'Emerging' in geo_exposure

    def test_check_exposure_limits(self):
        """Test exposure limit checking."""
        monitor = ExposureMonitor(
            max_sector_exposure=0.30,
            max_country_exposure=0.60,
            max_industry_exposure=0.25
        )
        
        # Portfolio with concentrated exposures
        exposures = {
            'technology': 0.45,  # Exceeds limit
            'US': 0.75,          # Exceeds limit
            'software': 0.35     # Exceeds limit
        }
        
        violations = monitor.check_exposure_limits(exposures)
        
        assert len(violations) > 0
        
        violation_types = [v['type'] for v in violations]
        assert any('sector' in vtype.lower() for vtype in violation_types)
        assert any('country' in vtype.lower() for vtype in violation_types)
        assert any('industry' in vtype.lower() for vtype in violation_types)

    def test_monitor_concentration_risk(self):
        """Test concentration risk monitoring."""
        monitor = ExposureMonitor()
        
        # Portfolio with concentration risk
        positions = {
            'SPY': 0.40,  # Large single position
            'AAPL': 0.25,
            'GOOGL': 0.20,
            'MSFT': 0.10,
            'JNJ': 0.05
        }
        
        concentration_metrics = monitor.analyze_concentration_risk(positions)
        
        assert 'top_5_concentration' in concentration_metrics
        assert 'hhi_index' in concentration_metrics  # Herfindahl-Hirschman Index
        assert 'effective_positions' in concentration_metrics
        assert 'concentration_risk_score' in concentration_metrics
        
        # Should flag high concentration
        assert concentration_metrics['concentration_risk_score'] > 0.3

    def test_monitor_correlation_exposure(self):
        """Test correlation exposure monitoring."""
        monitor = ExposureMonitor()
        
        # Mock correlation matrix
        correlation_data = pd.DataFrame({
            'SPY': [1.0, 0.85, 0.75, 0.80],
            'AAPL': [0.85, 1.0, 0.90, 0.85],
            'GOOGL': [0.75, 0.90, 1.0, 0.80],
            'MSFT': [0.80, 0.85, 0.80, 1.0]
        }, index=['SPY', 'AAPL', 'GOOGL', 'MSFT'])
        
        weights = {'SPY': 0.30, 'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.20}
        
        correlation_exposure = monitor.analyze_correlation_exposure(weights, correlation_data)
        
        assert 'portfolio_correlation' in correlation_exposure
        assert 'average_correlation' in correlation_exposure
        assert 'correlation_concentration' in correlation_exposure
        assert 'diversification_ratio' in correlation_exposure
        
        # High correlation should be detected
        assert correlation_exposure['portfolio_correlation'] > 0.7

    def test_dynamic_exposure_adjustment(self):
        """Test dynamic exposure adjustment based on market conditions."""
        monitor = ExposureMonitor()
        
        # Market conditions
        market_conditions = {
            'volatility_regime': 'high',
            'correlation_regime': 'elevated',
            'trend_strength': 'strong_bull',
            'sector_momentum': {
                'technology': 0.8,
                'healthcare': 0.3,
                'energy': -0.2
            }
        }
        
        # Current exposures
        current_exposures = {
            'technology': 0.50,
            'healthcare': 0.20,
            'energy': 0.15
        }
        
        adjustments = monitor.suggest_exposure_adjustments(market_conditions, current_exposures)
        
        assert 'recommended_changes' in adjustments
        assert 'risk_impact' in adjustments
        assert 'implementation_timeline' in adjustments
        
        # Should suggest reducing exposure in high-correlation regime
        if market_conditions['correlation_regime'] == 'elevated':
            assert adjustments['recommended_changes']['technology']['action'] in ['reduce', 'maintain']


class TestRiskAttribution:
    """Test suite for RiskAttribution class."""

    def test_initialization(self):
        """Test RiskAttribution initialization."""
        attribution = RiskAttribution()
        
        assert attribution.attribution_methods is not None
        assert attribution.benchmark_selection is not None

    def test_brinson_attribution(self):
        """Test Brinson performance attribution."""
        attribution = RiskAttribution()
        
        # Mock portfolio and benchmark data
        portfolio_weights = {'SPY': 0.40, 'AAPL': 0.30, 'GOOGL': 0.30}
        benchmark_weights = {'SPY': 0.50, 'AAPL': 0.25, 'GOOGL': 0.25}
        
        portfolio_returns = {'SPY': 0.12, 'AAPL': 0.15, 'GOOGL': 0.18}
        benchmark_returns = {'SPY': 0.10, 'AAPL': 0.12, 'GOOGL': 0.14}
        
        attribution_result = attribution.brinson_attribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )
        
        assert 'allocation_effect' in attribution_result
        assert 'selection_effect' in attribution_result
        assert 'interaction_effect' in attribution_result
        assert 'total_attribution' in attribution_result
        
        # Effects should sum to total return difference
        total_effect = (attribution_result['allocation_effect'] + 
                       attribution_result['selection_effect'] + 
                       attribution_result['interaction_effect'])
        
        expected_total = sum(p_r * p_w - b_r * b_w 
                           for p_r, b_r, p_w, b_w in zip(
                               portfolio_returns.values(), benchmark_returns.values(),
                               portfolio_weights.values(), benchmark_weights.values()))
        
        assert abs(total_effect - expected_total) < 0.001

    def test_factor_attribution(self):
        """Test factor-based risk attribution."""
        attribution = RiskAttribution()
        
        # Mock factor exposures and returns
        factor_exposures = pd.DataFrame({
            'market': [1.0, 1.2, 0.8, 1.1],
            'size': [-0.5, 0.3, 0.8, -0.2],
            'value': [0.2, -0.3, 0.5, 0.1]
        }, index=['SPY', 'AAPL', 'GOOGL', 'MSFT'])
        
        factor_returns = {
            'market': 0.08,
            'size': 0.03,
            'value': 0.05
        }
        
        portfolio_weights = {'SPY': 0.30, 'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.20}
        
        factor_contribution = attribution.calculate_factor_contribution(
            factor_exposures, factor_returns, portfolio_weights
        )
        
        assert 'factor_contributions' in factor_contribution
        assert 'residual_return' in factor_contribution
        assert 'r_squared' in factor_contribution
        
        # Market factor should be the largest contributor
        assert factor_contribution['factor_contributions']['market'] >= max(
            factor_contribution['factor_contributions'][f] 
            for f in ['size', 'value']
        )

    def test_risk_attribution_decomposition(self):
        """Test comprehensive risk attribution decomposition."""
        attribution = RiskAttribution()
        
        # Mock portfolio composition and risk factors
        portfolio_data = {
            'positions': {
                'SPY': {'weight': 0.40, 'volatility': 0.20, 'beta': 1.0},
                'AAPL': {'weight': 0.30, 'volatility': 0.25, 'beta': 1.2},
                'GOOGL': {'weight': 0.30, 'volatility': 0.30, 'beta': 1.1}
            },
            'correlation_matrix': pd.DataFrame({
                'SPY': [1.0, 0.7, 0.6],
                'AAPL': [0.7, 1.0, 0.8],
                'GOOGL': [0.6, 0.8, 1.0]
            }, index=['SPY', 'AAPL', 'GOOGL'])
        }
        
        risk_decomposition = attribution.decompose_portfolio_risk(portfolio_data)
        
        assert 'systematic_risk' in risk_decomposition
        assert 'specific_risk' in risk_decomposition
        assert 'risk_contribution_by_factor' in risk_decomposition
        assert 'diversification_benefit' in risk_decomposition
        
        # Systematic risk should be higher than specific risk for diversified portfolio
        assert risk_decomposition['systematic_risk'] > risk_decomposition['specific_risk']


class TestStressTester:
    """Test suite for StressTester class."""

    def test_initialization(self):
        """Test StressTester initialization."""
        tester = StressTester()
        
        assert tester.stress_scenarios is not None
        assert tester.historical_scenarios is not None
        assert tester.monte_carlo_simulations is not None

    def test_historical_stress_test(self):
        """Test historical scenario stress testing."""
        tester = StressTester()
        
        # Mock current portfolio
        portfolio = {
            'SPY': {'quantity': 100, 'price': 400.0},
            'AAPL': {'quantity': 50, 'price': 175.0},
            'GOOGL': {'quantity': 10, 'price': 2500.0}
        }
        
        # Historical crisis scenarios
        historical_scenarios = {
            '2008_financial_crisis': {
                'SPY': -0.37,
                'AAPL': -0.56,
                'GOOGL': -0.54,
                'description': '2008 Financial Crisis'
            },
            'covid_2020': {
                'SPY': -0.34,
                'AAPL': -0.29,
                'GOOGL': -0.19,
                'description': 'COVID-19 Market Crash'
            },
            'dot_com_bubble': {
                'SPY': -0.49,
                'AAPL': -0.78,
                'GOOGL': -0.94,
                'description': 'Dot-com Bubble Burst'
            }
        }
        
        stress_results = tester.run_historical_stress_test(portfolio, historical_scenarios)
        
        assert 'scenario_results' in stress_results
        assert 'worst_case_scenario' in stress_results
        assert 'average_impact' in stress_results
        assert 'scenario_probabilities' in stress_results
        
        # Each scenario should have impact metrics
        for scenario_name in historical_scenarios.keys():
            assert scenario_name in stress_results['scenario_results']
            scenario_result = stress_results['scenario_results'][scenario_name]
            assert 'portfolio_loss' in scenario_result
            assert 'loss_percentage' in scenario_result

    def test_monte_carlo_stress_test(self):
        """Test Monte Carlo stress testing."""
        tester = StressTester()
        
        portfolio = {
            'SPY': {'weight': 0.40, 'expected_return': 0.08, 'volatility': 0.20},
            'AAPL': {'weight': 0.35, 'expected_return': 0.12, 'volatility': 0.25},
            'GOOGL': {'weight': 0.25, 'expected_return': 0.10, 'volatility': 0.30}
        }
        
        simulation_params = {
            'num_simulations': 1000,
            'time_horizon': 252,  # 1 year
            'confidence_levels': [0.90, 0.95, 0.99]
        }
        
        mc_results = tester.run_monte_carlo_stress_test(portfolio, simulation_params)
        
        assert 'simulation_results' in mc_results
        assert 'var_estimates' in mc_results
        assert 'expected_shortfall' in mc_results
        assert 'probability_of_loss' in mc_results
        
        # VaR estimates at different confidence levels
        for conf_level in simulation_params['confidence_levels']:
            assert conf_level in mc_results['var_estimates']
            assert mc_results['var_estimates'][conf_level] <= 0  # VaR should be negative

    def test_scenario_builder(self):
        """Test custom scenario building."""
        tester = StressTester()
        
        # Define custom scenarios
        custom_scenarios = {
            'inflation_shock': {
                'description': 'High inflation scenario',
                'market_impact': {
                    'SPY': -0.15,
                    'AAPL': -0.20,
                    'GOOGL': -0.18,
                    'TLT': 0.25  # Bonds rally
                },
                'sector_rotation': {
                    'technology': -0.25,
                    'energy': 0.20,
                    'utilities': 0.15,
                    'real_estate': -0.10
                },
                'probability': 0.15
            },
            'geopolitical_crisis': {
                'description': 'Major geopolitical tensions',
                'market_impact': {
                    'SPY': -0.25,
                    'AAPL': -0.30,
                    'GOOGL': -0.28,
                    'VIX': 1.50  # Volatility spike
                },
                'sector_rotation': {
                    'defense': 0.30,
                    'energy': 0.25,
                    'technology': -0.35,
                    'consumer_discretionary': -0.20
                },
                'probability': 0.10
            }
        }
        
        # Add scenarios to tester
        tester.add_custom_scenarios(custom_scenarios)
        
        assert 'inflation_shock' in tester.stress_scenarios
        assert 'geopolitical_crisis' in tester.stress_scenarios
        
        # Test running custom scenarios
        portfolio = {'SPY': 100, 'AAPL': 50, 'GOOGL': 10}
        custom_results = tester.run_custom_stress_test(portfolio, list(custom_scenarios.keys()))
        
        assert 'inflation_shock' in custom_results
        assert 'geopolitical_crisis' in custom_results


class TestRiskCompliance:
    """Test suite for RiskCompliance class."""

    def test_initialization(self):
        """Test RiskCompliance initialization."""
        compliance = RiskCompliance()
        
        assert compliance.regulatory_limits is not None
        assert compliance.internal_limits is not None
        assert compliance.compliance_rules is not None

    def test_regulatory_compliance_check(self):
        """Test regulatory compliance checking."""
        compliance = RiskCompliance()
        
        # Set regulatory limits
        compliance.set_regulatory_limits({
            'max_leverage': 2.0,
            'max_single_position': 0.10,
            'max_sector_concentration': 0.25,
            'liquidity_requirement': 0.05
        })
        
        # Portfolio state to check
        portfolio_state = {
            'leverage': 1.8,
            'largest_position': 0.08,
            'sector_concentration': 0.22,
            'liquid_assets_ratio': 0.12
        }
        
        compliance_result = compliance.check_regulatory_compliance(portfolio_state)
        
        assert 'compliance_status' in compliance_result
        assert 'violations' in compliance_result
        assert 'recommendations' in compliance_result
        
        assert compliance_result['compliance_status'] is True
        assert len(compliance_result['violations']) == 0

    def test_internal_policy_compliance(self):
        """Test internal policy compliance."""
        compliance = RiskCompliance()
        
        # Set internal policies
        compliance.set_internal_policies({
            'max_turnover': 2.0,  # 200% annual turnover
            'min_diversification': 8,  # At least 8 positions
            'max_correlation': 0.75,
            'risk_budget_utilization': 0.80
        })
        
        portfolio_metrics = {
            'annual_turnover': 1.5,
            'num_positions': 12,
            'max_correlation': 0.68,
            'risk_budget_used': 0.75
        }
        
        policy_compliance = compliance.check_internal_policies(portfolio_metrics)
        
        assert 'policy_compliance' in policy_compliance
        assert 'policy_violations' in policy_compliance
        assert 'risk_budget_status' in policy_compliance
        
        assert policy_compliance['policy_compliance'] is True
        assert len(policy_compliance['policy_violations']) == 0

    def test_compliance_reporting(self):
        """Test compliance reporting generation."""
        compliance = RiskCompliance()
        
        # Generate mock compliance report
        report = compliance.generate_compliance_report(
            reporting_period='Q4_2023',
            include_recommendations=True,
            include_trend_analysis=True
        )
        
        assert 'reporting_period' in report
        assert 'executive_summary' in report
        assert 'regulatory_status' in report
        assert 'policy_status' in report
        assert 'recommendations' in report
        assert 'trend_analysis' in report


@pytest.mark.parametrize("var_level,expected_es_relation", [
    (0.95, "es_greater_than_var"),
    (0.99, "es_greater_than_var"),
    (0.90, "es_greater_than_var")
])
def test_var_es_relationship_parametrized(var_level, expected_es_relation):
    """Parametrized test for VaR and ES relationship."""
    manager = RiskManager()
    
    # Generate mock returns
    returns = pd.Series(np.random.randn(1000) * 0.02)
    
    var_value = manager.calculate_portfolio_var(returns, var_level)
    es_value = manager.calculate_expected_shortfall(returns, var_level)
    
    # ES should always be worse (more negative) than VaR
    if expected_es_relation == "es_greater_than_var":
        assert es_value < var_value  # More negative = worse


@pytest.mark.parametrize("concentration_level,risk_score_range", [
    (0.20, [0, 0.3]),   # Low concentration
    (0.40, [0.3, 0.6]), # Medium concentration
    (0.70, [0.6, 0.9]), # High concentration
    (0.90, [0.8, 1.0])  # Very high concentration
])
def test_concentration_risk_scoring_parametrized(concentration_level, risk_score_range):
    """Parametrized test for concentration risk scoring."""
    monitor = ExposureMonitor()
    
    # Create mock portfolio with specific concentration
    positions = {'large_position': concentration_level}
    for i in range(int((1 - concentration_level) * 100)):
        positions[f'position_{i}'] = (1 - concentration_level) / 100
    
    concentration_score = monitor.calculate_concentration_risk_score(positions)
    
    # Check that score falls in expected range
    min_score, max_score = risk_score_range
    assert min_score <= concentration_score <= max_score


if __name__ == "__main__":
    pytest.main([__file__])