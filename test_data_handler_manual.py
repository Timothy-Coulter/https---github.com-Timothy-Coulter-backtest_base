#!/usr/bin/env python3
"""
Simple test script to verify data handler functionality.
This script tests the core functionality without using pytest.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_data_handler_basic_functionality():
    """Test basic DataHandler functionality."""
    print("🧪 Testing DataHandler basic functionality...")
    
    try:
        # Import the classes
        from backtester.data.data_handler import DataHandler, DataLoader, DataValidator, DataPreprocessor
        
        print("✅ Import successful")
        
        # Test DataHandler initialization
        handler = DataHandler()
        assert handler.data_cache is not None
        assert handler.config is not None
        print("✅ DataHandler initialization works")
        
        # Test DataLoader
        loader = DataLoader()
        assert loader.data_sources is not None
        print("✅ DataLoader initialization works")
        
        # Test DataValidator
        validator = DataValidator()
        assert validator.validation_rules is not None
        print("✅ DataValidator initialization works")
        
        # Test DataPreprocessor
        preprocessor = DataPreprocessor()
        assert preprocessor.scalers is not None
        print("✅ DataPreprocessor initialization works")
        
        # Test get_data function
        from backtester.data import get_data
        assert callable(get_data)
        print("✅ get_data function import works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_validation():
    """Test data validation functionality."""
    print("\n🧪 Testing data validation...")
    
    try:
        from backtester.data.data_handler import DataValidator
        
        validator = DataValidator()
        
        # Create valid test data
        valid_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        # Test valid data
        is_valid, errors = validator.validate_ohlcv_data(valid_data)
        assert is_valid
        assert len(errors) == 0
        print("✅ Valid data validation works")
        
        # Test invalid data
        invalid_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [95, 106, 107],  # Invalid: High < Open
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        is_valid, errors = validator.validate_ohlcv_data(invalid_data)
        assert not is_valid
        assert len(errors) > 0
        print("✅ Invalid data detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data validation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("\n🧪 Testing data preprocessing...")
    
    try:
        from backtester.data.data_handler import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Create test data with missing values
        data_with_gaps = pd.DataFrame({
            'Close': [100, np.nan, 102, 103, np.nan],
            'Volume': [1000000, 1100000, np.nan, 1300000, 1400000]
        })
        
        # Test forward fill
        filled_data = preprocessor.handle_missing_values(data_with_gaps, method='forward_fill')
        assert not filled_data['Close'].isna().any()
        print("✅ Missing value handling works")
        
        # Test technical indicators
        price_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
        })
        
        indicators = preprocessor.calculate_technical_indicators(price_data)
        assert 'sma_5' in indicators.columns
        assert 'ema_5' in indicators.columns
        print("✅ Technical indicator calculation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data preprocessing test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\n🧪 Testing data loading...")
    
    try:
        from backtester.data.data_handler import DataHandler
        
        handler = DataHandler()
        
        # Test data export (simulate since we don't want to depend on network)
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        # Test export functionality
        csv_path = handler.export_data(test_data, format='csv', path='test_export.csv')
        assert os.path.exists(csv_path)
        
        # Clean up
        os.remove(csv_path)
        print("✅ Data export works")
        
        # Test get_data_info
        info = handler.get_data_info(test_data)
        assert 'shape' in info
        assert 'columns' in info
        print("✅ Data info extraction works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting DataHandler Test Suite")
    print("=" * 50)
    
    tests = [
        test_data_handler_basic_functionality,
        test_data_validation,
        test_data_preprocessing,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DataHandler functionality is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)