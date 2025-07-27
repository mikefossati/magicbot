"""
Parameter Space Definition System

Defines the search space for strategy parameters including ranges, constraints,
and dependencies between parameters.
"""

from typing import Dict, List, Any, Union, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import structlog

logger = structlog.get_logger()

class ParameterRange(ABC):
    """Base class for parameter ranges"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def sample(self) -> Any:
        """Sample a random value from this range"""
        pass
    
    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Check if value is within this range"""
        pass
    
    @abstractmethod
    def discretize(self, num_points: int) -> List[Any]:
        """Generate discrete points for grid search"""
        pass

class IntegerRange(ParameterRange):
    """Integer parameter range"""
    
    def __init__(self, name: str, min_value: int, max_value: int, step: int = 1, description: str = ""):
        super().__init__(name, description)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
    
    def sample(self) -> int:
        """Sample random integer in range"""
        return np.random.randint(self.min_value, self.max_value + 1)
    
    def validate(self, value: Any) -> bool:
        """Validate integer value"""
        if not isinstance(value, (int, np.integer)):
            return False
        return self.min_value <= value <= self.max_value
    
    def discretize(self, num_points: int) -> List[int]:
        """Generate evenly spaced integers"""
        if num_points <= 1:
            return [self.min_value]
        
        # Use step if specified, otherwise distribute evenly
        if self.step > 1:
            values = list(range(self.min_value, self.max_value + 1, self.step))
            if len(values) > num_points:
                # Sample evenly from available values
                indices = np.linspace(0, len(values) - 1, num_points, dtype=int)
                return [values[i] for i in indices]
            return values
        else:
            return np.linspace(self.min_value, self.max_value, num_points, dtype=int).tolist()

class FloatRange(ParameterRange):
    """Float parameter range"""
    
    def __init__(self, name: str, min_value: float, max_value: float, log_scale: bool = False, description: str = ""):
        super().__init__(name, description)
        self.min_value = min_value
        self.max_value = max_value
        self.log_scale = log_scale
    
    def sample(self) -> float:
        """Sample random float in range"""
        if self.log_scale:
            log_min = np.log10(self.min_value)
            log_max = np.log10(self.max_value)
            log_val = np.random.uniform(log_min, log_max)
            return 10 ** log_val
        else:
            return np.random.uniform(self.min_value, self.max_value)
    
    def validate(self, value: Any) -> bool:
        """Validate float value"""
        if not isinstance(value, (int, float, np.number)):
            return False
        return self.min_value <= value <= self.max_value
    
    def discretize(self, num_points: int) -> List[float]:
        """Generate evenly spaced floats"""
        if num_points <= 1:
            return [self.min_value]
        
        if self.log_scale:
            log_min = np.log10(self.min_value)
            log_max = np.log10(self.max_value)
            log_values = np.linspace(log_min, log_max, num_points)
            return (10 ** log_values).tolist()
        else:
            return np.linspace(self.min_value, self.max_value, num_points).tolist()

class ChoiceRange(ParameterRange):
    """Discrete choice parameter"""
    
    def __init__(self, name: str, choices: List[Any], description: str = ""):
        super().__init__(name, description)
        self.choices = choices
    
    def sample(self) -> Any:
        """Sample random choice"""
        return np.random.choice(self.choices)
    
    def validate(self, value: Any) -> bool:
        """Validate choice value"""
        return value in self.choices
    
    def discretize(self, num_points: int) -> List[Any]:
        """Return all or subset of choices"""
        if num_points >= len(self.choices):
            return self.choices.copy()
        
        # Sample evenly distributed choices
        indices = np.linspace(0, len(self.choices) - 1, num_points, dtype=int)
        return [self.choices[i] for i in indices]

class BooleanRange(ParameterRange):
    """Boolean parameter"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
    
    def sample(self) -> bool:
        """Sample random boolean"""
        return np.random.choice([True, False])
    
    def validate(self, value: Any) -> bool:
        """Validate boolean value"""
        return isinstance(value, bool)
    
    def discretize(self, num_points: int) -> List[bool]:
        """Return both boolean values"""
        return [False, True] if num_points >= 2 else [False]

class ParameterConstraint:
    """Constraint between parameters"""
    
    def __init__(self, constraint_func: Callable[[Dict[str, Any]], bool], description: str = ""):
        self.constraint_func = constraint_func
        self.description = description
    
    def check(self, parameters: Dict[str, Any]) -> bool:
        """Check if parameters satisfy constraint"""
        try:
            return self.constraint_func(parameters)
        except Exception as e:
            logger.warning("Constraint check failed", error=str(e))
            return False

class ParameterDependency:
    """Dependency between parameters"""
    
    def __init__(
        self,
        dependent_param: str,
        controlling_param: str,
        dependency_func: Callable[[Any], ParameterRange],
        description: str = ""
    ):
        self.dependent_param = dependent_param
        self.controlling_param = controlling_param
        self.dependency_func = dependency_func
        self.description = description
    
    def get_dependent_range(self, controlling_value: Any) -> ParameterRange:
        """Get range for dependent parameter based on controlling value"""
        return self.dependency_func(controlling_value)

class ParameterSpace:
    """
    Complete parameter search space definition.
    
    Manages parameter ranges, constraints, and dependencies for optimization.
    """
    
    def __init__(self):
        self.parameters: Dict[str, ParameterRange] = {}
        self.constraints: List[ParameterConstraint] = []
        self.dependencies: List[ParameterDependency] = []
        
    def add_parameter(self, param_range: ParameterRange) -> 'ParameterSpace':
        """Add a parameter range to the space"""
        self.parameters[param_range.name] = param_range
        return self
    
    def add_integer(
        self,
        name: str,
        min_value: int,
        max_value: int,
        step: int = 1,
        description: str = ""
    ) -> 'ParameterSpace':
        """Add integer parameter"""
        return self.add_parameter(IntegerRange(name, min_value, max_value, step, description))
    
    def add_float(
        self,
        name: str,
        min_value: float,
        max_value: float,
        log_scale: bool = False,
        description: str = ""
    ) -> 'ParameterSpace':
        """Add float parameter"""
        return self.add_parameter(FloatRange(name, min_value, max_value, log_scale, description))
    
    def add_choice(
        self,
        name: str,
        choices: List[Any],
        description: str = ""
    ) -> 'ParameterSpace':
        """Add choice parameter"""
        return self.add_parameter(ChoiceRange(name, choices, description))
    
    def add_boolean(
        self,
        name: str,
        description: str = ""
    ) -> 'ParameterSpace':
        """Add boolean parameter"""
        return self.add_parameter(BooleanRange(name, description))
    
    def add_constraint(
        self,
        constraint_func: Callable[[Dict[str, Any]], bool],
        description: str = ""
    ) -> 'ParameterSpace':
        """Add parameter constraint"""
        self.constraints.append(ParameterConstraint(constraint_func, description))
        return self
    
    def add_dependency(
        self,
        dependent_param: str,
        controlling_param: str,
        dependency_func: Callable[[Any], ParameterRange],
        description: str = ""
    ) -> 'ParameterSpace':
        """Add parameter dependency"""
        self.dependencies.append(
            ParameterDependency(dependent_param, controlling_param, dependency_func, description)
        )
        return self
    
    def sample_parameters(self, respect_constraints: bool = True) -> Dict[str, Any]:
        """Sample random parameter values"""
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            parameters = {}
            
            # Sample base parameters
            for name, param_range in self.parameters.items():
                parameters[name] = param_range.sample()
            
            # Apply dependencies
            parameters = self._apply_dependencies(parameters)
            
            # Check constraints if required
            if not respect_constraints or self._check_constraints(parameters):
                return parameters
        
        logger.warning("Failed to find valid parameter sample after maximum attempts")
        # Return unconstrained sample as fallback
        return {name: param_range.sample() for name, param_range in self.parameters.items()}
    
    def generate_grid(self, grid_size: Union[int, Dict[str, int]]) -> List[Dict[str, Any]]:
        """
        Generate grid of parameter combinations for grid search.
        
        Args:
            grid_size: Number of points per parameter (int) or per-parameter counts (dict)
            
        Returns:
            List of parameter combinations
        """
        if isinstance(grid_size, int):
            points_per_param = {name: grid_size for name in self.parameters.keys()}
        else:
            points_per_param = grid_size
        
        # Generate discrete values for each parameter
        param_values = {}
        for name, param_range in self.parameters.items():
            num_points = points_per_param.get(name, 10)
            param_values[name] = param_range.discretize(num_points)
        
        # Generate all combinations
        param_names = list(param_values.keys())
        value_lists = [param_values[name] for name in param_names]
        
        combinations = []
        for values in self._cartesian_product(value_lists):
            parameters = dict(zip(param_names, values))
            
            # Apply dependencies
            parameters = self._apply_dependencies(parameters)
            
            # Check constraints
            if self._check_constraints(parameters):
                combinations.append(parameters)
        
        logger.info("Generated parameter grid", 
                   total_combinations=len(combinations),
                   requested_size=grid_size)
        
        return combinations
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameter values against ranges and constraints"""
        # Check individual parameter ranges
        for name, value in parameters.items():
            if name not in self.parameters:
                logger.warning("Unknown parameter", name=name)
                return False
            
            if not self.parameters[name].validate(value):
                logger.warning("Parameter out of range", name=name, value=value)
                return False
        
        # Check constraints
        return self._check_constraints(parameters)
    
    def get_bounds(self) -> Dict[str, Tuple[Any, Any]]:
        """Get parameter bounds for optimization algorithms"""
        bounds = {}
        
        for name, param_range in self.parameters.items():
            if isinstance(param_range, (IntegerRange, FloatRange)):
                bounds[name] = (param_range.min_value, param_range.max_value)
            elif isinstance(param_range, ChoiceRange):
                bounds[name] = (0, len(param_range.choices) - 1)
            elif isinstance(param_range, BooleanRange):
                bounds[name] = (0, 1)
        
        return bounds
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all parameters"""
        info = {}
        
        for name, param_range in self.parameters.items():
            param_info = {
                'type': type(param_range).__name__,
                'description': param_range.description
            }
            
            if isinstance(param_range, IntegerRange):
                param_info.update({
                    'min_value': param_range.min_value,
                    'max_value': param_range.max_value,
                    'step': param_range.step
                })
            elif isinstance(param_range, FloatRange):
                param_info.update({
                    'min_value': param_range.min_value,
                    'max_value': param_range.max_value,
                    'log_scale': param_range.log_scale
                })
            elif isinstance(param_range, ChoiceRange):
                param_info['choices'] = param_range.choices
            
            info[name] = param_info
        
        return info
    
    def _apply_dependencies(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter dependencies"""
        updated_parameters = parameters.copy()
        
        for dependency in self.dependencies:
            if dependency.controlling_param in parameters:
                controlling_value = parameters[dependency.controlling_param]
                dependent_range = dependency.get_dependent_range(controlling_value)
                
                # Resample dependent parameter if it's out of range
                if (dependency.dependent_param in updated_parameters and
                    not dependent_range.validate(updated_parameters[dependency.dependent_param])):
                    updated_parameters[dependency.dependent_param] = dependent_range.sample()
        
        return updated_parameters
    
    def _check_constraints(self, parameters: Dict[str, Any]) -> bool:
        """Check if parameters satisfy all constraints"""
        for constraint in self.constraints:
            if not constraint.check(parameters):
                return False
        return True
    
    def _cartesian_product(self, lists: List[List[Any]]) -> List[List[Any]]:
        """Generate cartesian product of lists"""
        if not lists:
            return [[]]
        
        result = [[]]
        for lst in lists:
            result = [prev + [item] for prev in result for item in lst]
        
        return result

# Predefined parameter spaces for common strategy types
class CommonParameterSpaces:
    """Collection of common parameter spaces for different strategy types"""
    
    @staticmethod
    def momentum_strategy() -> ParameterSpace:
        """Parameter space for momentum trading strategies"""
        space = ParameterSpace()
        
        # Trend detection parameters
        space.add_integer('trend_ema_fast', 3, 20, description='Fast EMA period for trend detection')
        space.add_integer('trend_ema_slow', 8, 50, description='Slow EMA period for trend detection')
        space.add_float('trend_strength_threshold', 0.0001, 0.01, log_scale=True, 
                       description='Minimum trend strength to generate signals')
        
        # RSI parameters
        space.add_integer('rsi_period', 5, 21, description='RSI calculation period')
        
        # Volume parameters
        space.add_float('volume_surge_multiplier', 1.05, 2.0, description='Volume surge threshold')
        space.add_boolean('volume_confirmation_required', description='Require volume confirmation')
        
        # Position sizing
        space.add_float('base_position_size', 0.01, 0.1, description='Base position size as fraction of capital')
        space.add_float('max_position_size', 0.05, 0.2, description='Maximum position size')
        
        # Risk management
        space.add_float('stop_loss_atr_multiplier', 1.0, 10.0, description='Stop loss distance in ATR multiples')
        space.add_float('take_profit_risk_reward', 1.0, 5.0, description='Take profit risk/reward ratio')
        
        # Add constraint: fast EMA must be less than slow EMA
        space.add_constraint(
            lambda p: p['trend_ema_fast'] < p['trend_ema_slow'],
            'Fast EMA must be less than slow EMA'
        )
        
        # Add constraint: max position size must be >= base position size
        space.add_constraint(
            lambda p: p['max_position_size'] >= p['base_position_size'],
            'Maximum position size must be >= base position size'
        )
        
        return space
    
    @staticmethod
    def mean_reversion_strategy() -> ParameterSpace:
        """Parameter space for mean reversion strategies"""
        space = ParameterSpace()
        
        # Bollinger Bands parameters
        space.add_integer('bb_period', 10, 50, description='Bollinger Bands period')
        space.add_float('bb_std_dev', 1.5, 3.0, description='Bollinger Bands standard deviations')
        
        # RSI parameters
        space.add_integer('rsi_period', 10, 30, description='RSI period')
        space.add_integer('rsi_oversold', 20, 35, description='RSI oversold threshold')
        space.add_integer('rsi_overbought', 65, 80, description='RSI overbought threshold')
        
        # Position sizing
        space.add_float('position_size', 0.02, 0.1, description='Position size as fraction of capital')
        
        # Risk management
        space.add_float('stop_loss_pct', 0.01, 0.05, description='Stop loss as percentage')
        space.add_float('take_profit_pct', 0.01, 0.05, description='Take profit as percentage')
        
        return space