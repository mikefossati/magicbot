"""
Parameter Optimization Framework

Comprehensive optimization system for trading strategy parameters with
multi-algorithm support, validation, and result analysis.
"""

from .base_optimizer import BaseOptimizer, OptimizationConfig, OptimizationResult
from .parameter_space import ParameterSpace, CommonParameterSpaces
from .objectives import OptimizationObjective, CommonObjectives
from .grid_search_optimizer import GridSearchOptimizer, AdaptiveGridSearchOptimizer
from .genetic_optimizer import GeneticOptimizer
from .multi_objective_optimizer import MultiObjectiveOptimizer

# Validation modules
from .validation.walk_forward_validator import WalkForwardValidator
from .validation.cross_validator import CrossValidator

# Storage and analysis
from .storage import OptimizationDatabase, ResultsAnalyzer, ModelRegistry

__all__ = [
    # Core optimization
    'BaseOptimizer',
    'OptimizationConfig', 
    'OptimizationResult',
    'ParameterSpace',
    'CommonParameterSpaces',
    'OptimizationObjective',
    'CommonObjectives',
    
    # Optimizers
    'GridSearchOptimizer',
    'AdaptiveGridSearchOptimizer',
    'GeneticOptimizer', 
    'MultiObjectiveOptimizer',
    
    # Validation
    'WalkForwardValidator',
    'CrossValidator',
    
    # Storage & Analysis
    'OptimizationDatabase',
    'ResultsAnalyzer',
    'ModelRegistry'
]