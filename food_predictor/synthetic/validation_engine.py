# food_predictor/synthetic/validation_engine.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    score: float
    violations: List[str]
    metrics: Dict[str, Any]

class BusinessLogicValidator:
    """
    Validates synthetic catering data against business logic constraints.
    Critical for ensuring Mostly AI output maintains domain coherence.
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        
    def _initialize_validation_rules(self) -> Dict[str, Dict]:
        """Define comprehensive business logic validation rules"""
        return {
            # Critical constraints (must pass)
            'veg_count_constraint': {
                'rule': lambda df: df['Veg_Count'] <= df['total_guest_count'],
                'severity': 'critical',
                'description': 'Vegetarian count cannot exceed total guest count'
            },
            'positive_quantities': {
                'rule': lambda df: df['quantity_numeric'] > 0,
                'severity': 'critical', 
                'description': 'All quantities must be positive'
            },
            'valid_veg_ratio': {
                'rule': lambda df: (df['veg_ratio'] >= 0) & (df['veg_ratio'] <= 1),
                'severity': 'critical',
                'description': 'Vegetarian ratio must be between 0 and 1'
            },
            'positive_guest_count': {
                'rule': lambda df: df['total_guest_count'] > 0,
                'severity': 'critical',
                'description': 'Guest count must be positive'
            },
            
            # Important constraints (should pass)
            'reasonable_quantities': {
                'rule': lambda df: (df['quantity_numeric'] >= 1) & (df['quantity_numeric'] <= 2000),
                'severity': 'important',
                'description': 'Per-person quantities should be reasonable (1g-2000g)'
            },
            'reasonable_guest_count': {
                'rule': lambda df: (df['total_guest_count'] >= 1) & (df['total_guest_count'] <= 1000),
                'severity': 'important', 
                'description': 'Guest count should be reasonable (1-1000)'
            },
            
            # Warning level constraints (nice to have)
            'common_event_types': {
                'rule': lambda df: df['Event_Type'].isin([
                    'Wedding', 'Reception', 'Birthday Party', 'Corporate Event', 
                    'Family Gathering', 'Conference', 'Workshop'
                ]),
                'severity': 'warning',
                'description': 'Event types should be common categories'
            },
            'valid_meal_times': {
                'rule': lambda df: df['Meal_Time'].isin(['Breakfast', 'Lunch', 'Dinner', 'Hi-tea']),
                'severity': 'warning',
                'description': 'Meal times should be standard categories'
            }
        }
    
    def validate_dataset(self, synthetic_data: pd.DataFrame) -> ValidationResult:
        """
        Comprehensive validation of synthetic dataset.
        
        Args:
            synthetic_data: Generated synthetic dataset
            
        Returns:
            ValidationResult with overall assessment
        """
        logger.info(f"Validating synthetic dataset with {len(synthetic_data)} records")
        
        violations = []
        metrics = {}
        severity_scores = {'critical': 0, 'important': 0, 'warning': 0}
        
        # Apply each validation rule
        for rule_name, rule_config in self.validation_rules.items():
            try:
                # Apply rule
                rule_result = rule_config['rule'](synthetic_data)
                pass_rate = rule_result.mean() if hasattr(rule_result, 'mean') else float(rule_result)
                
                severity = rule_config['severity']
                metrics[f'{rule_name}_pass_rate'] = pass_rate
                
                # Track violations
                if pass_rate < 1.0:
                    violation_count = len(synthetic_data) - rule_result.sum()
                    violations.append(f"{rule_config['description']}: {violation_count} violations ({pass_rate:.2%} pass rate)")
                
                # Update severity scores
                severity_scores[severity] = max(severity_scores[severity], pass_rate)
                
            except Exception as e:
                logger.error(f"Error validating rule '{rule_name}': {str(e)}")
                violations.append(f"Validation error for {rule_name}: {str(e)}")
        
        # Calculate overall score (weighted by severity)
        overall_score = (
            severity_scores['critical'] * 0.6 +
            severity_scores['important'] * 0.3 + 
            severity_scores['warning'] * 0.1
        )
        
        # Determine if dataset is valid
        is_valid = (
            severity_scores['critical'] >= 0.95 and  # 95%+ critical rules pass
            severity_scores['important'] >= 0.90 and  # 90%+ important rules pass
            overall_score >= 0.90  # 90%+ overall score
        )
        
        logger.info(f"Validation complete. Overall score: {overall_score:.2%}, Valid: {is_valid}")
        
        return ValidationResult(
            is_valid=is_valid,
            score=overall_score,
            violations=violations,
            metrics=metrics
        )
    
    def validate_item_category_consistency(self, synthetic_data: pd.DataFrame, reference_data: pd.DataFrame) -> float:
        """
        Validate that item-category mappings in synthetic data match reference patterns.
        
        Args:
            synthetic_data: Generated synthetic data
            reference_data: Original DF5 data for reference
            
        Returns:
            Consistency score (0-1)
        """
        # Build reference item-category mapping
        ref_mapping = reference_data.groupby('Item_name')['Category'].first().to_dict()
        
        # Check consistency in synthetic data
        consistency_scores = []
        
        for _, row in synthetic_data.iterrows():
            item = row['Item_name']
            category = row['Category']
            
            if item in ref_mapping:
                # Item exists in reference, check if category matches
                expected_category = ref_mapping[item]
                consistency_scores.append(1.0 if category == expected_category else 0.0)
            else:
                # New item, we can't validate (assume neutral)
                consistency_scores.append(0.5)
        
        return np.mean(consistency_scores)
    
    def validate_quantity_distributions(self, synthetic_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, float]:
        """
        Compare quantity distributions between synthetic and reference data.
        
        Args:
            synthetic_data: Generated synthetic data
            reference_data: Original DF5 data
            
        Returns:
            Dictionary of distribution similarity metrics
        """
        metrics = {}
        
        # Compare overall quantity distribution
        ref_quantities = reference_data['quantity_numeric']
        syn_quantities = synthetic_data['quantity_numeric']
        
        # Statistical similarity tests
        from scipy import stats
        
        # KS test for distribution similarity
        ks_stat, ks_pvalue = stats.ks_2samp(ref_quantities, syn_quantities)
        metrics['ks_test_pvalue'] = ks_pvalue
        metrics['ks_test_similar'] = ks_pvalue > 0.05  # Not significantly different
        
        # Mean and std comparison
        ref_mean, ref_std = ref_quantities.mean(), ref_quantities.std()
        syn_mean, syn_std = syn_quantities.mean(), syn_quantities.std()
        
        metrics['mean_similarity'] = 1 - abs(ref_mean - syn_mean) / ref_mean
        metrics['std_similarity'] = 1 - abs(ref_std - syn_std) / ref_std
        
        # Category-wise validation
        common_categories = set(reference_data['Category']) & set(synthetic_data['Category'])
        
        category_similarities = []
        for category in common_categories:
            ref_cat_qty = reference_data[reference_data['Category'] == category]['quantity_numeric']
            syn_cat_qty = synthetic_data[synthetic_data['Category'] == category]['quantity_numeric']
            
            if len(ref_cat_qty) > 5 and len(syn_cat_qty) > 5:  # Enough samples
                try:
                    ks_stat, ks_pvalue = stats.ks_2samp(ref_cat_qty, syn_cat_qty)
                    category_similarities.append(ks_pvalue)
                except:
                    pass
        
        metrics['category_distribution_similarity'] = np.mean(category_similarities) if category_similarities else 0.5
        
        return metrics

class SyntheticDataValidator:
    """
    High-level validator that orchestrates all validation checks.
    """
    
    def __init__(self):
        self.business_validator = BusinessLogicValidator()
    
    def comprehensive_validation(
        self, 
        synthetic_data: pd.DataFrame, 
        reference_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation suite on synthetic data.
        
        Args:
            synthetic_data: Generated synthetic dataset
            reference_data: Original DF5 dataset for comparison
            
        Returns:
            Complete validation report
        """
        logger.info("Running comprehensive validation suite")
        
        # 1. Business logic validation
        business_result = self.business_validator.validate_dataset(synthetic_data)
        
        # 2. Item-category consistency
        category_consistency = self.business_validator.validate_item_category_consistency(
            synthetic_data, reference_data
        )
        
        # 3. Quantity distribution validation
        distribution_metrics = self.business_validator.validate_quantity_distributions(
            synthetic_data, reference_data
        )
        
        # 4. Data quality metrics
        quality_metrics = self._calculate_quality_metrics(synthetic_data)
        
        # 5. Overall assessment
        overall_score = self._calculate_overall_score(
            business_result.score,
            category_consistency,
            distribution_metrics,
            quality_metrics
        )
        
        validation_report = {
            'overall_valid': business_result.is_valid and overall_score >= 0.85,
            'overall_score': overall_score,
            'business_logic': {
                'valid': business_result.is_valid,
                'score': business_result.score,
                'violations': business_result.violations,
                'metrics': business_result.metrics
            },
            'category_consistency': category_consistency,
            'distribution_similarity': distribution_metrics,
            'data_quality': quality_metrics,
            'recommendations': self._generate_recommendations(
                business_result, category_consistency, distribution_metrics
            )
        }
        
        logger.info(f"Validation complete. Overall score: {overall_score:.2%}")
        return validation_report
    
    def _calculate_quality_metrics(self, synthetic_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate general data quality metrics"""
        return {
            'completeness': 1 - synthetic_data.isnull().mean().mean(),
            'uniqueness_orders': synthetic_data['Order_Number'].nunique() / len(synthetic_data),
            'uniqueness_items': synthetic_data['Item_name'].nunique(),
            'record_count': len(synthetic_data)
        }
    
    def _calculate_overall_score(self, business_score: float, category_consistency: float, 
                                distribution_metrics: Dict, quality_metrics: Dict) -> float:
        """Calculate weighted overall validation score"""
        dist_score = np.mean([
            distribution_metrics.get('mean_similarity', 0),
            distribution_metrics.get('std_similarity', 0),
            distribution_metrics.get('category_distribution_similarity', 0)
        ])
        
        quality_score = np.mean([
            quality_metrics.get('completeness', 0),
            min(quality_metrics.get('uniqueness_orders', 0) * 2, 1)  # Cap at 1
        ])
        
        # Weighted combination
        overall_score = (
            business_score * 0.4 +           # Business logic most important
            category_consistency * 0.25 +    # Category mapping critical
            dist_score * 0.25 +              # Distribution similarity important  
            quality_score * 0.1              # General quality baseline
        )
        
        return overall_score
    
    def _generate_recommendations(self, business_result: ValidationResult, 
                                category_consistency: float, distribution_metrics: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not business_result.is_valid:
            recommendations.append("❌ Critical business logic violations detected. Review Mostly AI constraints.")
        
        if category_consistency < 0.9:
            recommendations.append("⚠️ Item-category mapping inconsistencies. Consider stronger conditional dependencies.")
        
        if distribution_metrics.get('ks_test_similar', False) is False:
            recommendations.append("📊 Quantity distributions differ significantly from reference data.")
        
        if business_result.score >= 0.95:
            recommendations.append("✅ Excellent business logic compliance.")
        
        if len(recommendations) == 0:
            recommendations.append("✅ Synthetic data passes all validation checks.")
        
        return recommendations