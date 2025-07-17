# food_predictor/synthetic/migration_utils.py
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

class MigrationManager:
    """
    Manages migration from Tonic AI to Mostly AI infrastructure.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "migration_backup"
        self.synthetic_dir = self.project_root / "food_predictor" / "synthetic"
        
    def backup_existing_files(self) -> str:
        """
        Backup existing Tonic AI files before migration.
        
        Returns:
            Path to backup directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"tonic_backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating backup at {backup_path}")
        
        # Files to backup
        files_to_backup = [
            "tonic_generator.py",
            "train_with_synthetic.py"
        ]
        
        for file_name in files_to_backup:
            source_file = self.synthetic_dir / file_name
            if source_file.exists():
                shutil.copy2(source_file, backup_path / file_name)
                logger.info(f"Backed up {file_name}")
        
        return str(backup_path)
    
    def create_migration_report(self, backup_path: str) -> str:
        """Create detailed migration report"""
        report_path = Path(backup_path) / "migration_report.md"
        
        report_content = f"""# Tonic AI → Mostly AI Migration Report

**Migration Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Backup Location:** {backup_path}

## Files Migrated

### Backed Up (Tonic AI)
- `tonic_generator.py` → Replaced with `mostly_generator.py`
- `train_with_synthetic.py` → Updated to use Mostly AI

### New Files Created (Mostly AI)
- `mostly_generator.py` → Core Mostly AI implementation
- `validation_engine.py` → Business logic validation
- `migration_utils.py` → Migration management utilities
- `run_mostly_migration.py` → Main migration script

## Configuration Changes Required

### Environment Variables
```bash
# Remove (if set)
unset TONIC_API_KEY

# Add
export MOSTLYAI_API_KEY="your_mostly_ai_key_here"
```

### Dependencies
```bash
# Remove
pip uninstall tonic-ai

# Add  
pip install mostlyai-sdk
```

## Code Changes

### Before (Tonic AI)
```python
from food_predictor.synthetic.tonic_generator import TonicAIGenerator
generator = TonicAIGenerator(api_key=api_key)
```

### After (Mostly AI)
```python
from food_predictor.synthetic.mostly_generator import MostlyAIGenerator
generator = MostlyAIGenerator(api_key=api_key)
```

## Validation Improvements

Mostly AI implementation includes:
- Enhanced business logic validation
- Item-category consistency checks
- Distribution similarity validation
- Comprehensive quality metrics

## Next Steps

1. Set up Mostly AI API key
2. Run migration script: `python run_mostly_migration.py`
3. Validate synthetic data quality
4. Retrain models with new synthetic data
5. Compare model performance metrics

## Rollback Plan

If migration issues occur:
1. Restore files from: `{backup_path}`
2. Reset environment variables
3. Reinstall tonic-ai dependencies
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Migration report created: {report_path}")
        return str(report_path)


def compare_synthetic_datasets(tonic_data: pd.DataFrame, mostly_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare synthetic datasets from Tonic AI vs Mostly AI.
    
    Args:
        tonic_data: Synthetic data from Tonic AI
        mostly_data: Synthetic data from Mostly AI
        
    Returns:
        Comparison metrics
    """
    comparison = {
        'size_comparison': {
            'tonic_records': len(tonic_data),
            'mostly_records': len(mostly_data),
            'size_difference': len(mostly_data) - len(tonic_data)
        },
        'column_comparison': {
            'tonic_columns': set(tonic_data.columns),
            'mostly_columns': set(mostly_data.columns),
            'common_columns': set(tonic_data.columns) & set(mostly_data.columns),
            'tonic_only': set(tonic_data.columns) - set(mostly_data.columns),
            'mostly_only': set(mostly_data.columns) - set(tonic_data.columns)
        }
    }
    
    # Compare distributions for common numerical columns
    common_numerical = []
    for col in comparison['column_comparison']['common_columns']:
        if pd.api.types.is_numeric_dtype(tonic_data[col]) and pd.api.types.is_numeric_dtype(mostly_data[col]):
            common_numerical.append(col)
    
    distribution_comparison = {}
    for col in common_numerical:
        tonic_stats = tonic_data[col].describe()
        mostly_stats = mostly_data[col].describe()
        
        distribution_comparison[col] = {
            'tonic_mean': tonic_stats['mean'],
            'mostly_mean': mostly_stats['mean'],
            'mean_difference': abs(tonic_stats['mean'] - mostly_stats['mean']),
            'tonic_std': tonic_stats['std'],
            'mostly_std': mostly_stats['std'],
            'std_difference': abs(tonic_stats['std'] - mostly_stats['std'])
        }
    
    comparison['distribution_comparison'] = distribution_comparison
    
    return comparison


def validate_migration_requirements() -> Dict[str, bool]:
    """
    Validate that all requirements for Mostly AI migration are met.
    
    Returns:
        Dictionary of requirement check results
    """
    requirements = {}
    
    # Check API key
    mostly_api_key = os.getenv('MOSTLYAI_API_KEY')
    requirements['api_key_set'] = mostly_api_key is not None and len(mostly_api_key) > 0
    
    # Check SDK availability
    try:
        import mostlyai
        requirements['mostlyai_sdk_installed'] = True
    except ImportError:
        requirements['mostlyai_sdk_installed'] = False
    
    # Check data availability
    data_files = [
        'food_predictor/data/datasets/DF5.xlsx'  # Original data file
    ]
    
    requirements['data_files_available'] = all(
        os.path.exists(file) for file in data_files
    )
    
    # Check directory structure
    required_dirs = [
        'food_predictor/synthetic',
        'food_predictor/data'
    ]
    
    requirements['directory_structure'] = all(
        os.path.exists(dir_path) for dir_path in required_dirs
    )
    
    # Overall readiness
    requirements['migration_ready'] = all(requirements.values())
    
    return requirements


def create_mostly_ai_config_template() -> str:
    """
    Create configuration template for Mostly AI.
    
    Returns:
        Path to created config file
    """
    config_content = '''# Mostly AI Configuration for DF5 Catering Data
# Copy this file to .env and fill in your values

# Mostly AI API Configuration
MOSTLYAI_API_KEY=mostly-767d4e24045e5ae6064ec3d20cf7c19041a434eea109e707ee032bab47795777
MOSTLYAI_BASE_URL=https://api.mostly.ai

# Data Generation Parameters
TARGET_MULTIPLIER=3.0          # Generate 3x original data size
VALIDATION_SPLIT=0.2           # 20% for validation
RANDOM_SEED=42                 # For reproducibility

# Quality Thresholds
MIN_BUSINESS_LOGIC_SCORE=0.95  # 95% business rule compliance
MIN_OVERALL_SCORE=0.85         # 85% overall quality score
MIN_CATEGORY_CONSISTENCY=0.90  # 90% item-category consistency

# Output Configuration
SYNTHETIC_DATA_DIR=data/synthetic
BACKUP_DIR=migration_backup
LOG_LEVEL=INFO

# Advanced Settings
PRIVACY_LEVEL=high
GENERATION_METHOD=deep_learning
ENABLE_POST_PROCESSING=true
'''
    
    config_path = "mostly_ai_config.template"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Config template created: {config_path}")
    return config_path


def cleanup_tonic_references(project_root: str) -> List[str]:
    """
    Find and list files that may contain Tonic AI references.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        List of files that may need manual updates
    """
    files_to_check = []
    
    # Patterns to search for
    tonic_patterns = [
        'tonic',
        'TonicAI', 
        'tonic_generator',
        'train_with_synthetic'
    ]
    
    # File extensions to check
    extensions = ['.py', '.md', '.txt', '.yml', '.yaml', '.json']
    
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if any(pattern.lower() in content for pattern in tonic_patterns):
                            files_to_check.append(file_path)
                except:
                    pass  # Skip files that can't be read
    
    return files_to_check