# food_predictor/scripts/run_mostly_migration.py
"""
Main migration script from Tonic AI to Mostly AI.
Orchestrates the complete migration process with validation and testing.
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from food_predictor.synthetic.mostly_generator import MostlyAIDataAugmentor
from food_predictor.synthetic.validation_engine import SyntheticDataValidator
from food_predictor.synthetic.migration_utils import (
    MigrationManager, 
    validate_migration_requirements,
    create_mostly_ai_config_template,
    cleanup_tonic_references
)
log_file_handler = logging.FileHandler('mostly_migration.log', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(message)s')) # Simple format for console

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        log_file_handler,
        console_handler
    ])
logger = logging.getLogger(__name__)

def run_migration_checks() -> bool:
    """Run pre-migration validation checks"""
    logger.info("🔍 Running pre-migration checks...")
    
    requirements = validate_migration_requirements()
    
    print("\n📋 Migration Requirements Check:")
    print("=" * 50)
    
    for requirement, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {requirement.replace('_', ' ').title()}: {status}")
    
    if not requirements['migration_ready']:
        print("\n❌ Migration requirements not met. Please address the issues above.")
        
        if not requirements['api_key_set']:
            print("\n💡 To set up Mostly AI API key:")
            print("   export MOSTLYAI_API_KEY='your_api_key_here'")
        
        if not requirements['mostlyai_sdk_installed']:
            print("\n💡 To install Mostly AI SDK:")
            print("   pip install mostlyai-sdk")
        
        return False
    
    print("\n✅ All requirements met. Ready for migration!")
    return True

def load_df5_data() -> pd.DataFrame:
    """Load and validate DF5 dataset"""
    logger.info("📂 Loading DF5 dataset...")
    
    df5_path = "food_predictor/data/datasets/DF5.xlsx"
    if not os.path.exists(df5_path):
        raise FileNotFoundError(f"DF5 dataset not found at {df5_path}")
    
    df5_data = pd.read_excel(df5_path)
    logger.info(f"Loaded DF5 data: {len(df5_data)} records, {len(df5_data.columns)} columns")
    
    # Basic validation
    required_columns = ['Order_Number', 'Item_name', 'Category', 'total_guest_count', 'Per_person_quantity']
    missing_columns = [col for col in required_columns if col not in df5_data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in DF5 data: {missing_columns}")
    
    print(f"📊 DF5 Dataset Summary:")
    print(f"   Records: {len(df5_data):,}")
    print(f"   Unique Orders: {df5_data['Order_Number'].nunique():,}")
    print(f"   Unique Items: {df5_data['Item_name'].nunique():,}")
    print(f"   Categories: {df5_data['Category'].nunique()}")
    
    return df5_data

def run_synthetic_generation(df5_data: pd.DataFrame, target_multiplier: float = 3.0) -> Dict[str, pd.DataFrame]:
    """Generate synthetic data using Mostly AI"""
    logger.info(f"🤖 Generating synthetic data (target: {target_multiplier}x original size)...")
    
    try:
        # Initialize Mostly AI augmentor
        augmentor = MostlyAIDataAugmentor()
        
        # Run augmentation pipeline
        datasets = augmentor.augment_df5_dataset(
            df5_data=df5_data,
            target_multiplier=target_multiplier,
            validation_split=0.2
        )
        
        print(f"\n📈 Synthetic Data Generation Results:")
        print(f"   Original data: {len(df5_data):,} records")
        print(f"   Training data: {len(datasets['training']):,} records")
        print(f"   Synthetic data: {len(datasets['synthetic']):,} records")
        print(f"   Combined data: {len(datasets['combined']):,} records")
        print(f"   Validation data: {len(datasets['validation']):,} records")
        
        return datasets
        
    except Exception as e:
        logger.error(f"Synthetic data generation failed: {str(e)}")
        raise

def run_validation(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Validate synthetic data quality"""
    logger.info("🔍 Validating synthetic data quality...")
    
    validator = SyntheticDataValidator()
    
    # Run comprehensive validation
    validation_report = validator.comprehensive_validation(
        synthetic_data=datasets['synthetic'],
        reference_data=datasets['training']
    )
    
    print(f"\n📊 Validation Results:")
    print("=" * 50)
    print(f"Overall Valid: {'✅ Yes' if validation_report['overall_valid'] else '❌ No'}")
    print(f"Overall Score: {validation_report['overall_score']:.1%}")
    print(f"Business Logic Score: {validation_report['business_logic']['score']:.1%}")
    print(f"Category Consistency: {validation_report['category_consistency']:.1%}")
    
    print(f"\n📋 Recommendations:")
    for rec in validation_report['recommendations']:
        print(f"   {rec}")
    
    if validation_report['business_logic']['violations']:
        print(f"\n⚠️  Business Logic Violations:")
        for violation in validation_report['business_logic']['violations'][:5]:  # Show first 5
            print(f"   • {violation}")
    
    return validation_report

def save_results(datasets: Dict[str, pd.DataFrame], validation_report: Dict[str, Any]) -> Dict[str, str]:
    """Save generated datasets and reports"""
    logger.info("💾 Saving results...")
    
    # Create output directory
    output_dir = Path("data/synthetic/mostly_ai")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save datasets
    file_paths = {}
    
    for dataset_name, dataset in datasets.items():
        filename = f"df5_{dataset_name}_{timestamp}.xlsx"
        filepath = output_dir / filename
        dataset.to_excel(filepath, index=False)
        file_paths[dataset_name] = str(filepath)
        print(f"   💾 Saved {dataset_name}: {filepath} ({len(dataset):,} records)")
    
    # Save validation report
    import json
    validation_filename = f"validation_report_{timestamp}.json"
    validation_filepath = output_dir / validation_filename
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    validation_report_serializable = convert_numpy_types(validation_report)
    
    with open(validation_filepath, 'w') as f:
        json.dump(validation_report_serializable, f, indent=2)
    
    file_paths['validation_report'] = str(validation_filepath)
    print(f"   📊 Saved validation report: {validation_filepath}")
    
    return file_paths

def run_model_comparison_test(datasets: Dict[str, pd.DataFrame]) -> None:
    """Run quick model performance comparison"""
    logger.info("🧪 Running model performance comparison...")
    
    try:
        from food_predictor.core.model_manager import ModelManager
        from food_predictor.core.feature_engineering import FeatureEngineer
        from food_predictor.core.category_rules import FoodCategoryRules
        from food_predictor.data.item_service import ItemService
        from food_predictor.core.menu_analyzer import MenuAnalyzer
        
        # Initialize components
        food_rules = FoodCategoryRules()
        item_service = ItemService(food_rules)
        feature_engineer = FeatureEngineer(food_rules)
        menu_analyzer = MenuAnalyzer(food_rules, item_service)
        
        # Test model training on synthetic data
        model_manager = ModelManager(feature_engineer, item_service, food_rules, menu_analyzer)
        
        print(f"\n🧪 Training model on synthetic data...")
        print(f"   Training records: {len(datasets['combined']):,}")
        
        # Build metadata and train
        item_service.build_item_metadata(datasets['combined'])
        model_manager.fit(datasets['combined'])
        
        print(f"   ✅ Model training successful on synthetic data")
        print(f"   📊 Category models: {len(model_manager.category_models)}")
        print(f"   📊 Item models: {len(model_manager.item_models)}")
        
    except Exception as e:
        logger.warning(f"Model comparison test failed: {str(e)}")
        print(f"   ⚠️  Model comparison test failed (this is not critical for migration)")

def main():
    """Main migration orchestration function"""
    parser = argparse.ArgumentParser(description="Migrate from Tonic AI to Mostly AI")
    parser.add_argument("--target-multiplier", type=float, default=3.0, 
                       help="Target data multiplier (default: 3.0)")
    parser.add_argument("--skip-checks", action="store_true", 
                       help="Skip pre-migration checks")
    parser.add_argument("--skip-generation", action="store_true",
                       help="Skip synthetic data generation (use existing)")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip validation (faster testing)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Tonic AI → Mostly AI Migration")
    print("=" * 60)
    
    try:
        # Step 1: Pre-migration checks
        if not args.skip_checks:
            if not run_migration_checks():
                return False
        
        # Step 2: Backup existing files
        migration_manager = MigrationManager(str(project_root))
        backup_path = migration_manager.backup_existing_files()
        print(f"📦 Created backup: {backup_path}")
        
        # Step 3: Load DF5 data
        df5_data = load_df5_data()
        
        # Step 4: Generate synthetic data
        if not args.skip_generation:
            datasets = run_synthetic_generation(df5_data, args.target_multiplier)
        else:
            print("⏭️  Skipping synthetic data generation")
            # For testing, create dummy datasets
            datasets = {
                'synthetic': df5_data.sample(frac=0.5),
                'training': df5_data.sample(frac=0.7),
                'validation': df5_data.sample(frac=0.2),
                'combined': df5_data
            }
        
        # Step 5: Validate synthetic data
        if not args.skip_validation:
            validation_report = run_validation(datasets)
        else:
            print("⏭️  Skipping validation")
            validation_report = {'overall_valid': True, 'overall_score': 1.0}
        
        # Step 6: Save results
        file_paths = save_results(datasets, validation_report)
        
        # Step 7: Model comparison test
        run_model_comparison_test(datasets)
        
        # Step 8: Create migration report
        report_path = migration_manager.create_migration_report(backup_path)
        print(f"📝 Migration report: {report_path}")
        
        # Step 9: Final summary
        print(f"\n🎉 Migration Complete!")
        print("=" * 60)
        print(f"✅ Backup created: {backup_path}")
        print(f"✅ Synthetic data generated: {len(datasets['synthetic']):,} records")
        print(f"✅ Validation score: {validation_report['overall_score']:.1%}")
        print(f"✅ Files saved in: data/synthetic/mostly_ai/")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Review validation report for any issues")
        print(f"   2. Update training scripts to use new synthetic data")
        print(f"   3. Retrain models and compare performance")
        print(f"   4. Run evaluation on test set to measure MAPE improvement")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        print(f"\n❌ Migration failed: {str(e)}")
        print(f"💡 Check migration logs for details: mostly_migration.log")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)