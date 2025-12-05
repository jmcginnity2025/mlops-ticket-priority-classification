"""
Main Pipeline Orchestrator
Runs the complete MLOps pipeline end-to-end
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_preprocessing import DataPreprocessor
from train_model import ModelTrainer
from evaluate_model import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline(model_type: str = 'multiclass', num_iterations: int = 2):
    """
    Run the complete MLOps pipeline

    Args:
        model_type: Type of model ('multiclass' or 'ranking')
        num_iterations: Number of training iterations to compare
    """
    logger.info("="*60)
    logger.info("STARTING MLOPS PIPELINE")
    logger.info("="*60)

    start_time = datetime.now()

    try:
        # Step 1: Data Preprocessing
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("="*60)

        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, output_dir = preprocessor.run_pipeline(
            task_type=model_type
        )

        logger.info(f"✓ Data preprocessing completed")
        logger.info(f"  Processed data saved to: {output_dir}")

        # Step 2: Model Training
        logger.info("\n" + "="*60)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("="*60)

        trainer = ModelTrainer()
        training_results = trainer.run_training_iterations(
            num_iterations=num_iterations,
            model_type=model_type
        )

        logger.info(f"✓ Model training completed")
        logger.info(f"  Trained {len(training_results)} model iterations")

        # Step 3: Model Evaluation
        logger.info("\n" + "="*60)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("="*60)

        evaluator = ModelEvaluator()
        evaluation_report = evaluator.generate_evaluation_report(model_type=model_type)

        logger.info(f"✓ Model evaluation completed")

        if evaluation_report:
            logger.info(f"  Best iteration: {evaluation_report['summary']['best_iteration']}")

            if model_type == 'multiclass':
                logger.info(f"  Best accuracy: {evaluation_report['summary']['best_accuracy']:.4f}")
                logger.info(f"  Best F1: {evaluation_report['summary']['best_f1']:.4f}")
            else:
                logger.info(f"  Best RMSE: {evaluation_report['summary']['best_rmse']:.4f}")
                logger.info(f"  Best R2: {evaluation_report['summary']['best_r2']:.4f}")

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total duration: {duration}")
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        logger.info("\nNext steps:")
        logger.info("1. Review MLflow experiments: mlflow ui")
        logger.info("2. Deploy model: python src/deploy_model.py --action deploy")
        logger.info("3. Setup monitoring: python src/monitor_model.py --action dashboard")

        return True

    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {str(e)}")
        logger.exception(e)
        return False


def run_preprocessing_only():
    """Run only data preprocessing"""
    logger.info("Running data preprocessing only...")

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, output_dir = preprocessor.run_pipeline()

    logger.info(f"✓ Preprocessing completed: {output_dir}")


def run_training_only(model_type: str, num_iterations: int):
    """Run only model training"""
    logger.info(f"Running model training only ({model_type})...")

    trainer = ModelTrainer()
    results = trainer.run_training_iterations(
        num_iterations=num_iterations,
        model_type=model_type
    )

    logger.info(f"✓ Training completed: {len(results)} iterations")


def run_evaluation_only(model_type: str):
    """Run only model evaluation"""
    logger.info(f"Running model evaluation only ({model_type})...")

    evaluator = ModelEvaluator()
    report = evaluator.generate_evaluation_report(model_type=model_type)

    logger.info("✓ Evaluation completed")

    if report:
        print("\nEvaluation Summary:")
        for key, value in report['summary'].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MLOps Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py --mode full --model_type multiclass --iterations 2

  # Run only preprocessing
  python run_pipeline.py --mode preprocess

  # Run only training
  python run_pipeline.py --mode train --model_type multiclass --iterations 3

  # Run only evaluation
  python run_pipeline.py --mode evaluate --model_type multiclass
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'preprocess', 'train', 'evaluate'],
        help='Pipeline mode to run'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='multiclass',
        choices=['multiclass', 'ranking'],
        help='Type of model to train/evaluate'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=2,
        help='Number of training iterations'
    )

    args = parser.parse_args()

    # Run selected mode
    if args.mode == 'full':
        success = run_full_pipeline(
            model_type=args.model_type,
            num_iterations=args.iterations
        )
        sys.exit(0 if success else 1)

    elif args.mode == 'preprocess':
        run_preprocessing_only()

    elif args.mode == 'train':
        run_training_only(args.model_type, args.iterations)

    elif args.mode == 'evaluate':
        run_evaluation_only(args.model_type)
