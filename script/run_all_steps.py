"""
RUN ALL STEPS: Complete Pipeline Execution
===========================================
Script ini menjalankan seluruh pipeline dari awal sampai akhir:
1. Insert data to Neo4j
2. Create relationships
3. Generate embeddings
4. Train model
5. Demo prediction and save

Usage:
    python run_all_steps.py

atau jalankan step by step:
    python step1_insert_data_to_neo4j.py
    python step2_create_relationships.py
    python step3_generate_embeddings.py
    python step4_train_model.py
    python step5_predict_and_save.py
"""

import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_execution.log')
    ]
)
logger = logging.getLogger(__name__)


def run_step(step_name, module_name):
    """Run a single step"""
    print("\n" + "=" * 70)
    print(f"EXECUTING: {step_name}")
    print("=" * 70 + "\n")
    
    try:
        # Import and run the module's main function
        module = __import__(module_name)
        module.main()
        
        logger.info(f"✅ {step_name} completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ {step_name} failed: {e}")
        return False


def main():
    """Run all pipeline steps"""
    print("\n" + "=" * 70)
    print("FOOTBALL PLAYER PREDICTION PIPELINE")
    print("Complete End-to-End Workflow")
    print("=" * 70 + "\n")
    
    print("This will execute the following steps:")
    print("  1. Insert data from CSV to Neo4j")
    print("  2. Create similarity relationships")
    print("  3. Generate graph embeddings")
    print("  4. Train prediction model")
    print("  5. Demo predictions and save to Neo4j")
    print("\n⚠️  WARNING: This may take 10-30 minutes depending on data size.")
    
    response = input("\nContinue? (yes/no): ").lower()
    if response != 'yes':
        print("Pipeline execution cancelled.")
        return
    
    # Define pipeline steps
    steps = [
        ("STEP 1: Insert Data to Neo4j", "step1_insert_data_to_neo4j"),
        ("STEP 2: Create Relationships", "step2_create_relationships"),
        ("STEP 3: Generate Embeddings", "step3_generate_embeddings"),
        ("STEP 4: Train Model", "step4_train_model"),
        ("STEP 5: Predict and Save", "step5_predict_and_save")
    ]
    
    results = []
    
    for step_name, module_name in steps:
        success = run_step(step_name, module_name)
        results.append((step_name, success))
        
        if not success:
            print(f"\n❌ Pipeline stopped at: {step_name}")
            print("Fix the error and run remaining steps manually.")
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70 + "\n")
    
    for step_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n" + "=" * 70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n✅ Your football player prediction system is ready!")
        print("\n🚀 Next steps:")
        print("  - Run 'streamlit run app.py' to use the web interface")
        print("  - Check Neo4j Browser to see all data and relationships")
        print("  - Review model_config.json for performance metrics")
    else:
        print("\n⚠️  Pipeline completed with errors. Review logs above.")


if __name__ == "__main__":
    main()
