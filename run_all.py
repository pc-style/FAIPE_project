#!/usr/bin/env python3
"""
Run all lab12 tasks in sequence
Saves all outputs to output/ and figures to figures/
"""
import os
import sys
import time

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 70)
    print("  LAB12: Unsupervised Learning & Topological Data Analysis")
    print("  Running all tasks...")
    print("=" * 70)
    
    start_time = time.time()
    
    tasks = [
        ("01_eda", "Task 1: Exploratory Data Analysis"),
        ("02_clustering", "Task 2: Classical Clustering Methods"),
        ("03_som_synthetic", "Task 3: SOM on Synthetic Data"),
        ("04_som_iris", "Task 4: SOM on Iris (Mandatory)"),
        ("05_som_large", "Task 5: Large SOM & Topology Quality"),
        ("06_neighborhoods", "Task 6: Neighborhood Functions"),
    ]
    
    for module_name, description in tasks:
        print(f"\n{'-' * 70}")
        print(f"  Running: {description}")
        print(f"{'-' * 70}")
        
        task_start = time.time()
        
        try:
            module = __import__(module_name)
            module.run()
            task_time = time.time() - task_start
            print(f"\n  Completed in {task_time:.1f}s")
        except Exception as e:
            print(f"\n  [ERROR] Error in {module_name}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("  ALL TASKS COMPLETE!")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)
    
    # Count outputs
    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    n_figures = len([f for f in os.listdir(figures_dir) if f.endswith('.png')])
    n_outputs = len([f for f in os.listdir(output_dir) if f.endswith('.txt')])
    
    print(f"\n  Figures saved: {n_figures} (in figures/)")
    print(f"  Outputs saved: {n_outputs} (in output/)")
    print("\n  Ready to compile PDF report!")

if __name__ == '__main__':
    main()
