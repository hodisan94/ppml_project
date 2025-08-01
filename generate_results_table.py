#!/usr/bin/env python3
"""
Automatic Results Table Generator
Generates a comprehensive table showing Attack Success Rate (ASR) for all models and attacks.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_attack_results():
    """Load all attack results from the output directory"""
    results = {}
    
    # Define model types and their directories
    model_types = {
        "naive": "output/results/naive",
        "federated": "output/results/federated", 
        "federated_dp": "output/results/federated_dp"
    }
    
    # Define attack types and their file patterns
    attack_types = {
        "mia": "mia_results_{model}.json",
        "inversion": "inversion_results_{model}_rf.json", 
        "aia": "aia_results_{model}_rf.json"
    }
    
    for model_type, model_dir in model_types.items():
        results[model_type] = {}
        
        for attack_type, file_pattern in attack_types.items():
            # Handle different naming patterns
            if attack_type == "mia":
                filename = f"mia_results_{model_type}.json"
            elif attack_type == "inversion":
                if model_type == "naive":
                    filename = "inversion_results_naive_rf.json"
                elif model_type == "federated":
                    filename = "inversion_results_federated.json"
                else:  # federated_dp
                    filename = "inversion_results_federated_dp.json"
            elif attack_type == "aia":
                if model_type == "naive":
                    filename = "aia_results_naive_rf.json"
                elif model_type == "federated":
                    filename = "aia_results_federated.json"
                else:  # federated_dp
                    filename = "aia_results_federated_dp.json"
            
            file_path = os.path.join(model_dir, filename)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        results[model_type][attack_type] = data
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    results[model_type][attack_type] = None
            else:
                print(f"File not found: {file_path}")
                results[model_type][attack_type] = None
    
    return results

def extract_asr_metrics(results):
    """Extract ASR metrics from the results"""
    asr_data = []
    
    for model_type, attacks in results.items():
        for attack_type, data in attacks.items():
            if data is None:
                asr_data.append({
                    'Model': model_type.replace('_', ' ').title(),
                    'Attack': attack_type.upper(),
                    'ASR': 'N/A',
                    'Accuracy': 'N/A',
                    'AUC': 'N/A',
                    'Precision': 'N/A',
                    'Recall': 'N/A',
                    'F1': 'N/A'
                })
                continue
            
            # Extract metrics based on attack type
            if attack_type == "mia":
                # For MIA, the data structure is different - it's a dict with attack types as keys
                if isinstance(data, dict) and any(key in data for key in ['confidence', 'entropy', 'loss']):
                    # Find the best attack (highest accuracy)
                    best_attack = None
                    best_accuracy = 0
                    
                    for attack_name, attack_results in data.items():
                        if isinstance(attack_results, dict) and 'accuracy' in attack_results:
                            accuracy = attack_results['accuracy']
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_attack = attack_name
                    
                    if best_attack:
                        attack_results = data[best_attack]
                        asr_data.append({
                            'Model': model_type.replace('_', ' ').title(),
                            'Attack': f"MIA ({best_attack})",
                            'ASR': f"{attack_results.get('accuracy', 0):.4f}",
                            'Accuracy': f"{attack_results.get('accuracy', 0):.4f}",
                            'AUC': f"{attack_results.get('auc', 0):.4f}",
                            'Precision': f"{attack_results.get('precision', 0):.4f}",
                            'Recall': f"{attack_results.get('recall', 0):.4f}",
                            'F1': f"{attack_results.get('f1', 0):.4f}"
                        })
                    else:
                        # Fallback if no valid attack found
                        asr_data.append({
                            'Model': model_type.replace('_', ' ').title(),
                            'Attack': 'MIA',
                            'ASR': 'N/A',
                            'Accuracy': 'N/A',
                            'AUC': 'N/A',
                            'Precision': 'N/A',
                            'Recall': 'N/A',
                            'F1': 'N/A'
                        })
                else:
                    # Fallback to old structure
                    results_data = data.get('results', {})
                    asr_data.append({
                        'Model': model_type.replace('_', ' ').title(),
                        'Attack': 'MIA',
                        'ASR': f"{results_data.get('accuracy', 0):.4f}",
                        'Accuracy': f"{results_data.get('accuracy', 0):.4f}",
                        'AUC': f"{results_data.get('auc', 0):.4f}",
                        'Precision': f"{results_data.get('precision', 0):.4f}",
                        'Recall': f"{results_data.get('recall', 0):.4f}",
                        'F1': f"{results_data.get('f1_score', 0):.4f}"
                    })
            
            elif attack_type == "inversion":
                # For inversion, use MSE (lower is better)
                mse = data.get('results', {}).get('mse', 0)
                # Convert MSE to a success rate (lower MSE = higher success)
                # Normalize to 0-1 range, assuming MSE typically ranges 0-10
                asr = max(0, 1 - (mse / 10))
                
                asr_data.append({
                    'Model': model_type.replace('_', ' ').title(),
                    'Attack': 'Inversion',
                    'ASR': f"{asr:.4f}",
                    'Accuracy': 'N/A',
                    'AUC': 'N/A',
                    'Precision': 'N/A',
                    'Recall': 'N/A',
                    'F1': f"MSE: {mse:.4f}"
                })
            
            elif attack_type == "aia":
                # For AIA, use accuracy directly
                accuracy = data.get('results', {}).get('accuracy', 0)
                
                asr_data.append({
                    'Model': model_type.replace('_', ' ').title(),
                    'Attack': 'AIA',
                    'ASR': f"{accuracy:.4f}",
                    'Accuracy': f"{accuracy:.4f}",
                    'AUC': 'N/A',
                    'Precision': 'N/A',
                    'Recall': 'N/A',
                    'F1': f"Improvement: {data.get('results', {}).get('improvement_over_baseline', 0):.4f}"
                })
    
    return asr_data

def create_results_table(asr_data):
    """Create a formatted results table"""
    df = pd.DataFrame(asr_data)
    
    # Create a pivot table for better visualization
    # Use a more flexible approach that doesn't require exact column matching
    pivot_df = df.pivot(index='Model', columns='Attack', values='ASR')
    
    return df, pivot_df

def generate_visualizations(df, pivot_df):
    """Generate visualizations of the results"""
    # Create output directory
    os.makedirs("output/results", exist_ok=True)
    
    # 1. Heatmap of ASR values
    plt.figure(figsize=(12, 8))
    
    # Convert ASR to numeric for plotting
    plot_df = df.copy()
    plot_df['ASR_Numeric'] = pd.to_numeric(plot_df['ASR'].replace('N/A', 0), errors='coerce')
    
    # Create a cleaner pivot for visualization
    # Group by model and attack type, taking the best ASR for each
    viz_data = []
    for model in plot_df['Model'].unique():
        model_data = plot_df[plot_df['Model'] == model]
        
        # For MIA, take the best attack
        mia_data = model_data[model_data['Attack'].str.contains('MIA')]
        if not mia_data.empty:
            best_mia = mia_data.loc[mia_data['ASR_Numeric'].idxmax()]
            viz_data.append({
                'Model': model,
                'Attack': 'MIA (Best)',
                'ASR': best_mia['ASR_Numeric']
            })
        
        # Add other attacks
        for attack in ['Inversion', 'AIA']:
            attack_data = model_data[model_data['Attack'] == attack]
            if not attack_data.empty:
                viz_data.append({
                    'Model': model,
                    'Attack': attack,
                    'ASR': attack_data.iloc[0]['ASR_Numeric']
                })
    
    viz_df = pd.DataFrame(viz_data)
    plot_pivot = viz_df.pivot(index='Model', columns='Attack', values='ASR')
    
    sns.heatmap(plot_pivot, annot=True, cmap='RdYlGn_r', fmt='.4f', 
                cbar_kws={'label': 'Attack Success Rate'})
    plt.title('Attack Success Rate (ASR) Heatmap\nHigher values = More successful attacks', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Model Configuration', fontsize=12)
    plt.tight_layout()
    plt.savefig('output/results/asr_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar chart comparing ASR across models
    plt.figure(figsize=(14, 8))
    
    # Group by model and calculate average ASR
    model_asr = plot_df.groupby('Model')['ASR_Numeric'].mean().reset_index()
    
    bars = plt.bar(model_asr['Model'], model_asr['ASR_Numeric'], 
                   color=['#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Average Attack Success Rate by Model Configuration', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Model Configuration', fontsize=12)
    plt.ylabel('Average ASR', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/results/asr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Detailed comparison table - Use cleaned data
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create cleaned table data (same logic as in save_results_to_files)
    clean_data = []
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        
        # For MIA, take the best attack
        mia_data = model_data[model_data['Attack'].str.contains('MIA')]
        if not mia_data.empty:
            best_mia = mia_data.loc[mia_data['ASR'].replace('N/A', '0').astype(float).idxmax()]
            clean_data.append({
                'Model': model,
                'Attack': 'MIA (Best)',
                'ASR': best_mia['ASR'],
                'Accuracy': best_mia['Accuracy'],
                'AUC': best_mia['AUC'],
                'F1': best_mia['F1']
            })
        
        # Add other attacks
        for attack in ['Inversion', 'AIA']:
            attack_data = model_data[model_data['Attack'] == attack]
            if not attack_data.empty:
                clean_data.append({
                    'Model': model,
                    'Attack': attack,
                    'ASR': attack_data.iloc[0]['ASR'],
                    'Accuracy': attack_data.iloc[0]['Accuracy'],
                    'AUC': attack_data.iloc[0]['AUC'],
                    'F1': attack_data.iloc[0]['F1']
                })
    
    # Create table data from cleaned results
    table_data = []
    for item in clean_data:
        # Format the data for display
        accuracy_display = item['Accuracy'] if item['Accuracy'] != 'N/A' else '-'
        auc_display = item['AUC'] if item['AUC'] != 'N/A' else '-'
        f1_display = item['F1'] if item['F1'] != 'N/A' else '-'
        
        table_data.append([
            item['Model'],
            item['Attack'],
            item['ASR'],
            accuracy_display,
            auc_display,
            f1_display
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'Attack', 'ASR', 'Accuracy', 'AUC', 'F1/MSE'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(6):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('Comprehensive Attack Results Table', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('output/results/detailed_results_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_files(df, pivot_df):
    """Save results to CSV and JSON files"""
    # Save detailed results to CSV
    df.to_csv('output/results/attack_results_detailed.csv', index=False)
    
    # Create a cleaner pivot table for CSV
    # Group by model and attack type, taking the best ASR for each
    clean_data = []
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        
        # For MIA, take the best attack
        mia_data = model_data[model_data['Attack'].str.contains('MIA')]
        if not mia_data.empty:
            best_mia = mia_data.loc[mia_data['ASR'].replace('N/A', '0').astype(float).idxmax()]
            clean_data.append({
                'Model': model,
                'Attack': 'MIA (Best)',
                'ASR': best_mia['ASR'],
                'Accuracy': best_mia['Accuracy'],
                'AUC': best_mia['AUC'],
                'F1': best_mia['F1']
            })
        
        # Add other attacks
        for attack in ['Inversion', 'AIA']:
            attack_data = model_data[model_data['Attack'] == attack]
            if not attack_data.empty:
                clean_data.append({
                    'Model': model,
                    'Attack': attack,
                    'ASR': attack_data.iloc[0]['ASR'],
                    'Accuracy': attack_data.iloc[0]['Accuracy'],
                    'AUC': attack_data.iloc[0]['AUC'],
                    'F1': attack_data.iloc[0]['F1']
                })
    
    clean_df = pd.DataFrame(clean_data)
    clean_pivot = clean_df.pivot(index='Model', columns='Attack', values='ASR')
    
    # Save pivot table to CSV
    clean_pivot.to_csv('output/results/attack_results_pivot.csv')
    
    # Save to JSON
    results_dict = {
        'summary': {
            'total_models': len(df['Model'].unique()),
            'total_attacks': len(clean_df['Attack'].unique()),
            'models': df['Model'].unique().tolist(),
            'attacks': clean_df['Attack'].unique().tolist()
        },
        'detailed_results': df.to_dict('records'),
        'clean_results': clean_df.to_dict('records'),
        'pivot_table': clean_pivot.to_dict()
    }
    
    with open('output/results/attack_results_summary.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

def print_results_table(df):
    """Print a nicely formatted table to console"""
    print("\n" + "="*100)
    print("ATTACK SUCCESS RATE (ASR) RESULTS TABLE")
    print("="*100)
    
    # Create cleaned data for display (same logic as in generate_visualizations)
    clean_data = []
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        
        # For MIA, take the best attack
        mia_data = model_data[model_data['Attack'].str.contains('MIA')]
        if not mia_data.empty:
            best_mia = mia_data.loc[mia_data['ASR'].replace('N/A', '0').astype(float).idxmax()]
            clean_data.append({
                'Model': model,
                'Attack': 'MIA (Best)',
                'ASR': best_mia['ASR'],
                'Accuracy': best_mia['Accuracy'],
                'AUC': best_mia['AUC'],
                'F1': best_mia['F1']
            })
        
        # Add other attacks
        for attack in ['Inversion', 'AIA']:
            attack_data = model_data[model_data['Attack'] == attack]
            if not attack_data.empty:
                clean_data.append({
                    'Model': model,
                    'Attack': attack,
                    'ASR': attack_data.iloc[0]['ASR'],
                    'Accuracy': attack_data.iloc[0]['Accuracy'],
                    'AUC': attack_data.iloc[0]['AUC'],
                    'F1': attack_data.iloc[0]['F1']
                })
    
    # Print cleaned table
    print(f"{'Model':<20} {'Attack':<20} {'ASR':<10} {'Accuracy':<10} {'AUC':<10} {'F1/MSE':<15}")
    print("-"*100)
    
    for item in clean_data:
        # Format the data for display
        accuracy_display = item['Accuracy'] if item['Accuracy'] != 'N/A' else '-'
        auc_display = item['AUC'] if item['AUC'] != 'N/A' else '-'
        f1_display = item['F1'] if item['F1'] != 'N/A' else '-'
        
        print(f"{item['Model']:<20} {item['Attack']:<20} {item['ASR']:<10} "
              f"{accuracy_display:<10} {auc_display:<10} {f1_display:<15}")
    
    print("\n" + "="*100)
    print("SUMMARY:")
    print("="*100)
    
    # Calculate averages using cleaned data
    clean_df = pd.DataFrame(clean_data)
    numeric_asr = pd.to_numeric(clean_df['ASR'].replace('N/A', 0), errors='coerce')
    avg_asr = numeric_asr.mean()
    print(f"Average ASR across all models and attacks: {avg_asr:.4f}")
    
    # Best and worst performing models
    model_avg = clean_df.groupby('Model').apply(
        lambda x: pd.to_numeric(x['ASR'].replace('N/A', 0), errors='coerce').mean()
    )
    
    best_model = model_avg.idxmax()
    worst_model = model_avg.idxmin()
    print(f"Best performing model: {best_model} (ASR: {model_avg[best_model]:.4f})")
    print(f"Worst performing model: {worst_model} (ASR: {model_avg[worst_model]:.4f})")
    
    # Best and worst attacks
    attack_avg = clean_df.groupby('Attack').apply(
        lambda x: pd.to_numeric(x['ASR'].replace('N/A', 0), errors='coerce').mean()
    )
    
    best_attack = attack_avg.idxmax()
    worst_attack = attack_avg.idxmin()
    print(f"Most successful attack: {best_attack} (ASR: {attack_avg[best_attack]:.4f})")
    print(f"Least successful attack: {worst_attack} (ASR: {attack_avg[worst_attack]:.4f})")

def main():
    """Main function to generate the results table"""
    print("Loading attack results...")
    results = load_attack_results()
    
    print("Extracting ASR metrics...")
    asr_data = extract_asr_metrics(results)
    
    print("Creating results table...")
    df, pivot_df = create_results_table(asr_data)
    
    print("Generating visualizations...")
    generate_visualizations(df, pivot_df)
    
    print("Saving results to files...")
    save_results_to_files(df, pivot_df)
    
    print("Printing results table...")
    print_results_table(df)
    
    print("\nResults saved to:")
    print("- output/results/attack_results_detailed.csv")
    print("- output/results/attack_results_pivot.csv")
    print("- output/results/attack_results_summary.json")
    print("- output/results/asr_heatmap.png")
    print("- output/results/asr_comparison.png")
    print("- output/results/detailed_results_table.png")

if __name__ == "__main__":
    main() 