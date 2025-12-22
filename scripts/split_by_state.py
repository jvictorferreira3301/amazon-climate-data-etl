#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to split climate CSV files by state.

Input:
    - Climate_Amazon_North_Monthly_2001-2024.csv
    - Climate_Amazon_North_2001-2024.csv

Output:
    - By state (AC, AM, AP, PA, RO, RR, TO):
        - Climate_{STATE}_Monthly_2001-2024.csv
        - Climate_{STATE}_Annual_2001-2024.csv
"""

import sys
from pathlib import Path
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "by_state"

# Input files
MONTHLY_FILE = SCRIPT_DIR / "Climate_Amazon_North_Monthly_2001-2024.csv"
ANNUAL_FILE = SCRIPT_DIR / "Climate_Amazon_North_2001-2024.csv"

# Northern Region States
NORTHERN_STATES = ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO']


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def split_by_state(df: pd.DataFrame, base_name: str, granularity: str) -> dict:
    """
    Split a DataFrame by state and save individual files.
    
    Parameters:
        df: Input DataFrame
        base_name: Base name for output files
        granularity: "Monthly" or "Annual"
    
    Returns:
        Dictionary with record count per state
    """
    results = {}
    
    for state in NORTHERN_STATES:
        df_state = df[df['UF'] == state].copy()
        
        if len(df_state) > 0:
            # Output file name
            filename = f"{base_name}_{state}_{granularity}_2001-2024.csv"
            filepath = OUTPUT_DIR / filename
            
            # Save CSV
            df_state.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            results[state] = {
                'records': len(df_state),
                'municipalities': df_state['CD_MUN'].nunique(),
                'file': filename
            }
            
            print(f"   ✅ {state}: {len(df_state):,} records | "
                  f"{df_state['CD_MUN'].nunique()} municipalities → {filename}")
        else:
            print(f"   ⚠️ {state}: No records found")
    
    return results


def main():
    """
    Main function.
    """
    print("=" * 70)
    print("📂 SPLITTING FILES BY STATE")
    print("=" * 70)
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print()
    
    # -------------------------------------------------------------------------
    # 1. Process MONTHLY file
    # -------------------------------------------------------------------------
    if MONTHLY_FILE.exists():
        print(f"📊 Processing MONTHLY file: {MONTHLY_FILE.name}")
        df_monthly = pd.read_csv(MONTHLY_FILE)
        print(f"   Total: {len(df_monthly):,} records")
        print()
        split_by_state(df_monthly, "Climate", "Monthly")
        print()
    else:
        print(f"⚠️ File not found: {MONTHLY_FILE}")
        print()
    
    # -------------------------------------------------------------------------
    # 2. Process ANNUAL file
    # -------------------------------------------------------------------------
    if ANNUAL_FILE.exists():
        print(f"📊 Processing ANNUAL file: {ANNUAL_FILE.name}")
        df_annual = pd.read_csv(ANNUAL_FILE)
        print(f"   Total: {len(df_annual):,} records")
        print()
        split_by_state(df_annual, "Climate", "Annual")
        print()
    else:
        print(f"⚠️ File not found: {ANNUAL_FILE}")
        print()
    
    # -------------------------------------------------------------------------
    # 3. Final summary
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("✅ PROCESSING COMPLETED!")
    print("=" * 70)
    
    # List generated files
    generated_files = list(OUTPUT_DIR.glob("*.csv"))
    print(f"\n📁 {len(generated_files)} files generated in: {OUTPUT_DIR}")
    for f in sorted(generated_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   • {f.name} ({size_mb:.2f} MB)")
    
    print()
    print("🏁 End of processing.")


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()
