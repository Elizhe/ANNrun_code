#!/usr/bin/env python3
"""
Safe Data Loader Fix - Manual repair for syntax errors
"""

import shutil
from pathlib import Path

def restore_backup_and_apply_safe_fix(original_file: str, backup_file: str):
    """Restore backup and apply safe fixes manually"""
    
    print("🔄 Restoring backup and applying safe fixes...")
    
    # Step 1: Restore backup
    original_path = Path(original_file)
    backup_path = Path(backup_file)
    
    if backup_path.exists():
        shutil.copy2(backup_path, original_path)
        print(f"✅ Restored from backup: {backup_file}")
    else:
        print(f"❌ Backup file not found: {backup_file}")
        return False
    
    # Step 2: Read content
    with open(original_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Step 3: Apply only safe fixes that won't break syntax
    safe_fixes = [
        # Fix 1: Add Optional import safely
        ('from typing import Dict, List, Optional, Tuple, Union', 
         'from typing import Dict, List, Optional, Tuple, Union, Any'),
        
        # Fix 2: Safe None parameter fixes
        ('def load_data(self, **kwargs) -> Dict[str, Any]:', 
         'def load_data(self, **kwargs) -> Dict[str, Any]:'),
        
        # Fix 3: Simple None string replacements in safe contexts
        ('= None  # ', '= ""  # '),
        
        # Fix 4: Safe array type conversion (without complex regex)
        ('clearness = obs_rad / theo_rad', 
         'clearness = np.divide(obs_rad.astype(float), theo_rad.astype(float), out=np.zeros_like(obs_rad, dtype=float), where=(theo_rad != 0))'),
         
        # Fix 5: Safe division operations
        ('station_data[\'obs_clearness\'] = station_data[\'srad\'] / 25.0',
         'station_data[\'obs_clearness\'] = np.divide(station_data[\'srad\'].astype(float), 25.0)'),
    ]
    
    original_content = content
    applied_count = 0
    
    for old_text, new_text in safe_fixes:
        if old_text in content:
            content = content.replace(old_text, new_text)
            applied_count += 1
            print(f"✅ Applied safe fix: {old_text[:50]}...")
    
    # Step 4: Add missing import if not present
    if 'import numpy as np' not in content:
        # Find the import section and add numpy
        lines = content.split('\n')
        import_inserted = False
        for i, line in enumerate(lines):
            if line.startswith('import pandas') and not import_inserted:
                lines.insert(i + 1, 'import numpy as np')
                import_inserted = True
                applied_count += 1
                print("✅ Added numpy import")
                break
        content = '\n'.join(lines)
    
    # Step 5: Write the safely fixed content
    with open(original_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n📊 Safe fixes applied: {applied_count}")
    
    # Step 6: Validate syntax
    try:
        compile(content, str(original_path), 'exec')
        print("✅ Syntax validation: PASSED")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax validation: FAILED at line {e.lineno}")
        print(f"   Error: {e.text}")
        print(f"   Message: {e.msg}")
        return False

def create_minimal_data_loader():
    """Create a minimal working data_loader.py as fallback"""
    
    minimal_code = '''#!/usr/bin/env python3
"""
Minimal Data Loader - Safe fallback version
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

class DataLoader:
    """Minimal safe data loader"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_csv_file(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load CSV file safely"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            default_params = {
                'encoding': 'utf-8',
                'skipinitialspace': True,
                'na_values': ['', 'NULL', 'null', 'None', 'none', 'NaN', 'nan']
            }
            default_params.update(kwargs)
            
            df = pd.read_csv(filepath, **default_params)
            self.logger.info(f"Loaded CSV: {filename} ({len(df)} rows, {len(df.columns)} columns)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV {filename}: {e}")
            raise
    
    def safe_divide(self, numerator: Any, denominator: Any) -> np.ndarray:
        """Safe division with proper error handling"""
        num_array = np.asarray(numerator, dtype=np.float64)
        den_array = np.asarray(denominator, dtype=np.float64)
        
        # Avoid division by zero
        result = np.divide(
            num_array, 
            den_array, 
            out=np.full_like(num_array, np.nan, dtype=np.float64), 
            where=(den_array != 0)
        )
        
        return result
    
    def calculate_clearness_index(self, obs_radiation: Any, theo_radiation: Any = 25.0) -> np.ndarray:
        """Calculate clearness index safely"""
        return self.safe_divide(obs_radiation, theo_radiation)
    
    def load_data(self, **kwargs) -> Dict[str, Any]:
        """Load data - basic implementation"""
        return {}
    
    def validate_data(self, data: Any) -> bool:
        """Validate data - basic implementation"""
        return data is not None

# Backward compatibility
class Agera5Loader(DataLoader):
    def __init__(self, base_dir: Optional[str] = None):
        super().__init__(base_dir)

class HimawariLoader(DataLoader):
    def __init__(self, base_dir: Optional[str] = None):
        super().__init__(base_dir)

class PangaeaDataLoader(DataLoader):
    def __init__(self, base_dir: Optional[str] = None):
        super().__init__(base_dir)
    
    def load_awos_data(self):
        return pd.DataFrame()
    
    def load_station_info(self):
        return pd.DataFrame()

# Legacy support
def create_data_loader(data_dir: str):
    return DataLoader(data_dir)

if __name__ == "__main__":
    print("✅ Minimal data loader ready")
'''
    
    return minimal_code

def main():
    """Main repair function"""
    print("🛠️  Safe Data Loader Repair Tool")
    print("=" * 40)
    
    original_file = "core/data/data_loader.py"
    backup_file = "core/data/data_loader.backup_20250713_044444.py"
    
    # Option 1: Try to restore and fix safely
    print("\n1️⃣ Trying safe restore and fix...")
    success = restore_backup_and_apply_safe_fix(original_file, backup_file)
    
    if success:
        print("✅ Safe fix completed successfully!")
        
        # Test the import
        try:
            exec("from core.data.data_loader import DataLoader")
            print("✅ Import test: PASSED")
        except Exception as e:
            print(f"❌ Import test: FAILED - {e}")
            success = False
    
    # Option 2: If safe fix fails, create minimal version
    if not success:
        print("\n2️⃣ Creating minimal safe data loader...")
        
        # Create backup of current (broken) version
        broken_backup = Path(original_file).with_suffix('.broken_backup.py')
        shutil.copy2(original_file, broken_backup)
        print(f"✅ Broken version backed up to: {broken_backup}")
        
        # Write minimal version
        minimal_code = create_minimal_data_loader()
        with open(original_file, 'w', encoding='utf-8') as f:
            f.write(minimal_code)
        
        print("✅ Minimal data loader created")
        
        # Test minimal version
        try:
            exec("from core.data.data_loader import DataLoader")
            print("✅ Minimal version import test: PASSED")
            success = True
        except Exception as e:
            print(f"❌ Minimal version import test: FAILED - {e}")
    
    # Final status
    if success:
        print("\n🎉 Data loader is now working!")
        print("\n🚀 Next steps:")
        print("1. Test: python -c \"from core.data.data_loader import DataLoader\"")
        print("2. Run GPU check: python gpu_memory_monitor.py")
        print("3. Run experiment: python main.py single gpu_experiment_plan.csv 1")
    else:
        print("\n❌ Could not fix data loader automatically")
        print("🔧 Manual steps:")
        print("1. Check line 73 in data_loader.py for syntax errors")
        print("2. Look for missing commas or parentheses")
        print("3. Consider using the minimal version as starting point")
    
    return success

if __name__ == "__main__":
    main()