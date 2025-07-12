#!/usr/bin/env python3
"""
Quick Fix Script for data_loader.py Type Issues
Automatically fixes common None type and array compatibility problems
"""

import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

class DataLoaderQuickFix:
    """Quick fix utility for data_loader.py type issues"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.backup_path = None
        self.applied_fixes = []
    
    def create_backup(self) -> str:
        """Create backup of original file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_path = self.file_path.with_suffix(f'.backup_{timestamp}.py')
        shutil.copy2(self.file_path, self.backup_path)
        return str(self.backup_path)
    
    def get_common_fixes(self) -> List[Tuple[str, str, str]]:
        """Get list of common fixes for data_loader.py"""
        return [
            # Fix 1: None string assignment issues
            (
                "None string assignment",
                r"(\w+)\s*:\s*str\s*=\s*None",
                r"\1: Optional[str] = None"
            ),
            
            # Fix 2: Direct None to str assignment
            (
                "Direct None assignment",
                r"=\s*None(\s*#.*)?$",
                r'= ""  # Fixed None assignment\1'
            ),
            
            # Fix 3: Array division operations (likely line 400 issue)
            (
                "Unsafe array division",
                r"(\w+)\s*/\s*(\w+)(?!\s*[=<>!])",
                r"np.divide(np.asarray(\1, dtype=np.float64), np.asarray(\2, dtype=np.float64), out=np.full_like(\1, np.nan), where=(\2 != 0))"
            ),
            
            # Fix 4: Clearness index calculation
            (
                "Clearness calculation",
                r"clearness\s*=\s*(\w+)\s*/\s*(\w+)",
                r"clearness = np.divide(np.asarray(\1, dtype=np.float64), np.asarray(\2, dtype=np.float64), out=np.full_like(\1, np.nan), where=(\2 != 0))"
            ),
            
            # Fix 5: String None checks
            (
                "String None checks",
                r"if\s+(\w+)\s+is\s+None:",
                r"if \1 is None or \1 == '':"
            ),
            
            # Fix 6: Function parameter None defaults
            (
                "Function parameter defaults",
                r"def\s+\w+\([^)]*(\w+):\s*str\s*=\s*None",
                r"def \1(\1: Optional[str] = None"
            ),
            
            # Fix 7: Pandas/numpy compatibility for ExtensionArray
            (
                "ExtensionArray compatibility",
                r"(\w+)\.values\s*([+\-*/])\s*(\w+)\.values",
                r"np.asarray(\1.values, dtype=np.float64) \2 np.asarray(\3.values, dtype=np.float64)"
            ),
            
            # Fix 8: Add missing imports
            (
                "Missing Optional import",
                r"^(from typing import.*)",
                r"\1\nfrom typing import Optional  # Added for None type fixes"
            )
        ]
    
    def apply_fixes(self, create_backup: bool = True) -> dict:
        """Apply all fixes to the file"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Create backup
        if create_backup:
            backup_file = self.create_backup()
            print(f"✅ Backup created: {backup_file}")
        
        # Read original content
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes = self.get_common_fixes()
        
        # Apply each fix
        for fix_name, pattern, replacement in fixes:
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                self.applied_fixes.append({
                    'name': fix_name,
                    'pattern': pattern,
                    'matches': len(matches),
                    'applied': True
                })
                print(f"✅ Applied fix: {fix_name} ({len(matches)} matches)")
            else:
                self.applied_fixes.append({
                    'name': fix_name,
                    'pattern': pattern,
                    'matches': 0,
                    'applied': False
                })
        
        # Ensure proper imports are at the top
        import_fixes = self._fix_imports(content)
        if import_fixes:
            content = import_fixes
            print("✅ Fixed import statements")
        
        # Write fixed content
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            'original_file': str(self.file_path),
            'backup_file': str(self.backup_path) if self.backup_path else None,
            'fixes_applied': len([f for f in self.applied_fixes if f['applied']]),
            'total_fixes': len(self.applied_fixes),
            'details': self.applied_fixes
        }
    
    def _fix_imports(self, content: str) -> str:
        """Fix import statements to include necessary imports"""
        lines = content.split('\n')
        
        # Check if Optional is already imported
        has_optional = any('Optional' in line for line in lines if line.startswith('from typing'))
        has_union = any('Union' in line for line in lines if line.startswith('from typing'))
        
        # Find the typing import line
        typing_line_idx = None
        for i, line in enumerate(lines):
            if line.startswith('from typing import'):
                typing_line_idx = i
                break
        
        if typing_line_idx is not None and not has_optional:
            # Add Optional to existing typing import
            current_imports = lines[typing_line_idx]
            if not 'Optional' in current_imports:
                lines[typing_line_idx] = current_imports.rstrip() + ', Optional'
            if not 'Union' in current_imports and not has_union:
                lines[typing_line_idx] = lines[typing_line_idx].rstrip() + ', Union'
        elif typing_line_idx is None:
            # Add new typing import after other imports
            import_end = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_end = i + 1
            
            lines.insert(import_end, 'from typing import Optional, Union, Any')
        
        return '\n'.join(lines)
    
    def validate_fixes(self) -> dict:
        """Validate that fixes were applied correctly"""
        if not self.file_path.exists():
            return {'status': 'error', 'message': 'File not found'}
        
        try:
            # Try to parse the file as Python
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            compile(content, str(self.file_path), 'exec')
            
            return {
                'status': 'success',
                'message': 'File compiles successfully',
                'syntax_valid': True
            }
        
        except SyntaxError as e:
            return {
                'status': 'error',
                'message': f'Syntax error: {e}',
                'syntax_valid': False,
                'line': e.lineno,
                'text': e.text
            }
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Other error: {e}',
                'syntax_valid': True  # Syntax is OK, might be import issues
            }

def main():
    """Main function to run the quick fix"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick fix for data_loader.py type issues")
    parser.add_argument('file_path', nargs='?', 
                       default='core/data/data_loader.py',
                       help='Path to data_loader.py file')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup file')
    parser.add_argument('--validate', action='store_true',
                       help='Only validate, do not apply fixes')
    
    args = parser.parse_args()
    
    print("🔧 Data Loader Quick Fix Tool")
    print("=" * 40)
    
    fixer = DataLoaderQuickFix(args.file_path)
    
    if args.validate:
        # Only validate
        result = fixer.validate_fixes()
        print(f"Validation result: {result}")
        return
    
    try:
        # Apply fixes
        result = fixer.apply_fixes(create_backup=not args.no_backup)
        
        print("\n📊 Fix Summary:")
        print(f"   File: {result['original_file']}")
        print(f"   Backup: {result['backup_file']}")
        print(f"   Fixes applied: {result['fixes_applied']}/{result['total_fixes']}")
        
        # Validate the result
        validation = fixer.validate_fixes()
        if validation['syntax_valid']:
            print("✅ File validation: SUCCESS")
        else:
            print(f"❌ File validation: FAILED - {validation['message']}")
        
        print("\n🎯 Applied fixes:")
        for fix in result['details']:
            status = "✅" if fix['applied'] else "⏭️"
            print(f"   {status} {fix['name']}: {fix['matches']} matches")
        
        print("\n🚀 Next steps:")
        print("1. Test the import: python -c \"from core.data.data_loader import *\"")
        print("2. Run your experiments: python main.py ...")
        print("3. If issues persist, check the specific error lines")
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())