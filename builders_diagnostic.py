#!/usr/bin/env python3
"""
Builders.py Diagnostic Tool
Analyzes the builders.py file to identify import and syntax issues
"""

import os
import sys
import ast
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any


class BuildersDiagnostic:
    """Diagnostic tool for builders.py file"""
    
    def __init__(self, builders_file_path: str):
        self.builders_path = Path(builders_file_path)
        self.results = {
            'file_exists': False,
            'syntax_valid': False,
            'imports_status': {},
            'class_definitions': [],
            'function_definitions': [],
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
    
    def check_file_existence(self) -> bool:
        """Check if builders.py exists"""
        exists = self.builders_path.exists()
        self.results['file_exists'] = exists
        
        if exists:
            size = self.builders_path.stat().st_size
            print(f"✅ File exists: {self.builders_path}")
            print(f"   Size: {size:,} bytes")
        else:
            print(f"❌ File not found: {self.builders_path}")
            self.results['errors'].append("File does not exist")
        
        return exists
    
    def check_syntax(self) -> bool:
        """Check Python syntax"""
        print("\n" + "=" * 60)
        print("SYNTAX CHECK")
        print("=" * 60)
        
        if not self.results['file_exists']:
            print("❌ Cannot check syntax - file does not exist")
            return False
        
        try:
            with open(self.builders_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to check syntax
            ast.parse(content)
            
            self.results['syntax_valid'] = True
            print("✅ Syntax is valid")
            return True
            
        except SyntaxError as e:
            self.results['syntax_valid'] = False
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            self.results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            
            # Show problematic line
            try:
                lines = content.split('\n')
                if e.lineno <= len(lines):
                    print(f"   Problematic line: {lines[e.lineno-1].strip()}")
            except:
                pass
            
            return False
            
        except Exception as e:
            self.results['syntax_valid'] = False
            error_msg = f"Error reading file: {e}"
            self.results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def analyze_imports(self) -> Dict[str, Any]:
        """Analyze import statements"""
        print("\n" + "=" * 60)
        print("IMPORT ANALYSIS")
        print("=" * 60)
        
        if not self.results['syntax_valid']:
            print("❌ Cannot analyze imports - syntax errors exist")
            return {}
        
        try:
            with open(self.builders_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            import_info = {}
            
            # Find all import statements
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.name
                        import_info[import_name] = {
                            'type': 'import',
                            'line': node.lineno,
                            'available': self._check_module_availability(import_name)
                        }
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        import_info[full_name] = {
                            'type': 'from_import',
                            'module': module,
                            'name': alias.name,
                            'line': node.lineno,
                            'available': self._check_module_availability(module) if module else False
                        }
            
            # Display results
            print("Found imports:")
            for import_name, info in import_info.items():
                status = "✅" if info['available'] else "❌"
                print(f"  {status} Line {info['line']}: {import_name}")
                if not info['available']:
                    self.results['errors'].append(f"Import not available: {import_name}")
            
            self.results['imports_status'] = import_info
            return import_info
            
        except Exception as e:
            error_msg = f"Error analyzing imports: {e}"
            self.results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return {}
    
    def _check_module_availability(self, module_name: str) -> bool:
        """Check if a module is available for import"""
        if not module_name:
            return False
        
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def analyze_classes_and_functions(self) -> Tuple[List[str], List[str]]:
        """Analyze class and function definitions"""
        print("\n" + "=" * 60)
        print("CLASS AND FUNCTION ANALYSIS")
        print("=" * 60)
        
        if not self.results['syntax_valid']:
            print("❌ Cannot analyze - syntax errors exist")
            return [], []
        
        try:
            with open(self.builders_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                elif isinstance(node, ast.FunctionDef):
                    # Only top-level functions (not methods)
                    if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                             if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                        functions.append({
                            'name': node.name,
                            'line': node.lineno
                        })
            
            print("Classes found:")
            for cls_info in classes:
                print(f"  📦 {cls_info['name']} (line {cls_info['line']}) - {len(cls_info['methods'])} methods")
                for method in cls_info['methods']:
                    print(f"     - {method}()")
            
            print(f"\nTop-level functions found:")
            for func_info in functions:
                print(f"  🔧 {func_info['name']}() (line {func_info['line']})")
            
            self.results['class_definitions'] = classes
            self.results['function_definitions'] = functions
            
            return classes, functions
            
        except Exception as e:
            error_msg = f"Error analyzing classes/functions: {e}"
            self.results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return [], []
    
    def check_specific_pytorch_issues(self) -> List[str]:
        """Check for specific PyTorch-related issues"""
        print("\n" + "=" * 60)
        print("PYTORCH SPECIFIC ISSUES")
        print("=" * 60)
        
        pytorch_issues = []
        
        if not self.results['file_exists']:
            return pytorch_issues
        
        try:
            with open(self.builders_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check for nn usage without import
            for i, line in enumerate(lines, 1):
                if 'nn.' in line and 'import' not in line:
                    # Check if nn is imported somewhere
                    nn_imported = any('import torch.nn' in l or 'from torch import nn' in l 
                                    for l in lines[:i])
                    if not nn_imported:
                        issue = f"Line {i}: 'nn' used but not imported: {line.strip()}"
                        pytorch_issues.append(issue)
                        print(f"❌ {issue}")
            
            # Check for PyTorch import issues
            pytorch_imports = ['torch', 'torch.nn', 'torch.optim']
            for import_name in pytorch_imports:
                available = self._check_module_availability(import_name)
                if not available:
                    issue = f"PyTorch module not available: {import_name}"
                    pytorch_issues.append(issue)
                    print(f"❌ {issue}")
            
            # Check for type hints with undefined types
            for i, line in enumerate(lines, 1):
                if '-> Tuple[nn.Module' in line:
                    nn_imported = any('import torch.nn' in l or 'from torch import nn' in l 
                                    for l in lines[:i])
                    if not nn_imported:
                        issue = f"Line {i}: Type hint uses 'nn.Module' but nn not imported"
                        pytorch_issues.append(issue)
                        print(f"❌ {issue}")
            
            if not pytorch_issues:
                print("✅ No PyTorch-specific issues found")
            
            return pytorch_issues
            
        except Exception as e:
            error_msg = f"Error checking PyTorch issues: {e}"
            print(f"❌ {error_msg}")
            return [error_msg]
    
    def check_tensorflow_issues(self) -> List[str]:
        """Check for TensorFlow-related issues"""
        print("\n" + "=" * 60)
        print("TENSORFLOW SPECIFIC ISSUES")
        print("=" * 60)
        
        tf_issues = []
        
        if not self.results['file_exists']:
            return tf_issues
        
        try:
            # Check TensorFlow availability
            tf_available = self._check_module_availability('tensorflow')
            if not tf_available:
                issue = "TensorFlow not available"
                tf_issues.append(issue)
                print(f"❌ {issue}")
            else:
                print("✅ TensorFlow is available")
            
            # Check for TensorFlow import patterns
            with open(self.builders_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'tensorflow' in content or 'keras' in content:
                print("✅ TensorFlow/Keras imports found in file")
            else:
                print("ℹ️  No TensorFlow/Keras imports found")
            
            return tf_issues
            
        except Exception as e:
            error_msg = f"Error checking TensorFlow issues: {e}"
            print(f"❌ {error_msg}")
            return [error_msg]
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        # File existence
        if not self.results['file_exists']:
            recommendations.append("❌ Create the builders.py file")
            return recommendations
        
        # Syntax issues
        if not self.results['syntax_valid']:
            recommendations.append("❌ Fix syntax errors before proceeding")
            return recommendations
        
        # Import issues
        failed_imports = [name for name, info in self.results['imports_status'].items() 
                         if not info['available']]
        
        if failed_imports:
            if any('torch' in imp for imp in failed_imports):
                recommendations.append("🔧 Install PyTorch: pip install torch")
            if any('tensorflow' in imp for imp in failed_imports):
                recommendations.append("🔧 Install TensorFlow: pip install tensorflow")
        
        # PyTorch-specific recommendations
        pytorch_issues = self.check_specific_pytorch_issues()
        if pytorch_issues:
            recommendations.append("🔧 Fix PyTorch import issues:")
            for issue in pytorch_issues:
                recommendations.append(f"   - {issue}")
        
        # TensorFlow-specific recommendations
        tf_issues = self.check_tensorflow_issues()
        if tf_issues:
            recommendations.append("🔧 Fix TensorFlow issues:")
            for issue in tf_issues:
                recommendations.append(f"   - {issue}")
        
        # General recommendations
        if len(self.results['class_definitions']) == 0:
            recommendations.append("⚠️  No classes found - check if file is complete")
        
        if len(self.results['errors']) == 0:
            recommendations.append("✅ File appears to be structurally sound")
        
        self.results['recommendations'] = recommendations
        
        for rec in recommendations:
            print(rec)
        
        return recommendations
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete diagnostic"""
        print("Builders.py Diagnostic Tool")
        print(f"Analyzing: {self.builders_path}")
        print("=" * 60)
        
        # Run all checks
        self.check_file_existence()
        
        if self.results['file_exists']:
            self.check_syntax()
            if self.results['syntax_valid']:
                self.analyze_imports()
                self.analyze_classes_and_functions()
        
        self.generate_recommendations()
        
        # Summary
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"File exists: {'✅' if self.results['file_exists'] else '❌'}")
        print(f"Syntax valid: {'✅' if self.results['syntax_valid'] else '❌'}")
        print(f"Import issues: {len([i for i in self.results['imports_status'].values() if not i['available']])}")
        print(f"Total errors: {len(self.results['errors'])}")
        print(f"Classes found: {len(self.results['class_definitions'])}")
        print(f"Functions found: {len(self.results['function_definitions'])}")
        
        return self.results
    
    def save_diagnostic_report(self, output_file: str = "builders_diagnostic.txt"):
        """Save diagnostic report to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Builders.py Diagnostic Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Summary:\n")
            f.write(f"- File exists: {self.results['file_exists']}\n")
            f.write(f"- Syntax valid: {self.results['syntax_valid']}\n")
            f.write(f"- Import issues: {len([i for i in self.results['imports_status'].values() if not i['available']])}\n")
            f.write(f"- Total errors: {len(self.results['errors'])}\n\n")
            
            if self.results['errors']:
                f.write("Errors:\n")
                for error in self.results['errors']:
                    f.write(f"- {error}\n")
                f.write("\n")
            
            if self.results['recommendations']:
                f.write("Recommendations:\n")
                for rec in self.results['recommendations']:
                    f.write(f"- {rec}\n")
        
        print(f"\n📄 Diagnostic report saved to: {output_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose builders.py file issues")
    parser.add_argument('--file', default='ANNrun_code/core/models/neural_networks/builders.py',
                       help='Path to builders.py file')
    parser.add_argument('--save-report', action='store_true',
                       help='Save diagnostic report to file')
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnostic = BuildersDiagnostic(args.file)
    results = diagnostic.run_full_diagnostic()
    
    # Save report if requested
    if args.save_report:
        diagnostic.save_diagnostic_report()
    
    # Exit with error code if issues found
    if results['errors']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()