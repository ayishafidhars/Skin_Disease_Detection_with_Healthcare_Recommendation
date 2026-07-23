"""
Data Validation and Cleaning Script for Skin Disease Dataset
Implements data quality checks, handles missing data, and removes noisy samples
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import shutil
from datetime import datetime

print("=" * 70)
print("SKIN DISEASE DATASET - DATA VALIDATION AND CLEANING")
print("=" * 70)

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease')

# Expected image properties
MIN_IMAGE_SIZE = (50, 50)  # Minimum acceptable image dimensions
MAX_IMAGE_SIZE = (5000, 5000)  # Maximum acceptable image dimensions
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
MIN_SAMPLES_PER_CLASS = 5  # Minimum samples required per class

# Results directory
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Quarantine directory for problematic images
QUARANTINE_DIR = os.path.join(BASE_DIR, 'quarantine')
os.makedirs(QUARANTINE_DIR, exist_ok=True)

class DatasetValidator:
    """Validates and cleans the dataset"""
    
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.stats = {
            'train': defaultdict(int),
            'val': defaultdict(int),
            'test': defaultdict(int)
        }
        self.issues = {
            'corrupted': [],
            'wrong_format': [],
            'too_small': [],
            'too_large': [],
            'grayscale': [],
            'duplicate_names': []
        }
        
    def validate_image(self, img_path):
        """Validate a single image file"""
        errors = []
        
        try:
            # Check file extension
            if not any(img_path.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                errors.append('invalid_extension')
                return errors
            
            # Try to open and check the image (single open for efficiency)
            with Image.open(img_path) as img:
                # Check image dimensions
                width, height = img.size
                
                if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                    errors.append('too_small')
                
                if width > MAX_IMAGE_SIZE[0] or height > MAX_IMAGE_SIZE[1]:
                    errors.append('too_large')
                
                # Check if image is grayscale (might be okay, just flagging)
                if img.mode == 'L':
                    errors.append('grayscale')
                
                # Check if image has valid data
                img_array = np.array(img)
                if img_array.size == 0:
                    errors.append('empty_image')
                
                # Check for completely black or white images (potential noise)
                if img.mode == 'RGB':
                    mean_val = np.mean(img_array)
                    if mean_val < 5 or mean_val > 250:
                        errors.append('extreme_brightness')
                        
        except (IOError, OSError, Image.DecompressionBombError) as e:
            errors.append('corrupted')
        except Exception as e:
            errors.append(f'unknown_error: {str(e)}')
            
        return errors
    
    def scan_dataset(self):
        """Scan entire dataset and collect statistics"""
        print("\n" + "=" * 70)
        print("SCANNING DATASET...")
        print("=" * 70)
        
        splits = ['train', 'val', 'test']
        total_images = 0
        total_issues = 0
        
        for split in splits:
            split_dir = os.path.join(self.dataset_dir, split)
            
            if not os.path.exists(split_dir):
                print(f"\n⚠ WARNING: {split} directory not found!")
                continue
            
            print(f"\n📂 Scanning {split.upper()} set...")
            
            # Get all class directories
            class_dirs = [d for d in os.listdir(split_dir) 
                         if os.path.isdir(os.path.join(split_dir, d))]
            
            for class_name in sorted(class_dirs):
                class_path = os.path.join(split_dir, class_name)
                
                # Get all image files
                image_files = [f for f in os.listdir(class_path)
                             if os.path.isfile(os.path.join(class_path, f))]
                
                valid_images = 0
                print(f"   {class_name}: Processing {len(image_files)} images...", end='', flush=True)
                
                for img_file in image_files:
                    img_path = os.path.join(class_path, img_file)
                    total_images += 1
                    
                    # Show progress every 100 images
                    if total_images % 100 == 0:
                        print('.', end='', flush=True)
                    
                    # Validate image
                    errors = self.validate_image(img_path)
                    
                    if not errors:
                        valid_images += 1
                    else:
                        total_issues += 1
                        # Log issues
                        for error in errors:
                            if error == 'corrupted':
                                self.issues['corrupted'].append((split, class_name, img_file))
                            elif error == 'invalid_extension':
                                self.issues['wrong_format'].append((split, class_name, img_file))
                            elif error == 'too_small':
                                self.issues['too_small'].append((split, class_name, img_file))
                            elif error == 'too_large':
                                self.issues['too_large'].append((split, class_name, img_file))
                            elif error == 'grayscale':
                                self.issues['grayscale'].append((split, class_name, img_file))
                
                # Store statistics
                self.stats[split][class_name] = {
                    'total': len(image_files),
                    'valid': valid_images,
                    'invalid': len(image_files) - valid_images
                }
                print(f" ✓ ({valid_images}/{len(image_files)} valid)")
        
        print(f"\n✓ Scan complete!")
        print(f"  Total images scanned: {total_images}")
        print(f"  Total issues found: {total_issues}")
        
        return total_images, total_issues
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)
        
        for split in ['train', 'val', 'test']:
            if not self.stats[split]:
                continue
                
            print(f"\n📊 {split.upper()} SET:")
            print("-" * 70)
            print(f"{'Class Name':<25} {'Total':<10} {'Valid':<10} {'Issues':<10}")
            print("-" * 70)
            
            total_samples = 0
            total_valid = 0
            total_invalid = 0
            
            for class_name in sorted(self.stats[split].keys()):
                stats = self.stats[split][class_name]
                total_samples += stats['total']
                total_valid += stats['valid']
                total_invalid += stats['invalid']
                
                # Warning for classes with too few samples
                warning = " ⚠ LOW" if stats['total'] < MIN_SAMPLES_PER_CLASS else ""
                
                print(f"{class_name:<25} {stats['total']:<10} {stats['valid']:<10} {stats['invalid']:<10}{warning}")
            
            print("-" * 70)
            print(f"{'TOTAL':<25} {total_samples:<10} {total_valid:<10} {total_invalid:<10}")
    
    def print_issues(self):
        """Print detailed issues found"""
        print("\n" + "=" * 70)
        print("ISSUES DETECTED")
        print("=" * 70)
        
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        if total_issues == 0:
            print("\n✓ No issues detected! Dataset is clean.")
            return
        
        print(f"\nTotal issues found: {total_issues}\n")
        
        if self.issues['corrupted']:
            print(f"🔴 CORRUPTED IMAGES: {len(self.issues['corrupted'])}")
            for split, class_name, img_file in self.issues['corrupted'][:10]:
                print(f"   - {split}/{class_name}/{img_file}")
            if len(self.issues['corrupted']) > 10:
                print(f"   ... and {len(self.issues['corrupted']) - 10} more")
        
        if self.issues['wrong_format']:
            print(f"\n🟡 WRONG FORMAT: {len(self.issues['wrong_format'])}")
            for split, class_name, img_file in self.issues['wrong_format'][:10]:
                print(f"   - {split}/{class_name}/{img_file}")
            if len(self.issues['wrong_format']) > 10:
                print(f"   ... and {len(self.issues['wrong_format']) - 10} more")
        
        if self.issues['too_small']:
            print(f"\n🟡 TOO SMALL: {len(self.issues['too_small'])}")
            for split, class_name, img_file in self.issues['too_small'][:10]:
                print(f"   - {split}/{class_name}/{img_file}")
            if len(self.issues['too_small']) > 10:
                print(f"   ... and {len(self.issues['too_small']) - 10} more")
        
        if self.issues['too_large']:
            print(f"\n🟡 TOO LARGE: {len(self.issues['too_large'])}")
            for split, class_name, img_file in self.issues['too_large']:
                print(f"   - {split}/{class_name}/{img_file}")
        
        if self.issues['grayscale']:
            print(f"\n🔵 GRAYSCALE IMAGES: {len(self.issues['grayscale'])} (may be acceptable)")
            if len(self.issues['grayscale']) <= 5:
                for split, class_name, img_file in self.issues['grayscale']:
                    print(f"   - {split}/{class_name}/{img_file}")
    
    def quarantine_problematic_files(self, remove_corrupted=True):
        """Move problematic files to quarantine folder"""
        print("\n" + "=" * 70)
        print("CLEANING DATASET")
        print("=" * 70)
        
        files_moved = 0
        
        if remove_corrupted and self.issues['corrupted']:
            print(f"\n📦 Moving {len(self.issues['corrupted'])} corrupted files to quarantine...")
            
            for split, class_name, img_file in self.issues['corrupted']:
                src_path = os.path.join(self.dataset_dir, split, class_name, img_file)
                dst_dir = os.path.join(QUARANTINE_DIR, split, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, img_file)
                
                try:
                    shutil.move(src_path, dst_path)
                    files_moved += 1
                except Exception as e:
                    print(f"   ⚠ Error moving {img_file}: {e}")
        
        if files_moved > 0:
            print(f"✓ Moved {files_moved} files to quarantine folder: {QUARANTINE_DIR}")
        else:
            print("✓ No files need to be quarantined.")
        
        return files_moved
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "=" * 70)
        print("GENERATING REPORT")
        print("=" * 70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(RESULTS_DIR, f'data_validation_report_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SKIN DISEASE DATASET - VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Location: {self.dataset_dir}\n\n")
            
            # Statistics
            f.write("=" * 70 + "\n")
            f.write("DATASET STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            
            for split in ['train', 'val', 'test']:
                if not self.stats[split]:
                    continue
                
                f.write(f"\n{split.upper()} SET:\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'Class Name':<25} {'Total':<10} {'Valid':<10} {'Issues':<10}\n")
                f.write("-" * 70 + "\n")
                
                total_samples = 0
                total_valid = 0
                total_invalid = 0
                
                for class_name in sorted(self.stats[split].keys()):
                    stats = self.stats[split][class_name]
                    total_samples += stats['total']
                    total_valid += stats['valid']
                    total_invalid += stats['invalid']
                    
                    warning = " [LOW SAMPLES]" if stats['total'] < MIN_SAMPLES_PER_CLASS else ""
                    f.write(f"{class_name:<25} {stats['total']:<10} {stats['valid']:<10} {stats['invalid']:<10}{warning}\n")
                
                f.write("-" * 70 + "\n")
                f.write(f"{'TOTAL':<25} {total_samples:<10} {total_valid:<10} {total_invalid:<10}\n\n")
            
            # Issues
            f.write("\n" + "=" * 70 + "\n")
            f.write("ISSUES DETECTED\n")
            f.write("=" * 70 + "\n\n")
            
            total_issues = sum(len(issues) for issues in self.issues.values())
            f.write(f"Total issues found: {total_issues}\n\n")
            
            for issue_type, issue_list in self.issues.items():
                if issue_list:
                    f.write(f"\n{issue_type.upper().replace('_', ' ')}: {len(issue_list)}\n")
                    for split, class_name, img_file in issue_list:
                        f.write(f"  - {split}/{class_name}/{img_file}\n")
            
            # Recommendations
            f.write("\n" + "=" * 70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 70 + "\n\n")
            
            if self.issues['corrupted']:
                f.write(f"• Remove or replace {len(self.issues['corrupted'])} corrupted images\n")
            
            if self.issues['too_small']:
                f.write(f"• Review {len(self.issues['too_small'])} images that are too small\n")
            
            # Check for class imbalance
            for split in ['train']:
                if self.stats[split]:
                    counts = [self.stats[split][cls]['valid'] for cls in self.stats[split]]
                    if counts:
                        max_samples = max(counts)
                        min_samples = min(counts)
                        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
                        
                        if imbalance_ratio > 3:
                            f.write(f"• Class imbalance detected in {split} set (ratio: {imbalance_ratio:.2f}:1)\n")
                            f.write(f"  Consider using class weights or data augmentation\n")
            
            if total_issues == 0:
                f.write("✓ Dataset is clean and ready for training!\n")
        
        print(f"✓ Report saved to: {report_path}")
        return report_path


def main():
    """Main execution function"""
    print(f"\nDataset directory: {DATASET_DIR}\n")
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ ERROR: Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)
    
    # Create validator
    validator = DatasetValidator(DATASET_DIR)
    
    # Scan dataset
    total_images, total_issues = validator.scan_dataset()
    
    # Print statistics
    validator.print_statistics()
    
    # Print issues
    validator.print_issues()
    
    # Clean dataset (move corrupted files to quarantine)
    if total_issues > 0:
        print("\n" + "=" * 70)
        response = input("Do you want to move corrupted files to quarantine? (y/n): ")
        if response.lower() == 'y':
            validator.quarantine_problematic_files(remove_corrupted=True)
        else:
            print("Skipping cleanup. Corrupted files remain in dataset.")
    
    # Generate report
    report_path = validator.generate_report()
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\n✓ Dataset validated successfully!")
    print(f"✓ Report saved: {report_path}")
    
    if total_issues == 0:
        print("\n🎉 Dataset is clean and ready for training!")
    else:
        print(f"\n⚠ {total_issues} issues found. Please review the report.")


if __name__ == "__main__":
    main()
