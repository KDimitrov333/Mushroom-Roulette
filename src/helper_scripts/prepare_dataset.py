import os
import shutil

def build_binary_dataset():
    source_dir = "./data/raw_mushrooms/MO_94/"
    dest_dir = "./data/binary_mushrooms/"
    
    safe_dir = os.path.join(dest_dir, "Safe")
    unsafe_dir = os.path.join(dest_dir, "Unsafe")
    
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(unsafe_dir, exist_ok=True)

    safe_edibles = [
        "Grifola frondosa", "Laetiporus sulphureus", "Pleurotus ostreatus", 
        "Pleurotus pulmonarius", "Coprinus comatus", "Hericium erinaceus", 
        "Hericium coralloides", "Clitocybe nuda", "Agaricus augustus", 
        "Cantharellus cibarius", "Cantharellus californicus", "Cantharellus cinnabarinus",
        "Armillaria mellea", "Armillaria tabescens", "Flammulina velutipes", 
        "Lycoperdon perlatum", "Lycoperdon pyriforme", "Hypomyces lactifluorum",
        "Cerioporus squamosus", "Amanita velosa", "Amanita calyptroderma"
    ]
    
    if not os.path.exists(source_dir):
        print(f"Cannot find source directory {source_dir}")
        return

    folders = os.listdir(source_dir)
    safe_count = 0
    unsafe_count = 0

    for folder_name in folders:
        source_folder = os.path.join(source_dir, folder_name)
        
        if not os.path.isdir(source_folder):
            continue
            
        is_safe = folder_name in safe_edibles
        target_dir = safe_dir if is_safe else unsafe_dir
        
        if is_safe:
            safe_count += 1
        else:
            unsafe_count += 1
            
        print(f"Extracting {folder_name} to {target_dir}")
        
        for filename in os.listdir(source_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Rename the file to prevent overwriting (e.g., Amanita_muscaria_img01.jpg)
                new_filename = f"{folder_name}_{filename}"
                src_file = os.path.join(source_folder, filename)
                dst_file = os.path.join(target_dir, new_filename)
                
                shutil.copy2(src_file, dst_file)

    total_processed = safe_count + unsafe_count
    
    if total_processed != 94:
        print(f"Expected 94 species folders, but processed {total_processed}.")
        raise ValueError("Failed to properly copy images.")
    
    print(f"Safe Species: {safe_count}")
    print(f"Unsafe Species: {unsafe_count}")


if __name__ == "__main__":
    build_binary_dataset()