import os
import shutil

base_dir = "./"
all_dirs = [d for d in os.listdir(base_dir) if d.startswith("LegEnv_Jun") and os.path.isdir(os.path.join(base_dir, d))]

# 提取 key 为 constant_... 这一部分
grouped = {}
for folder in all_dirs:
    parts = folder.split("_")
    date = parts[1]
    key = "_".join(parts[2:])
    if key not in grouped:
        grouped[key] = {}
    grouped[key][date] = folder

# 合并逻辑
for key, date_map in grouped.items():
    jun15_folder = date_map.get("Jun15")
    jun14_folder = date_map.get("Jun14")

    if jun15_folder and jun14_folder:
        src = os.path.join(base_dir, jun15_folder)
        dst = os.path.join(base_dir, jun14_folder)

        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)

            if os.path.isdir(src_item):
                if os.path.exists(dst_item):
                    shutil.rmtree(dst_item)
                shutil.copytree(src_item, dst_item)
            else:
                shutil.copy2(src_item, dst_item)

        print(f"✅ 合并 {jun15_folder} -> {jun14_folder}")

    elif jun15_folder and not jun14_folder:
        # 若没有 Jun14 文件夹，就把 Jun15 改名为 Jun14
        new_name = os.path.join(base_dir, f"LegEnv_Jun14_{key}")
        os.rename(os.path.join(base_dir, jun15_folder), new_name)
        print(f"📦 重命名 {jun15_folder} -> LegEnv_Jun14_{key}")
