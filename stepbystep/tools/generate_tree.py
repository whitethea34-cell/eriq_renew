#!/usr/bin/env python3
import os
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path

def get_tree(root_path, max_depth=3, exclude_regex=None, verbose=False):
    root = Path(root_path).resolve()
    tree_lines = []
    full_lines = []
    dir_count = 0
    file_count = 0
    
    default_exclude = [r'\.git', r'node_modules', r'\.DS_Store', r'.*\.log$', r'__pycache__', r'.*\.bak$']
    if exclude_regex:
        default_exclude.append(exclude_regex)
    
    combined_exclude = re.compile('|'.join(f'({x})' for x in default_exclude))

    def walk(current_path, depth):
        nonlocal dir_count, file_count
        try:
            items = sorted(list(current_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            if verbose: print(f"[Warning] Permission denied: {current_path}")
            return []

        lines = []
        for item in items:
            if combined_exclude.search(item.name) or combined_exclude.search(item.relative_to(root).as_posix()):
                if verbose: print(f"[Debug] Excluded: {item}")
                continue
            
            indent = "  " * depth
            symlink_str = f" -> {item.resolve()}" if item.is_symlink() else ""
            name = f"{item.name}/" if item.is_dir() else item.name
            line = f"{indent}{name}{symlink_str}"
            
            full_lines.append(line)
            
            if item.is_dir():
                dir_count += 1
                if depth < max_depth:
                    lines.append(line)
                    lines.extend(walk(item, depth + 1))
                else:
                    lines.append(f"{indent}{item.name}/ ...")
            else:
                file_count += 1
                lines.append(line)
        return lines

    tree_lines = walk(root, 0)
    return tree_lines, full_lines, dir_count, file_count

def update_readme(readme_path, tree_content, full_table, root_path, dir_count, file_count):
    readme_path = Path(readme_path).resolve()
    timestamp = datetime.now().isoformat(timespec='seconds')
    
    header = f"<!-- START_TREE -->\n### 4.1 自动生成目录树\n\n> 自动生成于: `{timestamp}`\n> 项目根路径: `{root_path}`\n\n```text\n"
    footer = f"\n```\n\n**统计**: 共 {dir_count} 个目录，{file_count} 个文件\n<!-- END_TREE -->"
    
    tree_block = header + "\n".join(tree_content) + footer
    
    appendix_header = "<!-- START_APPENDIX -->\n\n### 4.2 完整深度展开对照表\n\n| 层级路径 | 类型 |\n| :--- | :--- |\n"
    appendix_rows = []
    for line in full_table:
        name = line.strip()
        depth = (len(line) - len(name)) // 2
        appendix_rows.append(f"| {'--' * depth} {name} | {'目录' if name.endswith('/') else '文件'} |")
    
    appendix_block = appendix_header + "\n".join(appendix_rows) + "\n<!-- END_APPENDIX -->"

    bak_path = readme_path.with_suffix('.md.bak')
    # Check if tree or appendix content changed (excluding timestamp)
    if readme_path.exists():
        old_content = readme_path.read_text(encoding='utf-8')
        old_tree_match = re.search(r"```text\n(.*?)\n```", old_content, flags=re.DOTALL)
        old_appendix_match = re.search(r"### 7.1 完整深度展开对照表\n\n(.*?)\n<!-- END_APPENDIX -->", old_content, flags=re.DOTALL)
        
        new_tree_str = "\n".join(tree_content)
        new_appendix_str = "| 层级路径 | 类型 |\n| :--- | :--- |\n" + "\n".join(appendix_rows)
        
        if old_tree_match and old_appendix_match:
            if old_tree_match.group(1).strip() == new_tree_str.strip() and \
               old_appendix_match.group(1).strip() == new_appendix_str.strip():
                print("内容无变化（目录结构未变），跳过写入。")
                return False

    if not readme_path.exists():
        new_content = tree_block + "\n\n" + appendix_block
    else:
        content = readme_path.read_text(encoding='utf-8')
        
        # Backup
        import shutil
        shutil.copy(readme_path, bak_path)
        
        # Replace or append
        if "<!-- START_TREE -->" in content:
            pattern = r"<!-- START_TREE -->.*?<!-- END_TREE -->"
            content = re.sub(pattern, lambda m: tree_block, content, flags=re.DOTALL)
        else:
            content += "\n\n" + tree_block
            
        if "<!-- START_APPENDIX -->" in content:
            pattern = r"<!-- START_APPENDIX -->.*?<!-- END_APPENDIX -->"
            content = re.sub(pattern, lambda m: appendix_block, content, flags=re.DOTALL)
        else:
            content += "\n\n" + appendix_block
        
        new_content = content

    # Check if changed
    if bak_path.exists() and bak_path.read_text(encoding='utf-8') == new_content:
        print("内容无变化，跳过写入。")
        return False

    readme_path.write_text(new_content, encoding='utf-8')
    return True

def main():
    parser = argparse.ArgumentParser(description="生成目录树并更新 README")
    parser.add_argument("--path", default=".", help="目标目录路径")
    parser.add_argument("--readme", default="README.md", help="README 文件路径")
    parser.add_argument("--max-depth", type=int, default=3, help="最大展开深度")
    parser.add_argument("--exclude", help="自定义正则过滤")
    parser.add_argument("--verbose", action="store_true", help="输出调试信息")
    
    args = parser.parse_args()
    
    try:
        tree, full, d_count, f_count = get_tree(args.path, args.max_depth, args.exclude, args.verbose)
        updated = update_readme(args.readme, tree, full, os.path.abspath(args.path), d_count, f_count)
        
        if updated:
            print(f"目录树已更新至 {args.readme}")
        sys.exit(0)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
