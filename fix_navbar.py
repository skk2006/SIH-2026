import glob
import re

html_files = glob.glob('e:/sihh/SIH-2026/AI/source/templates/*.html')

for fpath in html_files:
    if fpath.endswith('retrieve.html'):
        continue
        
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'href="/retrieve"' in content:
        continue
    
    # We want to replace the line containing '<a href="vehicles">' and insert Retrieve
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if 'href="vehicles"' in line or 'href="/vehicles"' in line:
            if '<i class=' in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(indent + '<li><a href="/retrieve"><i class="fas fa-search"></i> Retrieve</a></li>')
            else:
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(indent + '<li><a href="/retrieve">Retrieve</a></li>')
        new_lines.append(line)
        
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))

print("Applied navbar links everywhere")
