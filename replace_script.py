import sys

with open('templates/index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if '<!-- Tailwind CSS -->' in line:
        skip = True
        new_lines.append(line)
        new_lines.append('    <link href="{{ url_for(\'static\', filename=\'output.css\') }}" rel="stylesheet">\n')
        continue
    
    if skip and '</style>' in line:
        skip = False
        continue
        
    if not skip:
        new_lines.append(line)

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Updated index.html')
