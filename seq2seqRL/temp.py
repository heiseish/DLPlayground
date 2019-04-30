lst = []
with open('requirements.txt', 'r') as f:
	lines = f.read().split('\n')
	for line in lines:
		s = ''
		for ch in line:
			if ch == '=':
				break
			s += ch
		lst.append(s)

with open('requirements.txt', 'w') as f:
	for dep in lst:
		f.write("%s\n" % dep)