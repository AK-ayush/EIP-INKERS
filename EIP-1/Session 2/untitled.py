import os
import sh

to_str = '#!/usr/bin/python3.6'

for f in os.listdir('./'):
	if os.path.isfile(os.path.join('./', f)):
		sh.sed("-i", "1s/.*/" + to_str + "/", os.path.join('./', f))
