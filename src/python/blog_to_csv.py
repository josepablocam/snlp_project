import Globals
from ReadData import *
blog = to_utf8(prepareBlogData(Globals.BLOG_DATA, splitwords=False))
for item in blog:
		print item[1], "|", item[0];
