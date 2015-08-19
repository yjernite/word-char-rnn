import sys, os
out = open("/tmp/words", 'w')

data_dir = sys.argv[1]
morfessor_dir = sys.argv[2]


for f in ["train.txt", "valid.txt", "test.txt"]:
    for l in open(data_dir + "/" + f):
        words = l.strip().split()
        for w in words:
            print >>out,  w

os.system("perl %s/bin/morfessor1.0.pl -data /tmp/words > /tmp/morph"%(morfessor_dir))
f = open("/tmp/morph")
words = {}
for line in f:
    if line[0] == "#": 
        continue
    word_parts = line.replace("+", "").strip().split()[1:]
    words["".join(word_parts)] =  word_parts

for word, factors in words.iteritems():
    print word, " ".join(factors)
