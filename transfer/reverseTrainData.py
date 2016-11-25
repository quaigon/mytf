import pickle

# rs = pickle.load(open("translated.pickle",))
#
# reversed = []
#
# for item in rs:
#     reversed.append((item[1], item[2], item[0]))
#
# print len(reversed)
#
# pickle.dump(reversed, open("finalTrain.pickle", "wb"))
#
rs = pickle.load(open("train.pickle"))

for one in rs:
    for board in [one[0], one[1]]:
        for row in xrange(8):
            print ' '.join('%s' % x for x in board[(row*8):((row+1)*8)])
        print

print len(rs)