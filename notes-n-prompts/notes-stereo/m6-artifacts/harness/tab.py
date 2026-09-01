import sys, collections, statistics
rows = collections.defaultdict(dict)
for l in sys.stdin:
    if not l.startswith('RESULT'): continue
    p = l.split()
    tag, seq = p[1], p[2]
    rows[tag][seq] = (float(p[3]), float(p[4]), float(p[5]), float(p[6]))
seqs = ['room%d' % i for i in range(1,7)]
print('%-8s %s   %-8s %-8s %-8s %-8s' % ('arm', ' '.join('%-8s'%s for s in seqs),
                                          'mATE001','mATE02','mRPEtra','mRPErot'))
out = []
for tag, d in rows.items():
    if not all(s in d for s in seqs): continue
    m = [statistics.mean(d[s][i] for s in seqs) for i in range(4)]
    out.append((m[0], tag, d, m))
for _, tag, d, m in sorted(out):
    print('%-8s %s   %-8.4f %-8.4f %-8.4f %-8.4f' % (
        tag, ' '.join('%-8.4f' % d[s][0] for s in seqs), m[0], m[1], m[2], m[3]))
