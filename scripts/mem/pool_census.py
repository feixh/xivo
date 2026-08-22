# gdb python: census of what the pooled `Feature` objects are still holding.
#
# Run through scripts/mem/pool_census.sh, which stops the process in
# MemoryManager::~MemoryManager -- i.e. after the whole sequence has been
# played, with every pool slot in the state the run left it.
#
# This is the measurement LeakSanitizer cannot make. The pool is a fixed set of
# `max_features` objects that are recycled, never destroyed, so anything a slot
# accumulates and never drops is invisible to a leak checker (it is reachable
# from a singleton, and freed at exit) while still growing without bound.
#
# What matters per slot is not one number but two:
#   * how many cv::Mat headers `descriptors_` holds, and
#   * how many *distinct* buffers those headers reference (cv::Mat::u), because
#     a descriptor stored as `all_descriptors.row(i)` shares -- and therefore
#     keeps alive -- the whole per-frame descriptor matrix it was cut from.
import gdb


def vec_range(v):
    """(start, count) of a std::vector."""
    impl = v["_M_impl"]
    start = impl["_M_start"]
    return start, int(impl["_M_finish"] - start)


def vec_capacity(v):
    impl = v["_M_impl"]
    return int(impl["_M_end_of_storage"] - impl["_M_start"])


mm = gdb.parse_and_eval("xivo::MemoryManager::instance()")
pool = mm["feature_slots_"].dereference()
slots, num_slots = vec_range(pool["slots_"])

total_desc = 0
per_slot = []
pinned = {}                 # UMatData* -> its size in bytes
own_bytes = 0               # bytes of buffers referenced by exactly one header
vec2_capacity = 0
live_points = 0

for i in range(num_slots):
    f = (slots + i).dereference()

    descs, n = vec_range(f["descriptors_"])
    per_slot.append(n)
    total_desc += n
    for j in range(n):
        u = (descs + j).dereference()["u"]
        if int(u) != 0:
            pinned[int(u)] = int(u.dereference()["size"])

    # the Track base: std::vector<Vec2>, 16 bytes per point
    base = f.cast(gdb.lookup_type("xivo::Track").pointer()).dereference()
    vec2_capacity += vec_capacity(base) * 16
    _, live = vec_range(base)
    live_points += live

header_bytes = total_desc * int(gdb.lookup_type("cv::Mat").sizeof)
pinned_bytes = sum(pinned.values())

print("POOL CENSUS  (feature slots = %d)" % num_slots)
print("  retained descriptors ......... %d" % total_desc)
print("  max per slot ................. %d" % (max(per_slot) if per_slot else 0))
print("  slots holding > 1 ............ %d" % sum(1 for n in per_slot if n > 1))
print("  distinct pinned buffers ...... %d" % len(pinned))
print("  pinned bytes ................. %d" % pinned_bytes)
print("  cv::Mat header bytes ......... %d" % header_bytes)
print("  Track point capacity bytes ... %d  (%d live points)"
      % (vec2_capacity, live_points))
print("  total ........................ %.2f MB"
      % ((pinned_bytes + header_bytes + vec2_capacity) / 1e6))
print("  features created ............. %s"
      % gdb.parse_and_eval("xivo::Feature::counter_"))
