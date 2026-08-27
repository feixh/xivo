#include <limits>

#include "group.h"
#include "feature.h"
#include "mm.h"

namespace xivo {

int Group::counter_ = 0;

// For GroupAdj struct
void GroupAdj::Add(int id) { insert(id); }
void GroupAdj::Remove(int id) { erase(id); }


////////////////////////////////////////
// FACTORY METHODS
////////////////////////////////////////
GroupPtr Group::Create(const SO3 &Rsb, const Vec3 &Tsb) {
  auto g = MemoryManager::instance()->GetGroup();
#ifndef NDEBUG
  CHECK(g);
#endif
  g->Reset(Rsb, Tsb);
  return g;
}

void Group::Deactivate(GroupPtr g) { 
  MemoryManager::instance()->DeactivateGroup(g);
}

void Group::Destroy(GroupPtr g) {
  MemoryManager::instance()->DestroyGroup(g);
}

void Group::Reset(const SO3 &Rsb, const Vec3 &Tsb) {
  id_ = counter_++;
  // Group IDs used to be capped at Feature::counter0 (10000) so that group and
  // feature IDs could share one space. One group is created per image, so that
  // aborted any run longer than 10000 frames -- 8.3 minutes at 20 Hz, which
  // killed 12 of TUM-VI's 28 sequences. The sharing is now confined to
  // Optimizer::VertexId, which interleaves the two, so the only bound left is
  // that encoding's: IDs must stay under INT_MAX/2.
  CHECK_LT(id_, std::numeric_limits<int>::max() / 2) << "Group ID overflow";
  lifetime_ = 0;
  sind_ = -1;
  status_ = GroupStatus::CREATED;
  X_.Rsb = Rsb;
  X_.Tsb = Tsb;
  VLOG(0) << "group #" << id_ << " created";
}

bool Group::instate() const {
  return status_ == GroupStatus::INSTATE || status_ == GroupStatus::GAUGE;
}

} // namespace xivo
