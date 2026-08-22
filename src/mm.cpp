#include "mm.h"
#include "feature.h"
#include "group.h"
#include "mapper.h"

#include "glog/logging.h"

namespace xivo {

template<typename T>
CircBufWithHash<T>::CircBufWithHash(int max_items, bool is_feature) {
  max_items_ = max_items;
  num_slots_initialized_ = 0;
  num_slots_active_ = 0;
  peak_active_ = 0;
  warned_nearly_full_ = false;
  slot_search_ind_ = 0;
  is_feature_buf_ = is_feature;

  for (int i=0; i < max_items_; i++) {
    T* addr = new T();
    slots_.push_back(addr);
    slots_initialized_.push_back(false);
    slots_active_.push_back(false);
    slots_map_[addr] = i;
  }
}

template<typename T>
CircBufWithHash<T>::~CircBufWithHash() {
  for (auto p: slots_map_) {
    delete p.first;
  }
}

// Exhausting a pool is a LOG(FATAL) that lands whenever the sequence happens to
// need one more slot -- ten minutes into a run, with a stack trace inside the
// tracker, and no hint that the culprit is a config number. Warning at 90% turns
// that into something diagnosable before it happens; see `CheckMemoryPools` for
// the up-front version of the same check.
template<typename T>
void CircBufWithHash<T>::NoteActivation() {
  ++num_slots_active_;
  if (num_slots_active_ > peak_active_) {
    peak_active_ = num_slots_active_;
    if (!warned_nearly_full_ && peak_active_ * 10 >= max_items_ * 9) {
      warned_nearly_full_ = true;
      // ERROR, not WARNING: glog's default stderrthreshold is ERROR, so a
      // WARNING here would only reach the log files.
      LOG(ERROR) << "MemoryManager: " << (is_feature_buf_ ? "feature" : "group")
                 << " pool " << peak_active_ << "/" << max_items_
                 << " active; raise memory.max_"
                 << (is_feature_buf_ ? "features" : "groups")
                 << " -- running out is fatal";
    }
  }
}

template<typename T>
T* CircBufWithHash<T>::GetItem() {
  bool slot_found = false;

  // Try to use an slot that has never been used before first
  if (num_slots_initialized_ < max_items_) {
    while (!slot_found) {
      if (!slots_initialized_[slot_search_ind_]) {

#ifndef NDEBUG
        CHECK(!slots_active_[slot_search_ind_]);
#endif

        slots_initialized_[slot_search_ind_] = true;
        slots_active_[slot_search_ind_] = true;
        T* ret = slots_[slot_search_ind_];

        slot_found = true;
        num_slots_initialized_++;
        NoteActivation();

        slot_search_ind_ = (slot_search_ind_ + 1) % max_items_;

        return ret;
      }
      else {
        slot_search_ind_ = (slot_search_ind_ + 1) % max_items_;
      }
    }
  }

  // If all slots have been used before, just find one that isn't "active".
  else {
    int start_index = slot_search_ind_;

    while (!slot_found) {
      if (!slots_active_[slot_search_ind_]) {
        T* ret = slots_[slot_search_ind_];

#ifdef USE_MAPPER
        LOG(WARNING) << "MemoryManager: Overwriting inactive " << typeid(T).name()
          << " #" << ret->id();

        // Remove the item from the Mapper before doing anything else
        RemoveFromMapper(ret);
#endif
        slots_active_[slot_search_ind_] = true;

        slot_found = true;
        NoteActivation();

        slot_search_ind_ = (slot_search_ind_ + 1) % max_items_;
        return ret;
      }
      else {
        slot_search_ind_ = (slot_search_ind_ + 1) % max_items_;
      }

      if (slot_search_ind_ == start_index) {
        if (is_feature_buf_) {
          LOG(FATAL) << "Out of feature slots in the memory manager";
        } else {
          LOG(FATAL) << "Out of group slots in the memory manager";
        }
      }
    }
  }

  return nullptr;
}


template<typename T>
void CircBufWithHash<T>::DeactivateItem(T *item) {
  int ind = slots_map_[item];
  slots_active_[ind] = false;
  num_slots_active_--;
}


template<typename T>
void CircBufWithHash<T>::DestroyItem(T *item) {
  int ind = slots_map_[item];
  slots_active_[ind] = false;
  slots_initialized_[ind] = false;
  num_slots_active_--;
  num_slots_initialized_--;
}


template<>
void CircBufWithHash<Feature>::RemoveFromMapper(FeaturePtr item) {
#ifdef USE_MAPPER
  Mapper::instance()->RemoveFeature(item);
#endif
}


template<>
void CircBufWithHash<Group>::RemoveFromMapper(GroupPtr item) {
#ifdef USE_MAPPER
  Mapper::instance()->RemoveGroup(item);
#endif
}


std::unique_ptr<MemoryManager> MemoryManager::instance_ = nullptr;

MemoryManagerPtr MemoryManager::Create(int max_features, int max_groups) {
  if (!instance_) {
    instance_ = std::unique_ptr<MemoryManager>(
        new MemoryManager(max_features, max_groups));
    LOG(INFO) << StrFormat(
        "MemoryManager instance created with %d features and %d groups",
        max_features, max_groups);
  } else {
    LOG(WARNING) << "MemoryManager instance already created!";
  }
  return instance_.get();
}

MemoryManager::MemoryManager(int max_features, int max_groups) {
  feature_slots_ = new CircBufWithHash<Feature>(max_features, true);
  group_slots_ = new CircBufWithHash<Group>(max_groups, false);
}

MemoryManager::~MemoryManager() {
  delete feature_slots_;
  delete group_slots_;
}

MemoryManagerPtr MemoryManager::instance() {
  if (!instance_) {
    throw std::runtime_error("MemoryManager instance not created yet");
  }
  return instance_.get();
}

FeaturePtr MemoryManager::GetFeature() {
  FeaturePtr addr = feature_slots_->GetItem();
  if (addr == nullptr) {
    LOG(FATAL) << "Memory Manager: Could not find an inactive feature slot."
      << std::endl;
  }
  return addr;
}

void MemoryManager::DeactivateFeature(FeaturePtr f) {
  feature_slots_->DeactivateItem(f);
}

void MemoryManager::DestroyFeature(FeaturePtr f) {
  feature_slots_->DestroyItem(f);
}

GroupPtr MemoryManager::GetGroup() {
  GroupPtr addr = group_slots_->GetItem();
  if (addr == nullptr) {
    LOG(FATAL) << "Memory manager: Could not find an inactive group slot."
      << std::endl;
  }
  return addr;
}

void MemoryManager::DeactivateGroup(GroupPtr g) {
  group_slots_->DeactivateItem(g);
}

void MemoryManager::DestroyGroup(GroupPtr g) {
  group_slots_->DestroyItem(g);
}

} // namespace xivo
