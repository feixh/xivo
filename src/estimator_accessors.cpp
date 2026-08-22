#include "estimator.h"

#include <opencv2/core/eigen.hpp>


namespace xivo {

VecXi Estimator::InstateFeatureSinds(int n_output) const {

  // Retrieve visibility graph
  Graph& graph{*Graph::instance()};

  // Get vectors of instate features and all features
  std::vector<xivo::FeaturePtr> instate_features = graph.GetInstateFeatures();
  MakePtrVectorUnique(instate_features);
  // std::min, not max: the fill loop below stops at n_output *and* at the end
  // of instate_features, and Eigen does not zero-initialise, so sizing the
  // output by the larger of the two returned uninitialised heap to the caller.
  int npts = std::min((int) instate_features.size(), n_output);

  // Sort features by uncertainty
  std::sort(instate_features.begin(), instate_features.end(),
            [this](FeaturePtr f1, FeaturePtr f2) -> bool {
              return FeatureCovComparison(f1, f2);
            });

  //std::vector<int> FeatureIDs;
  VecXi FeatureSinds(npts);

  int i = 0;
  for (auto it = instate_features.begin();
       it != instate_features.end() && i < n_output;
       ) {
    FeaturePtr f = *it;
    FeatureSinds(i,0) = f->sind();
    ++i;
    ++it;
  }
  return FeatureSinds;
}


VecXi Estimator::InstateFeatureRefGroups(int n_output) const {

  // Retrieve visibility graph
  Graph& graph{*Graph::instance()};

  // Get vectors of instate features and all features
  std::vector<xivo::FeaturePtr> instate_features = graph.GetInstateFeatures();
  MakePtrVectorUnique(instate_features);
  int npts = std::min((int) instate_features.size(), n_output);

  // Sort features by uncertainty
  std::sort(instate_features.begin(), instate_features.end(),
            [this](FeaturePtr f1, FeaturePtr f2) -> bool {
              return FeatureCovComparison(f1, f2);
            });

  //std::vector<int> FeatureIDs;
  VecXi FeatureRefGroups(npts);

  int i = 0;
  for (auto it = instate_features.begin();
       it != instate_features.end() && i < n_output;
       ) {
    FeaturePtr f = *it;
    FeatureRefGroups(i,0) = f->ref()->id();
    ++i;
    ++it;
  }
  return FeatureRefGroups;
}


VecXi Estimator::InstateFeatureIDs(int n_output) const {

  // Retrieve visibility graph
  Graph& graph{*Graph::instance()};

  // Get vectors of instate features and all features
  std::vector<xivo::FeaturePtr> instate_features = graph.GetInstateFeatures();
  MakePtrVectorUnique(instate_features);
  int npts = std::min((int) instate_features.size(), n_output);

  // Sort features by uncertainty
  std::sort(instate_features.begin(), instate_features.end(),
            [this](FeaturePtr f1, FeaturePtr f2) -> bool {
              return FeatureCovComparison(f1, f2);
            });

  //std::vector<int> FeatureIDs;
  VecXi FeatureIDs(npts);

  int i = 0;
  for (auto it = instate_features.begin();
       it != instate_features.end() && i < n_output;
       ) {
    FeaturePtr f = *it;
    FeatureIDs(i,0) = f->id();
    ++i;
    ++it;
  }
  return FeatureIDs;
}


MatX3 Estimator::InstateFeaturePositions(int n_output) const {

  // Retrieve visibility graph
  Graph& graph{*Graph::instance()};

  // Get vectors of instate features and all features
  std::vector<xivo::FeaturePtr> instate_features = graph.GetInstateFeatures();
  MakePtrVectorUnique(instate_features);
  int npts = std::min((int) instate_features.size(), n_output);

  // Sort features by uncertainty
  std::sort(instate_features.begin(), instate_features.end(),
            [this](FeaturePtr f1, FeaturePtr f2) -> bool {
              return FeatureCovComparison(f1, f2);
            });

  MatX3 feature_positions(npts,3);

  int i = 0;
  for (auto it = instate_features.begin();
       it != instate_features.end() && i < n_output;
       ) {
    FeaturePtr f = *it;
    Vec3 Xs = f->Xs();
    feature_positions(i,0) = Xs(0);
    feature_positions(i,1) = Xs(1);
    feature_positions(i,2) = Xs(2);
    ++i;
    ++it;
  }
  return feature_positions;
}


MatX3 Estimator::InstateFeatureXc(int n_output) const {

  // Retrieve visibility graph
  Graph& graph{*Graph::instance()};

  // Get vectors of instate features and all features
  std::vector<xivo::FeaturePtr> instate_features = graph.GetInstateFeatures();
  MakePtrVectorUnique(instate_features);
  int npts = std::min((int) instate_features.size(), n_output);

  // Sort features by uncertainty
  std::sort(instate_features.begin(), instate_features.end(),
            [this](FeaturePtr f1, FeaturePtr f2) {
              return FeatureCovComparison(f1, f2);
            });

  MatX3 feature_positions(npts,3);

  int i = 0;
  for (auto it = instate_features.begin();
       it != instate_features.end() && i < n_output;
       ) {
    FeaturePtr f = *it;
    Vec3 Xc = f->Xc();
    feature_positions(i,0) = Xc(0);
    feature_positions(i,1) = Xc(1);
    feature_positions(i,2) = Xc(2);
    ++i;
    ++it;
  }
  return feature_positions;
}


MatX3 Estimator::InstateFeaturexc(int n_output) const {

  // Retrieve visibility graph
  Graph& graph{*Graph::instance()};

  // Get vectors of instate features and all features
  std::vector<xivo::FeaturePtr> instate_features = graph.GetInstateFeatures();
  MakePtrVectorUnique(instate_features);
  int npts = std::min((int) instate_features.size(), n_output);

  // Sort features by uncertainty
  std::sort(instate_features.begin(), instate_features.end(),
            [this](FeaturePtr f1, FeaturePtr f2) {
              return FeatureCovComparison(f1, f2);
            });

  MatX3 feature_xc(npts,3);

  int i = 0;
  for (auto it = instate_features.begin();
       it != instate_features.end() && i < n_output;
       ) {
    FeaturePtr f = *it;
    Vec3 xc = f->x();
    feature_xc(i,0) = xc(0);
    feature_xc(i,1) = xc(1);
    feature_xc(i,2) = xc(2);
    ++i;
    ++it;
  }
  return feature_xc;
}


MatX2 Estimator::InstateFeaturePreds(int n_output) const {

  // Retrieve visibility graph
  Graph& graph{*Graph::instance()};

  // Get vectors of instate features and all features
  std::vector<xivo::FeaturePtr> instate_features = graph.GetInstateFeatures();
  MakePtrVectorUnique(instate_features);
  int npts = std::min((int) instate_features.size(), n_output);

  // Sort features by uncertainty
  std::sort(instate_features.begin(), instate_features.end(),
            [this](FeaturePtr f1, FeaturePtr f2) {
              return FeatureCovComparison(f1, f2);
            });

  MatX2 feature_xp(npts,2);

  int i = 0;
  for (auto it = instate_features.begin();
       it != instate_features.end() && i < n_output;
       ) {
    FeaturePtr f = *it;
    Vec2 xp = f->pred();
    feature_xp(i,0) = xp(0);
    feature_xp(i,1) = xp(1);
    ++i;
    ++it;
  }
  return feature_xp;
}


MatX2 Estimator::InstateFeatureMeas(int n_output) const {

  // Retrieve visibility graph
  Graph& graph{*Graph::instance()};

  // Get vectors of instate features and all features
  std::vector<xivo::FeaturePtr> instate_features = graph.GetInstateFeatures();
  MakePtrVectorUnique(instate_features);
  int npts = std::min((int) instate_features.size(), n_output);

  // Sort features by uncertainty
  std::sort(instate_features.begin(), instate_features.end(),
            [this](FeaturePtr f1, FeaturePtr f2) {
              return FeatureCovComparison(f1, f2);
            });

  MatX2 feature_xp(npts,2);

  int i = 0;
  for (auto it = instate_features.begin();
       it != instate_features.end() && i < n_output;
       ) {
    FeaturePtr f = *it;
    Vec2 xp = f->xp();
    feature_xp(i,0) = xp(0);
    feature_xp(i,1) = xp(1);
    ++i;
    ++it;
  }
  return feature_xp;
}


MatX6 Estimator::InstateFeatureCovs(int n_output) const {
  // Retrieve visibility graph
  Graph& graph{*Graph::instance()};

  // Get vectors of instate features and all features
  std::vector<xivo::FeaturePtr> instate_features = graph.GetInstateFeatures();
  MakePtrVectorUnique(instate_features);
  int npts = std::min((int) instate_features.size(), n_output);

  // Sort features by uncertainty
  std::sort(instate_features.begin(), instate_features.end(),
            [this](FeaturePtr f1, FeaturePtr f2) {
              return FeatureCovComparison(f1, f2);
            });

  MatX6 feature_covs(npts,6);

  int i = 0;
  for (auto it = instate_features.begin();
       it != instate_features.end() && i < n_output;
       ) {
    FeaturePtr f = *it;
    int foff = kFeatureBegin + 3*f->sind();
    Mat3 cov = P_.block<3,3>(foff, foff);

    feature_covs.block(i, 0, 1, 6) <<
      cov(0,0), cov(0,1), cov(0,2), cov(1,1), cov(1,2), cov(2,2);

    ++i;
    ++it;
  }

  return feature_covs;
}


void Estimator::InstateFeaturePositionsAndCovs(int max_output, int &npts,
  MatX3 &feature_positions, MatX6 &feature_covs, MatX2 &feature_last_px,
  VecXi &feature_ids)
{
  // Retrieve visibility graph
  Graph& graph{*Graph::instance()};

  // Get vectors of instate features and all features
  std::vector<xivo::FeaturePtr> instate_features = graph.GetInstateFeatures();
  MakePtrVectorUnique(instate_features);
  npts = std::min((int) instate_features.size(), max_output);

  // Sort features by uncertainty so we grab the "best" features
  std::sort(instate_features.begin(), instate_features.end(),
            [this](FeaturePtr f1, FeaturePtr f2) {
              return FeatureCovComparison(f1, f2);
            });

  feature_positions.resize(npts,3);
  feature_covs.resize(npts,6);
  feature_last_px.resize(npts,2);
  feature_ids.resize(npts);

  int i = 0;
  for (auto it = instate_features.begin();
       it != instate_features.end() && i < npts;
       ) {
    FeaturePtr f = *it;

    Vec3 Xs = f->Xs();
    feature_positions(i,0) = Xs(0);
    feature_positions(i,1) = Xs(1);
    feature_positions(i,2) = Xs(2);

    int foff = kFeatureBegin + 3*f->sind();
    Mat3 cov = P_.block<3,3>(foff, foff);

    feature_covs.block(i, 0, 1, 6) <<
      cov(0,0), cov(0,1), cov(0,2), cov(1,1), cov(1,2), cov(2,2);

    feature_ids(i) = f->id();

    Vec2 xp = f->xp();
    feature_last_px(i,0) = xp(0);
    feature_last_px(i,1) = xp(1);

    ++i;
    ++it;
  }
}


VecXi Estimator::InstateFeatureIDs() const
{
  int num_instate_features = instate_features_.size();

  // Get all features
  VecXi FeatureIDs(num_instate_features);

  int i = 0;
  for (auto it = instate_features_.begin();
       it != instate_features_.end() && i < num_instate_features;
       ) {
    FeaturePtr f = *it;
    FeatureIDs(i) = f->id();
    ++i;
    ++it;
  }
  return FeatureIDs;
}


VecXi Estimator::InstateFeatureSinds() const
{
  int num_instate_features = instate_features_.size();

  // Get all features
  VecXi FeatureSinds(num_instate_features);

  int i = 0;
  for (auto it = instate_features_.begin();
       it != instate_features_.end() && i < num_instate_features;
       ) {
    FeaturePtr f = *it;
    FeatureSinds(i) = f->sind();
    ++i;
    ++it;
  }
  return FeatureSinds;
}


VecXi Estimator::InstateFeatureRefGroups() const
{
  int num_instate_features = instate_features_.size();

  // Get all features
  VecXi FeatureRefGroups(num_instate_features);

  int i = 0;
  for (auto it = instate_features_.begin();
       it != instate_features_.end() && i < num_instate_features;
       ) {
    FeaturePtr f = *it;
    FeatureRefGroups(i) = f->ref()->id();
    ++i;
    ++it;
  }
  return FeatureRefGroups;
}


MatX3 Estimator::InstateFeaturePositions() const
{
  int num_features = instate_features_.size();

  MatX3 feature_positions(num_features,3);

  int i = 0;
  for (auto it = instate_features_.begin();
       it != instate_features_.end() && i < num_features;
       ) {
    FeaturePtr f = *it;
    Vec3 Xs = f->Xs();
    feature_positions(i,0) = Xs(0);
    feature_positions(i,1) = Xs(1);
    feature_positions(i,2) = Xs(2);
    ++i;
    ++it;
  }
  return feature_positions;
}


MatX3 Estimator::InstateFeatureXc() const
{
  int num_features = instate_features_.size();

  MatX3 feature_positions(num_features,3);

  int i = 0;
  for (auto it = instate_features_.begin();
       it != instate_features_.end() && i < num_features;
       ) {
    FeaturePtr f = *it;
    Vec3 Xc = f->Xc();
    feature_positions(i,0) = Xc(0);
    feature_positions(i,1) = Xc(1);
    feature_positions(i,2) = Xc(2);
    ++i;
    ++it;
  }
  return feature_positions;
}


MatX6 Estimator::InstateFeatureCovs() const {

  int num_features = instate_features_.size();

  MatX6 feature_covs(num_features,6);

  int i = 0;
  for (auto it = instate_features_.begin();
       it != instate_features_.end() && i < num_features;
       ) {
    FeaturePtr f = *it;
    int foff = kFeatureBegin + 3*f->sind();
    Mat3 cov = P_.block<3,3>(foff, foff);

    feature_covs.block(i, 0, 1, 6) <<
      cov(0,0), cov(0,1), cov(0,2), cov(1,1), cov(1,2), cov(2,2);

    ++i;
    ++it;
  }

  return feature_covs;
}


MatX3 Estimator::InstateFeaturexc() const
{
  int num_features = instate_features_.size();

  MatX3 feature_xc(num_features,3);

  int i = 0;
  for (auto it = instate_features_.begin();
       it != instate_features_.end() && i < num_features;
       ) {
    FeaturePtr f = *it;
    Vec3 xc = f->x();
    feature_xc(i,0) = xc(0);
    feature_xc(i,1) = xc(1);
    feature_xc(i,2) = xc(2);
    ++i;
    ++it;
  }
  return feature_xc;
}


MatX2 Estimator::InstateFeaturePreds() const
{
  int num_features = instate_features_.size();

  MatX2 feature_xp(num_features,2);

  int i = 0;
  for (auto it = instate_features_.begin();
       it != instate_features_.end() && i < num_features;
       ) {
    FeaturePtr f = *it;
    Vec2 xp = f->pred();
    feature_xp(i,0) = xp(0);
    feature_xp(i,1) = xp(1);
    ++i;
    ++it;
  }
  return feature_xp;
}


MatX2 Estimator::InstateFeatureMeas() const
{
  int num_features = instate_features_.size();

  MatX2 feature_xp(num_features,2);

  int i = 0;
  for (auto it = instate_features_.begin();
       it != instate_features_.end() && i < num_features;
       ) {
    FeaturePtr f = *it;
    Vec2 xp = f->xp();
    feature_xp(i,0) = xp(0);
    feature_xp(i,1) = xp(1);
    ++i;
    ++it;
  }
  return feature_xp;
}


VecXi Estimator::InstateGroupIDs() const
{
  int num_groups = instate_groups_.size();

  VecXi GroupIDs(num_groups);

  int i = 0;
  for (auto it = instate_groups_.begin();
       it != instate_groups_.end() && i<num_groups;) {
    GroupPtr g = *it;
    GroupIDs(i) = g->id();
    ++i;
    ++it;
  }
  return GroupIDs;
}


VecXi Estimator::InstateGroupSinds() const
{
  int num_groups = instate_groups_.size();

  VecXi GroupSinds(num_groups);

  int i = 0;
  for (auto it = instate_groups_.begin();
       it != instate_groups_.end() && i<num_groups;) {
    GroupPtr g = *it;
    GroupSinds(i) = g->sind();
    ++i;
    ++it;
  }
  return GroupSinds;
}


MatX7 Estimator::InstateGroupPoses() const
{
  int num_groups = instate_groups_.size();

  MatX7 group_poses(num_groups, 7);

  int i = 0;
  for (auto it = instate_groups_.begin();
       it != instate_groups_.end() && i < num_groups;
       ) {
    GroupPtr g = *it;
    Vec3 Tsb = g->Tsb();
    Mat3 Rsb = g->Rsb().matrix();
    Quat Qsb(Rsb);

    group_poses(i,0) = Qsb.x();
    group_poses(i,1) = Qsb.y();
    group_poses(i,2) = Qsb.z();
    group_poses(i,3) = Qsb.w();
    group_poses(i,4) = Tsb(0);
    group_poses(i,5) = Tsb(1);
    group_poses(i,6) = Tsb(2);

    ++i;
    ++it;
  }

  return group_poses;
}


MatX Estimator::InstateGroupCovs() const
{
  int num_groups = instate_groups_.size();

  MatX group_covs(num_groups, 21);

  int i = 0;
  for (auto it = instate_groups_.begin();
       it != instate_groups_.end() && i < num_groups;
       ) {
    GroupPtr g = *it;
    Mat6 cov = InstateGroupCov(g);

    int cnt;
    for (int ii = 0; ii<6; ii++) {
      cnt = 0;
      for (int jj = ii; jj<6; jj++) {
        group_covs(i,cnt) = cov(ii,jj);
        cnt++;
      }
    }

    ++i;
    ++it;
  }

  return group_covs;

}



Mat3 Estimator::InstateFeatureCov(FeaturePtr f) const {
#ifndef NDEBUG
  CHECK(f->instate());
#endif
  int foff = kFeatureBegin + 3*f->sind();
  return P_.block<3,3>(foff, foff);
}

Mat6 Estimator::InstateGroupCov(GroupPtr g) const {
#ifndef NDEBUG
  CHECK(g->instate());
#endif
  int goff = kGroupBegin + 6*g->sind();
  return P_.block<6,6>(goff, goff);
}


std::vector<std::tuple<int, Vec2, MatXf>> Estimator::tracked_features() {

  auto tracker = Tracker::instance();

  // store tracked feature information
  std::vector<std::tuple<int, Vec2, MatXf>> tracked_features_info;

  for (auto f : tracker->features_)
  {
    int id = f->id();

    // Convert from cv::Mat to matrix. Features have no descriptor at all when
    // extract_descriptor is off, in which case this stays empty rather than
    // reading descriptors_.back() on an empty vector.
    MatXf descriptor_eigen;
    if (f->has_descriptor()) {
      cv2eigen(f->descriptor(), descriptor_eigen);
    }

    Vec2 kp_vector = f->xp();

    tracked_features_info.push_back(std::tuple(id, kp_vector, descriptor_eigen));
  }

  return tracked_features_info;
}


std::vector<std::tuple<int, Vec2>> Estimator::tracked_features_no_descriptor() {

  auto tracker = Tracker::instance();

  // store tracked feature information
  std::vector<std::tuple<int, Vec2>> tracked_features_info;

  for (auto f : tracker->features_)
  {
    int id = f->id();

    Vec2 kp_vector = f->xp();

    tracked_features_info.push_back(std::tuple(id, kp_vector));
  }

  return tracked_features_info;
}


VecXi Estimator::JustDroppedFeatureIDs() const {
  int num_just_dropped = just_dropped_feature_ids_.size();
  VecXi output(num_just_dropped);

  for (int i=0; i < num_just_dropped; i++) {
    output[i] = just_dropped_feature_ids_[i];
  }

  return output;
}


}
