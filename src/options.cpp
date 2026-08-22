// Options objects for various depth-related algorithms,
// and policies for feature selection, etc.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "options.h"
#include "feature.h"
#include "param.h"

namespace xivo {

bool Criteria::Candidate(FeaturePtr f) {
  ParameterServer& P{*ParameterServer::instance()};
  number_t zmin = P.get("min_depth", 0.05).asDouble();
  number_t zmax = P.get("max_depth", 5.0).asDouble();
  number_t max_subfilter_outlier = P.get("max_subfilter_outlier", 0.01).asDouble();

  bool good = (f->status() == FeatureStatus::READY ||
          f->status() == FeatureStatus::INITIALIZING) &&
         (f->outlier_counter() < max_subfilter_outlier);
  good = good && (f->z() > zmin && f->z() < zmax);
  return good;
}

bool Criteria::CandidateStrict(FeaturePtr f) {
  ParameterServer& P{*ParameterServer::instance()};
  number_t zmin = P.get("min_depth", 0.05).asDouble();
  number_t zmax = P.get("max_depth", 5.0).asDouble();
  number_t max_subfilter_outlier = P.get("max_subfilter_outlier", 0.01).asDouble();

  bool good = f->status() == FeatureStatus::READY &&
         (f->outlier_counter() < max_subfilter_outlier);
  good = good && (f->z() > zmin && f->z() < zmax);
  return good;
}

bool Criteria::CandidateComparison(FeaturePtr f1, FeaturePtr f2) {
  ParameterServer& P{*ParameterServer::instance()};
  std::string score_type = P.get("comparison_score_type", "DepthUncertainty").asString();

  int s1 = as_integer(f1->status());
  int s2 = as_integer(f2->status());

  number_t score1, score2;
  if (score_type == "DepthUncertainty") {
    // Same quantity `Feature::score()` returns, spelled out here so all three
    // options go through one code path.
    score1 = -1.0 * (f1->P())(2,2);
    score2 = -1.0 * (f2->P())(2,2);
  }
  else if (score_type == "CovarianceDiagNorm") {
    score1 = -1.0 * f1->P().diagonal().norm();
    score2 = -1.0 * f2->P().diagonal().norm();
  }
  else if (score_type == "CovarianceDiagNormPlusOutlierCount") {
    // This is the one that is implemented in Corvis
    score1 = -1.0 * (f1->P().diagonal().norm() + f1->outlier_counter());
    score2 = -1.0 * (f2->P().diagonal().norm() + f2->outlier_counter());
  }
  else {
    LOG(ERROR) << "Invalid feature score type " << score_type
               << "; falling back to DepthUncertainty";
    score1 = -1.0 * (f1->P())(2,2);
    score2 = -1.0 * (f2->P())(2,2);
  }

  // score1/score2 used to be computed and then thrown away in favour of
  // f->score(), which is hard-wired to -P_(2,2) -- so `comparison_score_type`
  // was a silently ignored knob and "CovarianceDiagNorm" /
  // "CovarianceDiagNormPlusOutlierCount" were unreachable. The default
  // ("DepthUncertainty") is exactly what score() returns, so honoring it here
  // leaves default behaviour unchanged.
  if (s1 != s2) {
    return s1 > s2;
  }
  if (score1 != score2) {
    return score1 > score2;
  }
  // Tie-break on id so this is a *total* order, not merely a strict weak one.
  //
  // Without this the result of std::sort over tied features depends on their
  // order on input, and callers reach here via MakePtrVectorUnique, which sorts
  // by *pointer value*. Ties are the common case, not a rare one: every freshly
  // initialized candidate carries the same initial depth variance, so all of
  // them compare equal. The candidate list is then truncated at kMaxFeature,
  // which means which of several equally-uncertain features got promoted into
  // the state was decided by heap addresses -- and therefore by ASLR, varying
  // from run to run. Measured on TUM-VI room3: ~1 run in 8 produced a different
  // trajectory (ATE 0.1549 vs 0.1703). See notes-stereo/m3a-determinism.md.
  //
  // Ascending id prefers the older feature, which has the longer track.
  return f1->id() < f2->id();
}

} // namespace xivo
