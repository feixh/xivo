// Options objects for various depth-related algorithms,
// and policies for feature selection, etc.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "options.h"
#include "feature.h"
#include "param.h"

namespace xivo {

namespace {

/** The three `Criteria` entry points below are the hottest ParameterServer
 *  clients in the estimator, and each call used to walk the JSON tree afresh:
 *  `Candidate`/`CandidateStrict` are the predicates `Graph::GetFeaturesIf` runs
 *  over *every* feature every frame (three `Json::Value::get` lookups each), and
 *  `CandidateComparison` is a `std::sort` comparator, so it did a lookup *and*
 *  materialized a `std::string` O(n log n) times per frame. Together that was the
 *  1.5% of stereo CPU the profile attributed to `__introsort_loop<Feature**>` and
 *  its callees, for four numbers that cannot change during a run.
 *
 *  Cached per ParameterServer instance rather than once per process:
 *  `ParameterServer::Create` refuses to replace a live instance, so within one
 *  run the values are immutable, but a test binary constructs its own instance
 *  and must see its own config. Comparing the instance pointer costs one load.
 *
 *  Bit-identical by construction: the same JSON node yields the same `double`
 *  every time, so caching changes nothing but the number of times it is read.
 *  The only observable difference is that an invalid `comparison_score_type` now
 *  logs once instead of once per comparison. */
enum class ScoreType {
  DepthUncertainty,
  CovarianceDiagNorm,
  CovarianceDiagNormPlusOutlierCount
};

struct CriteriaParams {
  // Every member has a default initializer and the type is trivially
  // destructible, so the singleton below is constant-initialized: no
  // thread-safe-static guard, just the pointer compare.
  const ParameterServer *owner = nullptr;
  number_t zmin = 0.05;
  number_t zmax = 5.0;
  number_t max_subfilter_outlier = 0.01;
  ScoreType score = ScoreType::DepthUncertainty;
};

const CriteriaParams &Params() {
  static CriteriaParams c;
  const ParameterServer *P = ParameterServer::instance();
  if (c.owner == P) {
    return c;
  }
  CriteriaParams n;
  n.owner = P;
  if (P) {
    n.zmin = P->get("min_depth", 0.05).asDouble();
    n.zmax = P->get("max_depth", 5.0).asDouble();
    n.max_subfilter_outlier =
        P->get("max_subfilter_outlier", 0.01).asDouble();
    const std::string score_type =
        P->get("comparison_score_type", "DepthUncertainty").asString();
    if (score_type == "DepthUncertainty") {
      n.score = ScoreType::DepthUncertainty;
    } else if (score_type == "CovarianceDiagNorm") {
      n.score = ScoreType::CovarianceDiagNorm;
    } else if (score_type == "CovarianceDiagNormPlusOutlierCount") {
      n.score = ScoreType::CovarianceDiagNormPlusOutlierCount;
    } else {
      LOG(ERROR) << "Invalid feature score type " << score_type
                 << "; falling back to DepthUncertainty";
      n.score = ScoreType::DepthUncertainty;
    }
  }
  c = n;
  return c;
}

} // namespace

bool Criteria::Candidate(FeaturePtr f) {
  const CriteriaParams &P = Params();

  bool good = (f->status() == FeatureStatus::READY ||
          f->status() == FeatureStatus::INITIALIZING) &&
         (f->outlier_counter() < P.max_subfilter_outlier);
  good = good && (f->z() > P.zmin && f->z() < P.zmax);
  return good;
}

bool Criteria::CandidateStrict(FeaturePtr f) {
  const CriteriaParams &P = Params();

  bool good = f->status() == FeatureStatus::READY &&
         (f->outlier_counter() < P.max_subfilter_outlier);
  good = good && (f->z() > P.zmin && f->z() < P.zmax);
  return good;
}

bool Criteria::CandidateComparison(FeaturePtr f1, FeaturePtr f2) {
  const ScoreType score_type = Params().score;

  int s1 = as_integer(f1->status());
  int s2 = as_integer(f2->status());

  number_t score1, score2;
  if (score_type == ScoreType::DepthUncertainty) {
    // Same quantity `Feature::score()` returns, spelled out here so all three
    // options go through one code path.
    score1 = -1.0 * (f1->P())(2,2);
    score2 = -1.0 * (f2->P())(2,2);
  }
  else if (score_type == ScoreType::CovarianceDiagNorm) {
    score1 = -1.0 * f1->P().diagonal().norm();
    score2 = -1.0 * f2->P().diagonal().norm();
  }
  else {
    // CovarianceDiagNormPlusOutlierCount -- the one implemented in Corvis.
    // An unrecognized name was already resolved to DepthUncertainty in
    // `Params()`, so there is no fallback branch left here.
    score1 = -1.0 * (f1->P().diagonal().norm() + f1->outlier_counter());
    score2 = -1.0 * (f2->P().diagonal().norm() + f2->outlier_counter());
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
