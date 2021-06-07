// Copyright (c) Facebook, Inc. and its affiliates.
#include "cocoeval.h"
#include <time.h>
#include <algorithm>
#include <cstdint>
#include <numeric>

using namespace pybind11::literals;

namespace detectron2 {

namespace COCOeval {

// Sort detections from highest score to lowest, such that
// detection_instances[detection_sorted_indices[t]] >=
// detection_instances[detection_sorted_indices[t+1]].  Use stable_sort to match
// original COCO API
void SortInstancesByDetectionScore(
    const std::vector<InstanceAnnotation>& detection_instances,
    std::vector<uint64_t>* detection_sorted_indices) {
  detection_sorted_indices->resize(detection_instances.size());
  std::iota(
      detection_sorted_indices->begin(), detection_sorted_indices->end(), 0);
  std::stable_sort(
      detection_sorted_indices->begin(),
      detection_sorted_indices->end(),
      [&detection_instances](size_t j1, size_t j2) {
        return detection_instances[j1].score > detection_instances[j2].score;
      });
}

// Partition the ground truth objects based on whether or not to ignore them
// based on area
void SortInstancesByIgnore(
    const std::array<double, 2>& area_range,
    const std::vector<InstanceAnnotation>& ground_truth_instances,
    std::vector<uint64_t>* ground_truth_sorted_indices,
    std::vector<bool>* ignores) {
  ignores->clear();
  ignores->reserve(ground_truth_instances.size());
  for (auto o : ground_truth_instances) {
    ignores->push_back(
        o.ignore || o.area < area_range[0] || o.area > area_range[1]);
  }

  ground_truth_sorted_indices->resize(ground_truth_instances.size());
  std::iota(
      ground_truth_sorted_indices->begin(),
      ground_truth_sorted_indices->end(),
      0);
  std::stable_sort(
      ground_truth_sorted_indices->begin(),
      ground_truth_sorted_indices->end(),
      [&ignores](size_t j1, size_t j2) {
        return (int)(*ignores)[j1] < (int)(*ignores)[j2];
      });
}

// For each IOU threshold, greedily match each detected instance to a ground
// truth instance (if possible) and store the results
void MatchDetectionsToGroundTruth(
    const std::vector<InstanceAnnotation>& detection_instances,
    const std::vector<uint64_t>& detection_sorted_indices,
    const std::vector<InstanceAnnotation>& ground_truth_instances,
    const std::vector<uint64_t>& ground_truth_sorted_indices,
    const std::vector<bool>& ignores,
    const std::vector<std::vector<double>>& ious,
    const std::vector<double>& iou_thresholds,
    const std::array<double, 2>& area_range,
    ImageEvaluation* results) {
  // Initialize memory to store return data matches and ignore
  const int num_iou_thresholds = iou_thresholds.size();
  const int num_ground_truth = ground_truth_sorted_indices.size();
  const int num_detections = detection_sorted_indices.size();
  std::vector<uint64_t> ground_truth_matches(
      num_iou_thresholds * num_ground_truth, 0);
  std::vector<uint64_t>& detection_matches = results->detection_matches;
  std::vector<bool>& detection_ignores = results->detection_ignores;
  std::vector<bool>& ground_truth_ignores = results->ground_truth_ignores;
  detection_matches.resize(num_iou_thresholds * num_detections, 0);
  detection_ignores.resize(num_iou_thresholds * num_detections, false);
  ground_truth_ignores.resize(num_ground_truth);
  for (auto g = 0; g < num_ground_truth; ++g) {
    ground_truth_ignores[g] = ignores[ground_truth_sorted_indices[g]];
  }

  for (auto t = 0; t < num_iou_thresholds; ++t) {
    for (auto d = 0; d < num_detections; ++d) {
      // information about best match so far (match=-1 -> unmatched)
      double best_iou = std::min(iou_thresholds[t], 1 - 1e-10);
      int match = -1;
      for (auto g = 0; g < num_ground_truth; ++g) {
        // if this ground truth instance is already matched and not a
        // crowd, it cannot be matched to another detection
        if (ground_truth_matches[t * num_ground_truth + g] > 0 &&
            !ground_truth_instances[ground_truth_sorted_indices[g]].is_crowd) {
          continue;
        }

        // if detected instance matched to a regular ground truth
        // instance, we can break on the first ground truth instance
        // tagged as ignore (because they are sorted by the ignore tag)
        if (match >= 0 && !ground_truth_ignores[match] &&
            ground_truth_ignores[g]) {
          break;
        }

        // if IOU overlap is the best so far, store the match appropriately
        if (ious[d][ground_truth_sorted_indices[g]] >= best_iou) {
          best_iou = ious[d][ground_truth_sorted_indices[g]];
          match = g;
        }
      }
      // if match was made, store id of match for both detection and
      // ground truth
      if (match >= 0) {
        detection_ignores[t * num_detections + d] = ground_truth_ignores[match];
        detection_matches[t * num_detections + d] =
            ground_truth_instances[ground_truth_sorted_indices[match]].id;
        ground_truth_matches[t * num_ground_truth + match] =
            detection_instances[detection_sorted_indices[d]].id;
      }

      // set unmatched detections outside of area range to ignore
      const InstanceAnnotation& detection =
          detection_instances[detection_sorted_indices[d]];
      detection_ignores[t * num_detections + d] =
          detection_ignores[t * num_detections + d] ||
          (detection_matches[t * num_detections + d] == 0 &&
           (detection.area < area_range[0] || detection.area > area_range[1]));
    }
  }

  // store detection score results
  results->detection_scores.resize(detection_sorted_indices.size());
  for (size_t d = 0; d < detection_sorted_indices.size(); ++d) {
    results->detection_scores[d] =
        detection_instances[detection_sorted_indices[d]].score;
  }
}

std::vector<ImageEvaluation> EvaluateImages(
    const std::vector<std::array<double, 2>>& area_ranges,
    int max_detections,
    const std::vector<double>& iou_thresholds,
    const ImageCategoryInstances<std::vector<double>>& image_category_ious,
    const ImageCategoryInstances<InstanceAnnotation>&
        image_category_ground_truth_instances,
    const ImageCategoryInstances<InstanceAnnotation>&
        image_category_detection_instances) {
  const int num_area_ranges = area_ranges.size();
  const int num_images = image_category_ground_truth_instances.size();
  const int num_categories =
      image_category_ious.size() > 0 ? image_category_ious[0].size() : 0;
  std::vector<uint64_t> detection_sorted_indices;
  std::vector<uint64_t> ground_truth_sorted_indices;
  std::vector<bool> ignores;
  std::vector<ImageEvaluation> results_all(
      num_images * num_area_ranges * num_categories);

  // Store results for each image, category, and area range combination. Results
  // for each IOU threshold are packed into the same ImageEvaluation object
  for (auto i = 0; i < num_images; ++i) {
    for (auto c = 0; c < num_categories; ++c) {
      const std::vector<InstanceAnnotation>& ground_truth_instances =
          image_category_ground_truth_instances[i][c];
      const std::vector<InstanceAnnotation>& detection_instances =
          image_category_detection_instances[i][c];

      SortInstancesByDetectionScore(
          detection_instances, &detection_sorted_indices);
      if ((int)detection_sorted_indices.size() > max_detections) {
        detection_sorted_indices.resize(max_detections);
      }

      for (size_t a = 0; a < area_ranges.size(); ++a) {
        SortInstancesByIgnore(
            area_ranges[a],
            ground_truth_instances,
            &ground_truth_sorted_indices,
            &ignores);

        MatchDetectionsToGroundTruth(
            detection_instances,
            detection_sorted_indices,
            ground_truth_instances,
            ground_truth_sorted_indices,
            ignores,
            image_category_ious[i][c],
            iou_thresholds,
            area_ranges[a],
            &results_all
                [c * num_area_ranges * num_images + a * num_images + i]);
      }
    }
  }

  return results_all;
}

// Convert a python list to a vector
template <typename T>
std::vector<T> list_to_vec(const py::list& l) {
  std::vector<T> v(py::len(l));
  for (int i = 0; i < (int)py::len(l); ++i) {
    v[i] = l[i].cast<T>();
  }
  return v;
}

// Helper function to Accumulate()
// Considers the evaluation results applicable to a particular category, area
// range, and max_detections parameter setting, which begin at
// evaluations[evaluation_index].  Extracts a sorted list of length n of all
// applicable detection instances concatenated across all images in the dataset,
// which are represented by the outputs evaluation_indices, detection_scores,
// image_detection_indices, and detection_sorted_indices--all of which are
// length n. evaluation_indices[i] stores the applicable index into
// evaluations[] for instance i, which has detection score detection_score[i],
// and is the image_detection_indices[i]'th of the list of detections
// for the image containing i.  detection_sorted_indices[] defines a sorted
// permutation of the 3 other outputs
int BuildSortedDetectionList(
    const std::vector<ImageEvaluation>& evaluations,
    const int64_t evaluation_index,
    const int64_t num_images,
    const int max_detections,
    std::vector<uint64_t>* evaluation_indices,
    std::vector<double>* detection_scores,
    std::vector<uint64_t>* detection_sorted_indices,
    std::vector<uint64_t>* image_detection_indices) {
  assert(evaluations.size() >= evaluation_index + num_images);

  // Extract a list of object instances of the applicable category, area
  // range, and max detections requirements such that they can be sorted
  image_detection_indices->clear();
  evaluation_indices->clear();
  detection_scores->clear();
  image_detection_indices->reserve(num_images * max_detections);
  evaluation_indices->reserve(num_images * max_detections);
  detection_scores->reserve(num_images * max_detections);
  int num_valid_ground_truth = 0;
  for (auto i = 0; i < num_images; ++i) {
    const ImageEvaluation& evaluation = evaluations[evaluation_index + i];

    for (int d = 0;
         d < (int)evaluation.detection_scores.size() && d < max_detections;
         ++d) { // detected instances
      evaluation_indices->push_back(evaluation_index + i);
      image_detection_indices->push_back(d);
      detection_scores->push_back(evaluation.detection_scores[d]);
    }
    for (auto ground_truth_ignore : evaluation.ground_truth_ignores) {
      if (!ground_truth_ignore) {
        ++num_valid_ground_truth;
      }
    }
  }

  // Sort detections by decreasing score, using stable sort to match
  // python implementation
  detection_sorted_indices->resize(detection_scores->size());
  std::iota(
      detection_sorted_indices->begin(), detection_sorted_indices->end(), 0);
  std::stable_sort(
      detection_sorted_indices->begin(),
      detection_sorted_indices->end(),
      [&detection_scores](size_t j1, size_t j2) {
        return (*detection_scores)[j1] > (*detection_scores)[j2];
      });

  return num_valid_ground_truth;
}

// Helper function to Accumulate()
// Compute a precision recall curve given a sorted list of detected instances
// encoded in evaluations, evaluation_indices, detection_scores,
// detection_sorted_indices, image_detection_indices (see
// BuildSortedDetectionList()). Using vectors precisions and recalls
// and temporary storage, output the results into precisions_out, recalls_out,
// and scores_out, which are large buffers containing many precion/recall curves
// for all possible parameter settings, with precisions_out_index and
// recalls_out_index defining the applicable indices to store results.
void ComputePrecisionRecallCurve(
    const int64_t precisions_out_index,
    const int64_t precisions_out_stride,
    const int64_t recalls_out_index,
    const std::vector<double>& recall_thresholds,
    const int iou_threshold_index,
    const int num_iou_thresholds,
    const int num_valid_ground_truth,
    const std::vector<ImageEvaluation>& evaluations,
    const std::vector<uint64_t>& evaluation_indices,
    const std::vector<double>& detection_scores,
    const std::vector<uint64_t>& detection_sorted_indices,
    const std::vector<uint64_t>& image_detection_indices,
    std::vector<double>* precisions,
    std::vector<double>* recalls,
    std::vector<double>* precisions_out,
    std::vector<double>* scores_out,
    std::vector<double>* recalls_out) {
  assert(recalls_out->size() > recalls_out_index);

  // Compute precision/recall for each instance in the sorted list of detections
  int64_t true_positives_sum = 0, false_positives_sum = 0;
  precisions->clear();
  recalls->clear();
  precisions->reserve(detection_sorted_indices.size());
  recalls->reserve(detection_sorted_indices.size());
  assert(!evaluations.empty() || detection_sorted_indices.empty());
  for (auto detection_sorted_index : detection_sorted_indices) {
    const ImageEvaluation& evaluation =
        evaluations[evaluation_indices[detection_sorted_index]];
    const auto num_detections =
        evaluation.detection_matches.size() / num_iou_thresholds;
    const auto detection_index = iou_threshold_index * num_detections +
        image_detection_indices[detection_sorted_index];
    assert(evaluation.detection_matches.size() > detection_index);
    assert(evaluation.detection_ignores.size() > detection_index);
    const int64_t detection_match =
        evaluation.detection_matches[detection_index];
    const bool detection_ignores =
        evaluation.detection_ignores[detection_index];
    const auto true_positive = detection_match > 0 && !detection_ignores;
    const auto false_positive = detection_match == 0 && !detection_ignores;
    if (true_positive) {
      ++true_positives_sum;
    }
    if (false_positive) {
      ++false_positives_sum;
    }

    const double recall =
        static_cast<double>(true_positives_sum) / num_valid_ground_truth;
    recalls->push_back(recall);
    const int64_t num_valid_detections =
        true_positives_sum + false_positives_sum;
    const double precision = num_valid_detections > 0
        ? static_cast<double>(true_positives_sum) / num_valid_detections
        : 0.0;
    precisions->push_back(precision);
  }

  (*recalls_out)[recalls_out_index] = !recalls->empty() ? recalls->back() : 0;

  for (int64_t i = static_cast<int64_t>(precisions->size()) - 1; i > 0; --i) {
    if ((*precisions)[i] > (*precisions)[i - 1]) {
      (*precisions)[i - 1] = (*precisions)[i];
    }
  }

  // Sample the per instance precision/recall list at each recall threshold
  for (size_t r = 0; r < recall_thresholds.size(); ++r) {
    // first index in recalls >= recall_thresholds[r]
    std::vector<double>::iterator low = std::lower_bound(
        recalls->begin(), recalls->end(), recall_thresholds[r]);
    size_t precisions_index = low - recalls->begin();

    const auto results_ind = precisions_out_index + r * precisions_out_stride;
    assert(results_ind < precisions_out->size());
    assert(results_ind < scores_out->size());
    if (precisions_index < precisions->size()) {
      (*precisions_out)[results_ind] = (*precisions)[precisions_index];
      (*scores_out)[results_ind] =
          detection_scores[detection_sorted_indices[precisions_index]];
    } else {
      (*precisions_out)[results_ind] = 0;
      (*scores_out)[results_ind] = 0;
    }
  }
}
py::dict Accumulate(
    const py::object& params,
    const std::vector<ImageEvaluation>& evaluations) {
  const std::vector<double> recall_thresholds =
      list_to_vec<double>(params.attr("recThrs"));
  const std::vector<int> max_detections =
      list_to_vec<int>(params.attr("maxDets"));
  const int num_iou_thresholds = py::len(params.attr("iouThrs"));
  const int num_recall_thresholds = py::len(params.attr("recThrs"));
  const int num_categories = params.attr("useCats").cast<int>() == 1
      ? py::len(params.attr("catIds"))
      : 1;
  const int num_area_ranges = py::len(params.attr("areaRng"));
  const int num_max_detections = py::len(params.attr("maxDets"));
  const int num_images = py::len(params.attr("imgIds"));

  std::vector<double> precisions_out(
      num_iou_thresholds * num_recall_thresholds * num_categories *
          num_area_ranges * num_max_detections,
      -1);
  std::vector<double> recalls_out(
      num_iou_thresholds * num_categories * num_area_ranges *
          num_max_detections,
      -1);
  std::vector<double> scores_out(
      num_iou_thresholds * num_recall_thresholds * num_categories *
          num_area_ranges * num_max_detections,
      -1);

  // Consider the list of all detected instances in the entire dataset in one
  // large list.  evaluation_indices, detection_scores,
  // image_detection_indices, and detection_sorted_indices all have the same
  // length as this list, such that each entry corresponds to one detected
  // instance
  std::vector<uint64_t> evaluation_indices; // indices into evaluations[]
  std::vector<double> detection_scores; // detection scores of each instance
  std::vector<uint64_t> detection_sorted_indices; // sorted indices of all
                                                  // instances in the dataset
  std::vector<uint64_t>
      image_detection_indices; // indices into the list of detected instances in
                               // the same image as each instance
  std::vector<double> precisions, recalls;

  for (auto c = 0; c < num_categories; ++c) {
    for (auto a = 0; a < num_area_ranges; ++a) {
      for (auto m = 0; m < num_max_detections; ++m) {
        // The COCO PythonAPI assumes evaluations[] (the return value of
        // COCOeval::EvaluateImages() is one long list storing results for each
        // combination of category, area range, and image id, with categories in
        // the outermost loop and images in the innermost loop.
        const int64_t evaluations_index =
            c * num_area_ranges * num_images + a * num_images;
        int num_valid_ground_truth = BuildSortedDetectionList(
            evaluations,
            evaluations_index,
            num_images,
            max_detections[m],
            &evaluation_indices,
            &detection_scores,
            &detection_sorted_indices,
            &image_detection_indices);

        if (num_valid_ground_truth == 0) {
          continue;
        }

        for (auto t = 0; t < num_iou_thresholds; ++t) {
          // recalls_out is a flattened vectors representing a
          // num_iou_thresholds X num_categories X num_area_ranges X
          // num_max_detections matrix
          const int64_t recalls_out_index =
              t * num_categories * num_area_ranges * num_max_detections +
              c * num_area_ranges * num_max_detections +
              a * num_max_detections + m;

          // precisions_out and scores_out are flattened vectors
          // representing a num_iou_thresholds X num_recall_thresholds X
          // num_categories X num_area_ranges X num_max_detections matrix
          const int64_t precisions_out_stride =
              num_categories * num_area_ranges * num_max_detections;
          const int64_t precisions_out_index = t * num_recall_thresholds *
                  num_categories * num_area_ranges * num_max_detections +
              c * num_area_ranges * num_max_detections +
              a * num_max_detections + m;

          ComputePrecisionRecallCurve(
              precisions_out_index,
              precisions_out_stride,
              recalls_out_index,
              recall_thresholds,
              t,
              num_iou_thresholds,
              num_valid_ground_truth,
              evaluations,
              evaluation_indices,
              detection_scores,
              detection_sorted_indices,
              image_detection_indices,
              &precisions,
              &recalls,
              &precisions_out,
              &scores_out,
              &recalls_out);
        }
      }
    }
  }

  time_t rawtime;
  struct tm local_time;
  std::array<char, 200> buffer;
  time(&rawtime);
#ifdef _WIN32
  localtime_s(&local_time, &rawtime);
#else
  localtime_r(&rawtime, &local_time);
#endif
  strftime(
      buffer.data(), 200, "%Y-%m-%d %H:%num_max_detections:%S", &local_time);
  return py::dict(
      "params"_a = params,
      "counts"_a = std::vector<int64_t>(
          {num_iou_thresholds,
           num_recall_thresholds,
           num_categories,
           num_area_ranges,
           num_max_detections}),
      "date"_a = buffer,
      "precision"_a = precisions_out,
      "recall"_a = recalls_out,
      "scores"_a = scores_out);
}

} // namespace COCOeval

} // namespace detectron2
