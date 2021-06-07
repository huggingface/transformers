// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

namespace detectron2 {

namespace COCOeval {

// Annotation data for a single object instance in an image
struct InstanceAnnotation {
  InstanceAnnotation(
      uint64_t id,
      double score,
      double area,
      bool is_crowd,
      bool ignore)
      : id{id}, score{score}, area{area}, is_crowd{is_crowd}, ignore{ignore} {}
  uint64_t id;
  double score = 0.;
  double area = 0.;
  bool is_crowd = false;
  bool ignore = false;
};

// Stores intermediate results for evaluating detection results for a single
// image that has D detected instances and G ground truth instances. This stores
// matches between detected and ground truth instances
struct ImageEvaluation {
  // For each of the D detected instances, the id of the matched ground truth
  // instance, or 0 if unmatched
  std::vector<uint64_t> detection_matches;

  // The detection score of each of the D detected instances
  std::vector<double> detection_scores;

  // Marks whether or not each of G instances was ignored from evaluation (e.g.,
  // because it's outside area_range)
  std::vector<bool> ground_truth_ignores;

  // Marks whether or not each of D instances was ignored from evaluation (e.g.,
  // because it's outside aRng)
  std::vector<bool> detection_ignores;
};

template <class T>
using ImageCategoryInstances = std::vector<std::vector<std::vector<T>>>;

// C++ implementation of COCO API cocoeval.py::COCOeval.evaluateImg().  For each
// combination of image, category, area range settings, and IOU thresholds to
// evaluate, it matches detected instances to ground truth instances and stores
// the results into a vector of ImageEvaluation results, which will be
// interpreted by the COCOeval::Accumulate() function to produce precion-recall
// curves.  The parameters of nested vectors have the following semantics:
//   image_category_ious[i][c][d][g] is the intersection over union of the d'th
//     detected instance and g'th ground truth instance of
//     category category_ids[c] in image image_ids[i]
//   image_category_ground_truth_instances[i][c] is a vector of ground truth
//     instances in image image_ids[i] of category category_ids[c]
//   image_category_detection_instances[i][c] is a vector of detected
//     instances in image image_ids[i] of category category_ids[c]
std::vector<ImageEvaluation> EvaluateImages(
    const std::vector<std::array<double, 2>>& area_ranges, // vector of 2-tuples
    int max_detections,
    const std::vector<double>& iou_thresholds,
    const ImageCategoryInstances<std::vector<double>>& image_category_ious,
    const ImageCategoryInstances<InstanceAnnotation>&
        image_category_ground_truth_instances,
    const ImageCategoryInstances<InstanceAnnotation>&
        image_category_detection_instances);

// C++ implementation of COCOeval.accumulate(), which generates precision
// recall curves for each set of category, IOU threshold, detection area range,
// and max number of detections parameters.  It is assumed that the parameter
// evaluations is the return value of the functon COCOeval::EvaluateImages(),
// which was called with the same parameter settings params
py::dict Accumulate(
    const py::object& params,
    const std::vector<ImageEvaluation>& evalutations);

} // namespace COCOeval
} // namespace detectron2
