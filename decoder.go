package main

import "fmt"

const boxValues = 4

// MaxPerClass reads a YOLOv8 export's raw output, [1, 4+classes, anchors] (or the transposed
// [1, anchors, 4+classes]): for every anchor a box and one score per class. It returns the best score
// each class reached at any anchor; the boxes and non-maximum suppression are not needed for that.
func MaxPerClass(output []float32, dims []int64, classCount int) ([]float32, error) {
	if len(dims) != 3 || dims[0] != 1 {
		return nil, fmt.Errorf("unexpected output shape %v", dims)
	}
	a, b := int(dims[1]), int(dims[2])
	fields := boxValues + classCount
	if len(output) < a*b {
		return nil, fmt.Errorf("output has %d values, shape %v needs %d", len(output), dims, a*b)
	}
	best := make([]float32, classCount)
	switch fields {
	case a: // [1, fields, anchors]
		for c := 0; c < classCount; c++ {
			row := (boxValues + c) * b
			var max float32
			for i := 0; i < b; i++ {
				if v := output[row+i]; v > max {
					max = v
				}
			}
			best[c] = max
		}
	case b: // [1, anchors, fields]
		for i := 0; i < a; i++ {
			row := i*b + boxValues
			for c := 0; c < classCount; c++ {
				if v := output[row+c]; v > best[c] {
					best[c] = v
				}
			}
		}
	default:
		return nil, fmt.Errorf("output shape %v has no axis of %d", dims, fields)
	}
	return best, nil
}
