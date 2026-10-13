package main

import "testing"

func TestMaxPerClassBothLayouts(t *testing.T) {
	// 3 anchors, 2 classes: [1, 6, 3] (fields first)
	a := []float32{
		0, 0, 0, // x
		0, 0, 0, // y
		0, 0, 0, // w
		0, 0, 0, // h
		0.1, 0.9, 0.3, // class 0
		0.5, 0.2, 0.4, // class 1
	}
	got, err := MaxPerClass(a, []int64{1, 6, 3}, 2)
	if err != nil || got[0] != 0.9 || got[1] != 0.5 {
		t.Fatalf("fields first: got %v %v", got, err)
	}
	// the same, anchors first: [1, 3, 6]
	b := []float32{
		0, 0, 0, 0, 0.1, 0.5,
		0, 0, 0, 0, 0.9, 0.2,
		0, 0, 0, 0, 0.3, 0.4,
	}
	got, err = MaxPerClass(b, []int64{1, 3, 6}, 2)
	if err != nil || got[0] != 0.9 || got[1] != 0.5 {
		t.Fatalf("anchors first: got %v %v", got, err)
	}
	if _, err := MaxPerClass(make([]float32, 100), []int64{1, 10, 10}, 2); err == nil {
		t.Error("a shape with no axis of 4+classes must be refused")
	}
	if _, err := MaxPerClass(nil, []int64{2, 6, 3}, 2); err == nil {
		t.Error("a batch of 2 must be refused")
	}
	if _, err := MaxPerClass(make([]float32, 5), []int64{1, 6, 3}, 2); err == nil {
		t.Error("too few values must be refused")
	}
}
