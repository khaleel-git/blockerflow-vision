package main

import _ "embed"

// The NudeNet 320n detector, release v3.4-weights of notAI-tech/NudeNet (AGPL-3.0).
//
//go:embed model/nudenet_320n.onnx
var nudenet []byte

// toCHW turns RGB bytes (row-major, three a pixel) into the model's input: three planes, 0..1.
func toCHW(rgb []byte, dst []float32) {
	plane := Side * Side
	for i := 0; i < plane; i++ {
		dst[i] = float32(rgb[i*3]) / 255
		dst[plane+i] = float32(rgb[i*3+1]) / 255
		dst[2*plane+i] = float32(rgb[i*3+2]) / 255
	}
}
