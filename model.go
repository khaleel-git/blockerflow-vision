package main

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

// The NudeNet 320n detector, release v3.4-weights of notAI-tech/NudeNet (AGPL-3.0).
//
//go:embed model/nudenet_320n.onnx
var nudenet []byte

// LibraryPath is where the ONNX Runtime library is looked for: BLOCKERFLOW_ORT_LIB if set, else
// ../Frameworks/libonnxruntime.dylib from this executable (Contents/Helpers -> Contents/Frameworks in
// an app bundle).
func LibraryPath() string {
	if p := os.Getenv("BLOCKERFLOW_ORT_LIB"); p != "" {
		return p
	}
	exe, err := os.Executable()
	if err != nil {
		return "libonnxruntime.dylib"
	}
	return filepath.Join(filepath.Dir(exe), "..", "Frameworks", "libonnxruntime.dylib")
}

// Model runs NudeNet on one 320 x 320 picture at a time. Not safe for concurrent use.
type Model struct {
	session *ort.DynamicAdvancedSession
	input   *ort.Tensor[float32]
	data    []float32
	Accel   byte
}

// NewModel loads ONNX Runtime and the embedded model, on CoreML if it will take it, else the CPU.
func NewModel() (*Model, error) {
	path := LibraryPath()
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("ONNX Runtime library not found at %s", path)
	}
	ort.SetSharedLibraryPath(path)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}
	ins, outs, err := ort.GetInputOutputInfoWithONNXData(nudenet)
	if err != nil || len(ins) == 0 || len(outs) == 0 {
		return nil, fmt.Errorf("model has no readable inputs: %v", err)
	}
	accel := byte(AccelCoreML)
	session, err := newSession(ins[0].Name, outs[0].Name, true)
	if err != nil {
		logf("CoreML could not load the model, using the CPU: %v", err)
		accel = AccelCPU
		session, err = newSession(ins[0].Name, outs[0].Name, false)
	}
	if err != nil {
		return nil, err
	}
	data := make([]float32, 3*Side*Side)
	tensor, err := ort.NewTensor(ort.NewShape(1, 3, Side, Side), data)
	if err != nil {
		session.Destroy()
		return nil, err
	}
	return &Model{session: session, input: tensor, data: data, Accel: accel}, nil
}

// newSession opens the model on one thread that sleeps between runs. ONNX Runtime's default (a thread
// per core that spins after each run) cost three times the CPU for no gain a person would notice. With
// CoreML most of the network runs on the GPU or Neural Engine, which halves the CPU again and gives
// the same scores.
func newSession(in, out string, coreML bool) (*ort.DynamicAdvancedSession, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	if err := opts.SetIntraOpNumThreads(1); err != nil {
		return nil, err
	}
	if err := opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "0"); err != nil {
		return nil, err
	}
	if coreML {
		cml := map[string]string{"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"}
		if dir := coreMLCache(); dir != "" {
			cml["ModelCacheDirectory"] = dir
		}
		if err := opts.AppendExecutionProviderCoreMLV2(cml); err != nil {
			return nil, err
		}
	}
	return ort.NewDynamicAdvancedSessionWithONNXData(nudenet, []string{in}, []string{out}, opts)
}

// coreMLCache is where CoreML keeps the model it compiled, so later starts skip the compile:
// ~/Library/Caches/eu.khaleel.blockerflow/vision-coreml/<model hash>. "" (no cache) if it cannot be made.
func coreMLCache() string {
	base, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(nudenet)
	dir := filepath.Join(base, "eu.khaleel.blockerflow", "vision-coreml", hex.EncodeToString(sum[:6]))
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return ""
	}
	return dir
}

// toCHW turns RGB bytes (row-major, three a pixel) into the model's input: three planes, 0..1.
func toCHW(rgb []byte, dst []float32) {
	plane := Side * Side
	for i := 0; i < plane; i++ {
		dst[i] = float32(rgb[i*3]) / 255
		dst[plane+i] = float32(rgb[i*3+1]) / 255
		dst[2*plane+i] = float32(rgb[i*3+2]) / 255
	}
}

// Score returns the best score each class reached anywhere in the picture, in Labels order.
func (m *Model) Score(rgb []byte) ([]float32, error) {
	if len(rgb) != RequestLen {
		return nil, fmt.Errorf("picture has %d bytes, want %d", len(rgb), RequestLen)
	}
	toCHW(rgb, m.data)
	outputs := []ort.Value{nil}
	if err := m.session.Run([]ort.Value{m.input}, outputs); err != nil {
		return nil, err
	}
	defer outputs[0].Destroy()
	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output type %T", outputs[0])
	}
	return MaxPerClass(out.GetData(), out.GetShape(), len(Labels))
}

// Close releases the model.
func (m *Model) Close() {
	if m.input != nil {
		m.input.Destroy()
	}
	if m.session != nil {
		m.session.Destroy()
	}
}
