//go:build windows

package main

// Windows: ONNX Runtime's C API called straight from Go (syscall), so the program needs no C compiler
// and builds for x64 and arm64 from one machine. Only the CPU provider is used.
//
// The C API is a table of function pointers (OrtApi). Its order never changes between releases (new
// functions are only added at the end), so a function is called by its position in the table. The
// positions below are counted from onnxruntime_c_api.h of ONNX Runtime 1.23.2 (ORT_API_VERSION 23).

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"
)

const ortAPIVersion = 23

// Positions in OrtApi.
const (
	fnGetErrorMessage                  = 2
	fnCreateEnv                        = 3
	fnCreateSessionFromArray           = 8
	fnRun                              = 9
	fnCreateSessionOptions             = 10
	fnSetSessionGraphOptimizationLevel = 23
	fnSetIntraOpNumThreads             = 24
	fnSetInterOpNumThreads             = 25
	fnSessionGetInputName              = 36
	fnSessionGetOutputName             = 37
	fnCreateTensorWithDataAsOrtValue   = 49
	fnGetTensorMutableData             = 51
	fnGetDimensionsCount               = 61
	fnGetDimensions                    = 62
	fnGetTensorTypeAndShape            = 65
	fnCreateCpuMemoryInfo              = 69
	fnAllocatorFree                    = 76
	fnGetAllocatorWithDefaultOptions   = 78
	fnReleaseEnv                       = 92
	fnReleaseStatus                    = 93
	fnReleaseMemoryInfo                = 94
	fnReleaseSession                   = 95
	fnReleaseValue                     = 96
	fnReleaseTensorTypeAndShapeInfo    = 99
	fnReleaseSessionOptions            = 100
	fnAddSessionConfigEntry            = 130
)

const (
	ortLoggingLevelError = 3
	ortEnableAll         = 99 // GraphOptimizationLevel
	ortDeviceAllocator   = 0
	ortMemTypeDefault    = 0
	ortTensorFloat       = 1 // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
)

// LibraryPath is where onnxruntime.dll is looked for: BLOCKERFLOW_ORT_LIB if set, else next to this
// executable (Blockerflow's package puts both in its vision folder).
func LibraryPath() string {
	if p := os.Getenv("BLOCKERFLOW_ORT_LIB"); p != "" {
		return p
	}
	exe, err := os.Executable()
	if err != nil {
		return "onnxruntime.dll"
	}
	return filepath.Join(filepath.Dir(exe), "onnxruntime.dll")
}

// ort is the loaded library's function table.
type ort struct{ table uintptr }

func (o ort) fn(i int) uintptr {
	return *(*uintptr)(cPtr(o.table + uintptr(i)*unsafe.Sizeof(uintptr(0))))
}

// call runs a function that returns an OrtStatus*: nil on success, else the error it describes.
func (o ort) call(i int, args ...uintptr) error {
	status, _, _ := syscall.SyscallN(o.fn(i), args...)
	if status == 0 {
		return nil
	}
	msg, _, _ := syscall.SyscallN(o.fn(fnGetErrorMessage), status)
	err := fmt.Errorf("onnxruntime: %s", cString(msg))
	syscall.SyscallN(o.fn(fnReleaseStatus), status)
	return err
}

func (o ort) release(i int, p uintptr) {
	if p != 0 {
		syscall.SyscallN(o.fn(i), p)
	}
}

func cString(p uintptr) string {
	if p == 0 {
		return ""
	}
	var b []byte
	for i := uintptr(0); ; i++ {
		c := *(*byte)(cPtr(p + i))
		if c == 0 {
			return string(b)
		}
		b = append(b, c)
	}
}

// cPtr turns an address ONNX Runtime handed out (memory it owns, never Go's) into a pointer.
func cPtr(p uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&p)) }

func cBytes(s string) []byte { return append([]byte(s), 0) }

// Model runs NudeNet on one 320 x 320 picture at a time. Not safe for concurrent use.
type Model struct {
	api     ort
	env     uintptr
	session uintptr
	memory  uintptr
	input   uintptr // OrtValue over data
	data    []float32
	shape   []int64
	inName  []byte
	outName []byte
	Accel   byte
}

// threads: four that sleep between runs, as on Android (device-measured: one thread was 2.5 times
// slower), fewer on a PC with fewer than eight logical cores so the rest stays free.
func threads() int {
	return max(1, min(4, runtime.NumCPU()/2))
}

// NewModel loads onnxruntime.dll and the embedded model on the CPU.
func NewModel() (*Model, error) {
	path := LibraryPath()
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("ONNX Runtime library not found at %s", path)
	}
	dll, err := syscall.LoadDLL(path)
	if err != nil {
		return nil, fmt.Errorf("cannot load %s: %w", path, err)
	}
	getAPIBase, err := dll.FindProc("OrtGetApiBase")
	if err != nil {
		return nil, err
	}
	base, _, _ := getAPIBase.Call()
	if base == 0 {
		return nil, fmt.Errorf("OrtGetApiBase returned nothing")
	}
	// OrtApiBase { GetApi(uint32), GetVersionString() }
	getAPI := *(*uintptr)(cPtr(base))
	table, _, _ := syscall.SyscallN(getAPI, ortAPIVersion)
	if table == 0 {
		getVersion := *(*uintptr)(cPtr(base + unsafe.Sizeof(uintptr(0))))
		v, _, _ := syscall.SyscallN(getVersion)
		return nil, fmt.Errorf("ONNX Runtime %s does not offer API version %d", cString(v), ortAPIVersion)
	}
	m := &Model{api: ort{table}, Accel: AccelCPU}
	if err := m.open(); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

func (m *Model) open() error {
	a := m.api
	logID := cBytes("blockerflow-vision")
	if err := a.call(fnCreateEnv, ortLoggingLevelError, uintptr(unsafe.Pointer(&logID[0])), uintptr(unsafe.Pointer(&m.env))); err != nil {
		return err
	}
	var opts uintptr
	if err := a.call(fnCreateSessionOptions, uintptr(unsafe.Pointer(&opts))); err != nil {
		return err
	}
	defer a.release(fnReleaseSessionOptions, opts)
	key, value := cBytes("session.intra_op.allow_spinning"), cBytes("0")
	for _, err := range []error{
		a.call(fnSetIntraOpNumThreads, opts, uintptr(threads())),
		a.call(fnSetInterOpNumThreads, opts, 1),
		a.call(fnSetSessionGraphOptimizationLevel, opts, ortEnableAll),
		a.call(fnAddSessionConfigEntry, opts, uintptr(unsafe.Pointer(&key[0])), uintptr(unsafe.Pointer(&value[0]))),
	} {
		if err != nil {
			return err
		}
	}
	if err := a.call(fnCreateSessionFromArray, m.env, uintptr(unsafe.Pointer(&nudenet[0])), uintptr(len(nudenet)), opts, uintptr(unsafe.Pointer(&m.session))); err != nil {
		return err
	}
	var err error
	if m.inName, err = m.ioName(fnSessionGetInputName); err != nil {
		return err
	}
	if m.outName, err = m.ioName(fnSessionGetOutputName); err != nil {
		return err
	}
	if err := a.call(fnCreateCpuMemoryInfo, ortDeviceAllocator, ortMemTypeDefault, uintptr(unsafe.Pointer(&m.memory))); err != nil {
		return err
	}
	m.data = make([]float32, 3*Side*Side)
	m.shape = []int64{1, 3, Side, Side}
	return a.call(fnCreateTensorWithDataAsOrtValue, m.memory,
		uintptr(unsafe.Pointer(&m.data[0])), uintptr(len(m.data)*4),
		uintptr(unsafe.Pointer(&m.shape[0])), uintptr(len(m.shape)), ortTensorFloat,
		uintptr(unsafe.Pointer(&m.input)))
}

// ioName reads the first input or output name, NUL-terminated, into Go memory.
func (m *Model) ioName(fn int) ([]byte, error) {
	var alloc, name uintptr
	if err := m.api.call(fnGetAllocatorWithDefaultOptions, uintptr(unsafe.Pointer(&alloc))); err != nil {
		return nil, err
	}
	if err := m.api.call(fn, m.session, 0, alloc, uintptr(unsafe.Pointer(&name))); err != nil {
		return nil, err
	}
	s := cString(name)
	m.api.call(fnAllocatorFree, alloc, name)
	return cBytes(s), nil
}

// Score returns the best score each class reached anywhere in the picture, in Labels order.
func (m *Model) Score(rgb []byte) ([]float32, error) {
	if len(rgb) != RequestLen {
		return nil, fmt.Errorf("picture has %d bytes, want %d", len(rgb), RequestLen)
	}
	toCHW(rgb, m.data)
	a := m.api
	inNames := []uintptr{uintptr(unsafe.Pointer(&m.inName[0]))}
	outNames := []uintptr{uintptr(unsafe.Pointer(&m.outName[0]))}
	inputs := []uintptr{m.input}
	outputs := []uintptr{0}
	err := a.call(fnRun, m.session, 0,
		uintptr(unsafe.Pointer(&inNames[0])), uintptr(unsafe.Pointer(&inputs[0])), 1,
		uintptr(unsafe.Pointer(&outNames[0])), 1, uintptr(unsafe.Pointer(&outputs[0])))
	runtime.KeepAlive(m)
	runtime.KeepAlive(inNames)
	runtime.KeepAlive(outNames)
	if err != nil {
		return nil, err
	}
	out := outputs[0]
	defer a.release(fnReleaseValue, out)

	var info, count uintptr
	if err := a.call(fnGetTensorTypeAndShape, out, uintptr(unsafe.Pointer(&info))); err != nil {
		return nil, err
	}
	defer a.release(fnReleaseTensorTypeAndShapeInfo, info)
	if err := a.call(fnGetDimensionsCount, info, uintptr(unsafe.Pointer(&count))); err != nil {
		return nil, err
	}
	if count == 0 || count > 8 {
		return nil, fmt.Errorf("output has %d dimensions", count)
	}
	dims := make([]int64, count)
	if err := a.call(fnGetDimensions, info, uintptr(unsafe.Pointer(&dims[0])), count); err != nil {
		return nil, err
	}
	n := int64(1)
	for _, d := range dims {
		if d <= 0 {
			return nil, fmt.Errorf("unexpected output shape %v", dims)
		}
		n *= d
	}
	var p uintptr
	if err := a.call(fnGetTensorMutableData, out, uintptr(unsafe.Pointer(&p))); err != nil {
		return nil, err
	}
	values := unsafe.Slice((*float32)(cPtr(p)), n)
	return MaxPerClass(values, dims, len(Labels))
}

// Close releases the model.
func (m *Model) Close() {
	a := m.api
	if a.table == 0 {
		return
	}
	a.release(fnReleaseValue, m.input)
	a.release(fnReleaseMemoryInfo, m.memory)
	a.release(fnReleaseSession, m.session)
	a.release(fnReleaseEnv, m.env)
	m.input, m.memory, m.session, m.env = 0, 0, 0, 0
}
