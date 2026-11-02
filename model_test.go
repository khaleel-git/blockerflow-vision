package main

import (
	"bufio"
	"encoding/binary"
	"image"
	"image/draw"
	_ "image/jpeg"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func libOrSkip(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(LibraryPath()); err != nil {
		t.Skipf("ONNX Runtime library not found (%v); set BLOCKERFLOW_ORT_LIB to run the model tests", err)
	}
}

// portraitRequest is the test photo as a request: padded to a square at the bottom and right, scaled
// to 320 by averaging, RGB.
func portraitRequest(t *testing.T) []byte {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "portrait.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	b := img.Bounds()
	side := max(b.Dx(), b.Dy())
	sq := image.NewRGBA(image.Rect(0, 0, side, side))
	draw.Draw(sq, image.Rect(0, 0, b.Dx(), b.Dy()), img, b.Min, draw.Src)
	req := make([]byte, RequestLen)
	for y := 0; y < Side; y++ {
		for x := 0; x < Side; x++ {
			var r, g, bl, n int
			for sy := y * side / Side; sy < (y+1)*side/Side; sy++ {
				for sx := x * side / Side; sx < (x+1)*side/Side; sx++ {
					p := sq.PixOffset(sx, sy)
					r, g, bl, n = r+int(sq.Pix[p]), g+int(sq.Pix[p+1]), bl+int(sq.Pix[p+2]), n+1
				}
			}
			o := (y*Side + x) * 3
			req[o], req[o+1], req[o+2] = byte(r/n), byte(g/n), byte(bl/n)
		}
	}
	return req
}

func checkPortraitScores(t *testing.T, s []float32) {
	t.Helper()
	idx := map[string]int{}
	for i, l := range Labels {
		idx[l] = i
	}
	if face := max(s[idx["FACE_FEMALE"]], s[idx["FACE_MALE"]]); face < 0.5 {
		t.Errorf("the model should see the face, got %.2f (%v)", face, s)
	}
	for _, l := range []string{"FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED", "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED"} {
		if s[idx[l]] >= 0.15 {
			t.Errorf("an ordinary portrait scored %s %.2f", l, s[idx[l]])
		}
	}
}

func TestModelSeesAFaceAndNothingExplicit(t *testing.T) {
	libOrSkip(t)
	m, err := NewModel()
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s, err := m.Score(portraitRequest(t))
	if err != nil {
		t.Fatal(err)
	}
	checkPortraitScores(t, s)
	if _, err := m.Score(make([]byte, 10)); err == nil {
		t.Error("a picture of the wrong size must be refused")
	}
}

// TestProgramEndToEnd builds the command and talks to it as Blockerflow does.
func TestProgramEndToEnd(t *testing.T) {
	libOrSkip(t)
	bin := filepath.Join(t.TempDir(), "blockerflow-vision")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin)
	cmd.Env = append(os.Environ(), "BLOCKERFLOW_ORT_LIB="+LibraryPath())
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	r := bufio.NewReader(stdout)
	_, _, classes, names, _ := readHello(t, r)
	if classes != 18 || len(names) != 18 {
		t.Fatalf("hello: %d classes", classes)
	}
	req := portraitRequest(t)
	for i := 0; i < 2; i++ {
		if _, err := stdin.Write(req); err != nil {
			t.Fatal(err)
		}
		status, err := r.ReadByte()
		if err != nil || status != StatusOK {
			t.Fatalf("status %d %v", status, err)
		}
		s := make([]float32, 18)
		if err := binary.Read(r, binary.LittleEndian, s); err != nil {
			t.Fatal(err)
		}
		checkPortraitScores(t, s)
	}
	stdin.Close()
	done := make(chan error, 1)
	go func() { io.Copy(io.Discard, r); done <- cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("must exit cleanly when its input closes: %v", err)
		}
	case <-time.After(5 * time.Second):
		cmd.Process.Kill()
		t.Error("did not exit after its input closed")
	}
}
