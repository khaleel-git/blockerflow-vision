package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"strings"
	"testing"
)

type fakeScorer struct {
	scores []float32
	err    error
	got    [][]byte
}

func (f *fakeScorer) Score(rgb []byte) ([]float32, error) {
	f.got = append(f.got, append([]byte(nil), rgb...))
	return f.scores, f.err
}

func readHello(t *testing.T, r io.Reader) (version, side, classes uint32, names []string, accel byte) {
	t.Helper()
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil || string(magic) != Magic {
		t.Fatalf("magic %q %v", magic, err)
	}
	var v [3]uint32
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		t.Fatal(err)
	}
	var n uint32
	binary.Read(r, binary.LittleEndian, &n)
	buf := make([]byte, n)
	io.ReadFull(r, buf)
	a := make([]byte, 1)
	io.ReadFull(r, a)
	return v[0], v[1], v[2], strings.Split(string(buf), "\n"), a[0]
}

func TestHelloDescribesTheModel(t *testing.T) {
	var out bytes.Buffer
	if err := writeHello(&out, AccelCoreML); err != nil {
		t.Fatal(err)
	}
	version, side, classes, names, accel := readHello(t, &out)
	if version != 1 || side != 320 || classes != 18 || len(names) != 18 || accel != AccelCoreML {
		t.Fatalf("hello: version %d side %d classes %d names %d accel %d", version, side, classes, len(names), accel)
	}
	if names[0] != "FEMALE_GENITALIA_COVERED" || names[17] != "BUTTOCKS_COVERED" {
		t.Errorf("label order changed: %v", names)
	}
	if out.Len() != 0 {
		t.Errorf("%d stray bytes after the hello", out.Len())
	}
}

func TestServeAnswersEachRequestAndStopsAtEOF(t *testing.T) {
	scores := make([]float32, len(Labels))
	scores[1], scores[17] = 0.75, 0.25
	f := &fakeScorer{scores: scores}
	req1 := bytes.Repeat([]byte{7}, RequestLen)
	req2 := bytes.Repeat([]byte{9}, RequestLen)
	var out bytes.Buffer
	if err := serve(bytes.NewReader(append(req1, req2...)), &out, f); err != nil {
		t.Fatalf("clean EOF must end serve without an error, got %v", err)
	}
	if len(f.got) != 2 || !bytes.Equal(f.got[0], req1) || !bytes.Equal(f.got[1], req2) {
		t.Fatalf("scorer saw %d requests", len(f.got))
	}
	for i := 0; i < 2; i++ {
		status, _ := out.ReadByte()
		if status != StatusOK {
			t.Fatalf("reply %d: status %d", i, status)
		}
		got := make([]float32, len(Labels))
		binary.Read(&out, binary.LittleEndian, got)
		if got[1] != 0.75 || got[17] != 0.25 {
			t.Errorf("reply %d: %v", i, got)
		}
	}
	if out.Len() != 0 {
		t.Errorf("%d stray bytes", out.Len())
	}
}

func TestServeReportsAScoringErrorAndCarriesOn(t *testing.T) {
	f := &fakeScorer{err: errors.New("CoreML said no")}
	req := make([]byte, RequestLen)
	var out bytes.Buffer
	if err := serve(bytes.NewReader(append(req, req...)), &out, f); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		status, _ := out.ReadByte()
		var n uint32
		binary.Read(&out, binary.LittleEndian, &n)
		msg := make([]byte, n)
		io.ReadFull(&out, msg)
		if status != StatusError || string(msg) != "CoreML said no" {
			t.Errorf("reply %d: status %d message %q", i, status, msg)
		}
	}
}

func TestServeRefusesWrongScoreCount(t *testing.T) {
	f := &fakeScorer{scores: []float32{0.5}}
	var out bytes.Buffer
	serve(bytes.NewReader(make([]byte, RequestLen)), &out, f)
	if status, _ := out.ReadByte(); status != StatusError {
		t.Errorf("1 score instead of 18 must be an error reply, got status %d", status)
	}
}

func TestServeFailsOnAShortRequest(t *testing.T) {
	var out bytes.Buffer
	err := serve(bytes.NewReader(make([]byte, RequestLen-1)), &out, &fakeScorer{})
	if err == nil {
		t.Error("a request cut short must be an error")
	}
	if out.Len() != 0 {
		t.Error("nothing must be answered to a partial request")
	}
}

func TestToCHWScalesAndReordersChannels(t *testing.T) {
	rgb := make([]byte, RequestLen)
	rgb[0], rgb[1], rgb[2] = 255, 0, 51 // first pixel
	dst := make([]float32, RequestLen)
	toCHW(rgb, dst)
	plane := Side * Side
	if dst[0] != 1 || dst[plane] != 0 || math.Abs(float64(dst[2*plane]-0.2)) > 1e-6 {
		t.Errorf("got r %v g %v b %v", dst[0], dst[plane], dst[2*plane])
	}
}
