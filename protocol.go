package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"
)

// Protocol v1, over stdin and stdout, little-endian, fixed sizes:
//
//	hello  (helper to client, once): "BFV1", uint32 version, uint32 side, uint32 classes,
//	        uint32 n + n bytes of class names joined by "\n", uint8 accelerator
//	request (client to helper):      Side*Side*3 bytes, RGB, row-major
//	reply   (helper to client):      uint8 status; StatusOK: classes float32 scores in Labels order;
//	                                 StatusError: uint32 n + n bytes of message
//
// The helper exits when its input closes.
const (
	Magic      = "BFV1"
	Version    = 1
	Side       = 320
	RequestLen = Side * Side * 3

	StatusOK    = 0
	StatusError = 1

	AccelCPU    = 0
	AccelCoreML = 1
)

type scorer interface {
	Score(rgb []byte) ([]float32, error)
}

func writeHello(w io.Writer, accel byte) error {
	names := strings.Join(Labels, "\n")
	bw := bufio.NewWriter(w)
	bw.WriteString(Magic)
	binary.Write(bw, binary.LittleEndian, [4]uint32{Version, Side, uint32(len(Labels)), uint32(len(names))})
	bw.WriteString(names)
	bw.WriteByte(accel)
	return bw.Flush()
}

// serve answers requests until in ends. It returns nil when in ends between requests, an error when
// it ends inside one or out cannot be written.
func serve(in io.Reader, out io.Writer, m scorer) error {
	br := bufio.NewReaderSize(in, RequestLen)
	bw := bufio.NewWriter(out)
	req := make([]byte, RequestLen)
	for {
		n, err := io.ReadFull(br, req)
		if err == io.EOF && n == 0 {
			return nil
		}
		if err != nil {
			return fmt.Errorf("request cut short after %d of %d bytes: %w", n, RequestLen, err)
		}
		scores, err := m.Score(req)
		if err == nil && len(scores) != len(Labels) {
			err = fmt.Errorf("model gave %d scores, want %d", len(scores), len(Labels))
		}
		if err != nil {
			msg := err.Error()
			bw.WriteByte(StatusError)
			binary.Write(bw, binary.LittleEndian, uint32(len(msg)))
			bw.WriteString(msg)
		} else {
			bw.WriteByte(StatusOK)
			binary.Write(bw, binary.LittleEndian, scores)
		}
		if err := bw.Flush(); err != nil {
			return errors.Join(errors.New("cannot answer"), err)
		}
	}
}
