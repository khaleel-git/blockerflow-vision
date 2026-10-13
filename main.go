// Command blockerflow-vision runs the NudeNet 320n nudity detector for Blockerflow for Mac.
//
// Blockerflow starts it as a separate program and talks to it over stdin and stdout (see protocol.go):
// it sends 320 x 320 pictures and gets back one score per NudeNet class. Everything runs on this Mac.
package main

import (
	"fmt"
	"os"
)

func logf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "blockerflow-vision: "+format+"\n", args...)
}

func main() {
	if len(os.Args) > 1 && (os.Args[1] == "-version" || os.Args[1] == "--version") {
		fmt.Printf("blockerflow-vision protocol %d, NudeNet 320n v3.4-weights\n", Version)
		return
	}
	m, err := NewModel()
	if err != nil {
		logf("%v", err)
		os.Exit(1)
	}
	defer m.Close()
	if err := writeHello(os.Stdout, m.Accel); err != nil {
		os.Exit(1)
	}
	if err := serve(os.Stdin, os.Stdout, m); err != nil {
		logf("%v", err)
		os.Exit(1)
	}
}
