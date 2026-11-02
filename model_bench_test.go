package main

import (
	"fmt"
	"testing"
)

// TestPrintPortraitScores logs the 18 scores, to compare builds and machines (go test -v -run Print).
func TestPrintPortraitScores(t *testing.T) {
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
	line := ""
	for i, l := range Labels {
		if s[i] >= 0.001 {
			line += fmt.Sprintf(" %s=%.4f", l, s[i])
		}
	}
	t.Logf("portrait:%s", line)
}

func BenchmarkScore(b *testing.B) {
	if _, err := NewModel(); err != nil {
		b.Skip(err)
	}
	m, _ := NewModel()
	defer m.Close()
	req := portraitRequest(&testing.T{})
	m.Score(req)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Score(req); err != nil {
			b.Fatal(err)
		}
	}
}
