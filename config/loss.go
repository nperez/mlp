package config

import (
	"log"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/losses"
)

type LossFunc func([]ag.Node, []ag.Node, bool) ag.Node

var lossLookup = map[string]LossFunc{
	"MSE": losses.MSESeq,
	"MAE": losses.MAESeq,
	"CrossEntropy": func(a, b []ag.Node, c bool) ag.Node {
		return losses.CrossEntropySeq(a, hotOneIndex(b), c)
	},
}

func hotOneIndex(expected []ag.Node) []int {
	output := make([]int, len(expected))
	for i, exp := range expected {
		output[i] = exp.Value().ArgMax()
	}
	return output
}
func GetLossFunction(loss string) LossFunc {
	if f, ok := lossLookup[loss]; ok {
		return f
	} else {
		log.Fatalf("Unknown loss function: %s", loss)
		return nil
	}
}
