package ingestor

import (
	"fmt"

	"github.com/nlpodyssey/spago/ag"
	"nickandperla.net/mlp/config"
)

type Ingestor interface {
	Ingest() ([]ag.Node, []ag.Node)
	Close()
}

func NewIngestor(ic config.IngestorConfig) Ingestor {
	switch ic.Type {
	case config.CSVIngestor:
		return NewCSVIngestor(ic)
	default:
		panic(fmt.Errorf("Unknown IngestorType: %v", ic.Type))
	}
}
