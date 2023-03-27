package ingestor

import (
	"encoding/csv"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"nickandperla.net/mlp/config"
)

type csvIngestor struct {
	file      *os.File
	csv       *csv.Reader
	config    config.IngestorConfig
	reservoir [][]string
}

// Uses a simple Reservoir Sampling method for creating batches

func NewCSVIngestor(conf config.IngestorConfig) Ingestor {

	csvFile, err := os.Open(conf.DatasetPath)

	if err != nil {
		log.Fatalf("Failed open dataset [%s]: %v", conf.DatasetPath, err)
	}

	r := csv.NewReader(csvFile)
	r.FieldsPerRecord = int(conf.FieldCount)
	r.ReuseRecord = true

	return &csvIngestor{
		file:      csvFile,
		csv:       r,
		config:    conf,
		reservoir: make([][]string, conf.BatchSize),
	}
}

func (i *csvIngestor) Ingest() ([]ag.Node, []ag.Node) {
	resLen := len(i.reservoir)
	inputs, expected := make([]ag.Node, resLen), make([]ag.Node, resLen)

	for q := 0; q < resLen; q++ {
		record, err := i.csv.Read()
		if err != nil {
			if err == io.EOF {
				log.Fatalf("Unexpected EOF initializing reservoir. BatchSize is larger than dataset")
			}
			log.Fatalf("Unexpected error initializing reservior at offset %d: %v", i.csv.InputOffset(), err)
		}

		i.reservoir[q] = record
	}

	upBound := resLen - 1 // need to make sure upper bound is valid index
	for {
		upBound++
		record, err := i.csv.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Fatalf("Unexpected error initializing reservior at offset %d: %v", i.csv.InputOffset(), err)
		}

		index := rand.Int31n(int32(upBound))
		if index <= int32(resLen-1) {
			i.reservoir[index] = record
		}
	}

	infloats, exfloats := make([]float32, len(i.config.InputIndicies)), make([]float32, len(i.config.ExpectedIndicies))

	for q := 0; q < resLen; q++ {
		record := i.reservoir[q]

		for x := 0; x < len(i.config.InputIndicies); x++ {
			index := i.config.InputIndicies[x]
			f, err := strconv.ParseFloat(record[index], 32)
			if err != nil {
				line, col := i.csv.FieldPos(int(index))
				log.Printf("FULL RECORD:\n\n%v\n\n", record)
				log.Fatalf("Failed to parse float64 from CSV dataset at line: %d, col: %d, field: %d, val: '%s'", line, col, index+1, record[index])
			}
			infloats[x] = float32(f)
		}
		inputs[q] = ag.Var(mat.NewVecDense[float32](infloats)).WithGrad(true)

		for z := 0; z < len(i.config.ExpectedIndicies); z++ {
			index := i.config.ExpectedIndicies[z]
			f, err := strconv.ParseFloat(record[index], 32)
			if err != nil {
				line, col := i.csv.FieldPos(int(index))
				log.Fatalf("Failed to parse float64 from CSV dataset at line: %d, col: %d, field: %d, val: '%s'", line, col, index+1, record[index])
			}
			exfloats[z] = float32(f)
		}
		expected[q] = ag.Var(mat.NewVecDense[float32](exfloats))
	}

	i.resetReader()

	return inputs, expected
}

func (i *csvIngestor) resetReader() {
	if _, err := i.file.Seek(0, 0); err != nil {
		log.Fatalf("Failed to rewind the CSV file back to the beinning: %v", err)
	}
	r := csv.NewReader(i.file)
	r.FieldsPerRecord = int(i.config.FieldCount)
	i.csv = r
}

func (i *csvIngestor) Close() {
	i.file.Close()
}
