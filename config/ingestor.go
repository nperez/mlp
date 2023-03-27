package config

type IngestorType uint8

const (
	CSVIngestor = IngestorType(0)
)

type IngestorConfig struct {
	Type             IngestorType
	DatasetPath      string
	BatchSize        uint32
	InputIndicies    []uint32
	ExpectedIndicies []uint32
	FieldCount       uint32
}
