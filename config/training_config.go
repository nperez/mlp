package config

type TrainingConfig struct {
	StepCount        int
	Layers           []LayerConf
	ModelPath        string
	Optimizer        string
	LossFunction     string
	IngestorConfig   IngestorConfig
	StepReportingMod int
}
