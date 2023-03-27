package trainer

import (
	"log"

	"github.com/BurntSushi/toml"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/gd"
	"github.com/nlpodyssey/spago/gd/adagrad"
	"github.com/nlpodyssey/spago/gd/adam"
	"github.com/nlpodyssey/spago/gd/lamb"
	"github.com/nlpodyssey/spago/gd/radam"
	"github.com/nlpodyssey/spago/gd/rmsprop"
	"github.com/nlpodyssey/spago/gd/sgd"
	"github.com/nlpodyssey/spago/nn"
	"nickandperla.net/mlp/config"
	"nickandperla.net/mlp/ingestor"
	"nickandperla.net/mlp/mlp"
)

type Trainer struct {
	Config       config.TrainingConfig
	Model        nn.StandardModel
	Optimizer    *gd.Optimizer
	Ingestor     ingestor.Ingestor
	LossFunction config.LossFunc
}

func NewTrainerFromConfigPath(path string) *Trainer {

	var conf config.TrainingConfig

	_, err := toml.DecodeFile(path, &conf)
	if err != nil {
		log.Fatalf("Failed to open training config [%s]: %v", path, err)
	}

	return NewTrainerFromConfig(conf)
}

func NewTrainerFromConfig(conf config.TrainingConfig) *Trainer {

	model := mlp.NewMLP(conf.Layers)

	return &Trainer{
		Config:       conf,
		Model:        model,
		Optimizer:    buildOptimizer(conf, model),
		Ingestor:     ingestor.NewIngestor(conf.IngestorConfig),
		LossFunction: config.GetLossFunction(conf.LossFunction),
	}
}

func buildOptimizer(conf config.TrainingConfig, model nn.StandardModel) *gd.Optimizer {
	var optimizer *gd.Optimizer

	switch config.GetOptimizerMethodName(conf.Optimizer) {
	case gd.SGD:
		optimizer = gd.NewOptimizer(model, sgd.New[float32](sgd.NewConfig(0.001, 0.9, true)))
	case gd.AdaGrad:
		optimizer = gd.NewOptimizer(model, adagrad.New[float32](adagrad.NewDefaultConfig()))
	case gd.Adam:
		optimizer = gd.NewOptimizer(model, adam.New[float32](adam.NewDefaultConfig()))
	case gd.RAdam:
		optimizer = gd.NewOptimizer(model, radam.New[float32](radam.NewDefaultConfig()))
	case gd.RMSProp:
		optimizer = gd.NewOptimizer(model, rmsprop.New[float32](rmsprop.NewDefaultConfig()))
	case gd.Lamb:
		optimizer = gd.NewOptimizer(model, lamb.New[float32](lamb.NewDefaultConfig()))
	default:
		log.Fatalf("Invalid or no optimizer configured")
	}

	return optimizer
}

func (t *Trainer) Train() {
	report_mod := t.Config.StepReportingMod
	for step := 0; step < t.Config.StepCount; step++ {
		inputs, expected := t.Ingestor.Ingest()
		output := t.Model.Forward(inputs...)
		loss := t.LossFunction(output, expected, true)
		ag.Backward(loss)
		if report_mod != 0 && step%report_mod == 0 {
			acc_accum := make([]ag.Node, len(expected))
			for i := range expected {
				out := output[i]
				exp := expected[i]
				if out.Value().ArgMax() == exp.Value().ArgMax() {
					acc_accum[i] = ag.Scalar(1.0)
				} else {
					acc_accum[i] = ag.Scalar(0.0)
				}

			}
			acc := ag.Mean(acc_accum)
			log.Printf("Step: %d, Loss: %.16f, Accuracy: %.16f", step, loss.Value().Scalar().F32(), acc.Value().Scalar().F32())
		}
		t.Optimizer.Do()
	}

}

func (t *Trainer) SaveModel() {
	t.Model.(*mlp.MLP).TrainingConfig = t.Config
	if err := nn.DumpToFile(t.Model, t.Config.ModelPath); err != nil {
		log.Fatalf("Failed to save model: %v", err)
	}
}
