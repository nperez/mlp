package mlp

import (
	"log"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/initializers"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"nickandperla.net/mlp/config"
)

type MLP struct {
	nn.Module
	Layers         []nn.StandardModel
	TrainingConfig config.TrainingConfig
}

func NewMLP(layerConf []config.LayerConf) *MLP {
	layers := make([]nn.StandardModel, 0)
	for i, conf := range layerConf {
		if i != 0 {
			if i != len(layerConf)-1 {
				if layerConf[i-1].OutputCount != conf.InputCount {
					panic("Misconfigured layers. OutputCount must equal InputCount of the next layer")
				}
			}
		}

		lin := linear.New[float32](conf.InputCount, conf.OutputCount)

		initialized := false

		layers = append(layers, lin)

		for _, act := range conf.Activation {
			name, err := activation.Activation(act)
			if err != nil {
				log.Fatalf("Activation(%s) Error: %v", act, err)
			}
			switch name {
			case activation.LeakyReLU:
				if !initialized {
					initialized = true
					initializers.XavierNormal(lin.W.Value(), initializers.Gain(name), rand.NewLockedRand(42))
				}
				layers = append(layers, activation.New(name, nn.NewParam(mat.NewScalar[float32](0.001))))
			case activation.Tanh:
				if !initialized {
					initialized = true
					initializers.XavierNormal(lin.W.Value(), initializers.Gain(name), rand.NewLockedRand(42))
				}
				fallthrough
			default:
				layers = append(layers, activation.New(name))
			}
		}
	}

	return &MLP{
		Layers: layers,
	}
}

func (m *MLP) Forward(input ...ag.Node) []ag.Node {
	return nn.Forward(m.Layers)(input...)
}
