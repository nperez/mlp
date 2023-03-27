package main

import (
	"flag"

	"nickandperla.net/mlp/trainer"
)

var trainingConf *string = flag.String("config", "./train.toml", "The config file to use. Defaults to './train.toml'")

func main() {

	flag.Parse()

	trainer := trainer.NewTrainerFromConfigPath(*trainingConf)
	trainer.Train()
	trainer.SaveModel()

}
