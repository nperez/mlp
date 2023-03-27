package config

import (
	"log"

	"github.com/nlpodyssey/spago/gd"
)

var optLookup = map[string]int{
	"SGD":     gd.SGD,
	"AdaGrad": gd.AdaGrad,
	"Adam":    gd.Adam,
	"RAdam":   gd.RAdam,
	"RMSProp": gd.RMSProp,
	"Lamb":    gd.Lamb,
}

func GetOptimizerMethodName(opt string) int {
	if enum, ok := optLookup[opt]; ok {
		return enum
	} else {
		log.Fatalf("Unknown optimizer: %s", opt)
		return 0
	}
}
