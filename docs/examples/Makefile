EXAMPLES := $(wildcard */.)
EXAMPLES_MD=$(patsubst %, %/README.md, $(EXAMPLES))

all: $(EXAMPLES_MD)

%/README.md: %/README.ipynb
	jupyter nbconvert \
		$? \
		--ClearOutputPreprocessor.enabled=True \
		--to markdown \
		--output README.md

.PHONY: all
