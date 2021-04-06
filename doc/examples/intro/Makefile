.SUFFIXES: .ipynb .md

.ipynb.md:
	jupyter nbconvert \
		$< \
		--ClearOutputPreprocessor.enabled=True \
		--to markdown \
		--output $@
