kenlm/build/bin/build_binary xls-r-300m-sv/language_model/5gram_correct.arpa xls-r-300m-sv/language_model/5gram.bin
rm xls-r-300m-sv/language_model/5gram_correct.arpa

import subprocess
import shlex
subprocess.call(shlex.split('./test.sh param1 param2'))

1. Mejorar el dataset
	1.1 Crear el dataset, con el preprocesamiento que ya tengamos
2. Pegarle esto a un modelo en serio
	2.1 Credenciales en ambiente
	
https://hub.docker.com/r/jhonparra18/pytorch_kubeflow
https://github.com/statscol/hf-asr-comp

