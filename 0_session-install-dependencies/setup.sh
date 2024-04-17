#!/bin/bash

if [[ "$USE_ONLY_CPU" == true || "$USE_ONLY_CPU" == "true" ]]; then
    echo "installing CPU only version of llama-cpp"
    pip install llama-cpp-python
else
    echo "installing GPU version of llama-cpp"
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
fi

pip install -r 0_session-install-dependencies/requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"