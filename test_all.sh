#!/bin/bash

# TWEET
echo "======================= TWEET ======================="
python main.py test crf tweet emotion model_weights/crf/tweet-emotion/crf_0.03-0.001
python main.py test crf tweet srl model_weights/crf/tweet-srl/crf_0.001-0.001

# REMAN
echo "======================= REMAN ======================="
python main.py test crf reman emotion model_weights/crf/reman-emotion/crf_0.001-0.001
python main.py test crf reman srl model_weights/crf/reman-srl/crf_0.001-0.001