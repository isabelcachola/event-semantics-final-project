# Event Semantics Final Project: Emotions and SRLs across domains

## Reading in data
Download the datasets and unzip to one `data` directory. This code assumed the same directory structure as the unziped data.

```{python}
from data import RemanData, TweetData

reman_data = RemanData([path/to/datadir], [train|test|dev|None])
reman_data.read_data()

tweet_data = TweetData([path/to/datadir], [train|test|dev|None])
tweet_data.read_data()
```
Both `RemanData` and `TweetData` take the data directory and datasplit as parameters. Use `split=None` to read full dataset.

In the Twitter dataset, annotators has the option to mark `tweeter` as the experiencer, rather than annotating a span of text. 
For these examples, the span indices are `(-1, -1)`. 

Current normalization mappings are in `mapping.py`. All emotions are mapping to one of:
- anger
- fear
- trust
- disgust
- joy
- sadness
- surprise
- anticipation
- None

All SRL labels are normalized to "experiencer", "target", or None (emotion srl is inherent to emotion label).

Emotion labels and SRLs that are normalized to None are skipped.